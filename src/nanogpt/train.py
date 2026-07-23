"""Training loop for the WP0.2 nanogpt port (record + gradient accumulation).

Record source (see src/nanogpt/__init__.py): the 2025-07-12_BosAlign
validation script, ``0c5449cc-....txt`` lines 568-783 ("int main").

PORT CHANGES IN THIS FILE — the complete list:

T1. **Gradient accumulation.** The record does one forward/backward per
    optimizer step (RECORD:763-764); the port does ``accum_factor`` of them
    and scales each micro-loss by ``1/accum_factor``. Arithmetic and proof of
    exactness: ``src/nanogpt/config.py`` module docstring. At
    ``device_count == 8`` the factor is 1 and this loop is the record's.
T2. **``assert world_size == 8`` relaxed.** The validation script has no such
    assert (the 07/13 retiming script does, at its line 574); the port instead
    *checks* that ``device_count`` divides 8 and that tokens/step is
    unchanged, which is the property the assert existed to protect.
T3. **Config-driven** total steps / target loss / seed / device count
    (record: literals at RECORD:574, README target 3.28, no seed at all,
    ``world_size`` from the env).
T4. **Seeding.** The record seeds nothing (its 20 runs differ only by
    nondeterministic init); the port seeds torch/numpy/random from
    ``config.seed`` so a run is reproducible and seed-paired comparisons are
    possible. Seed policy: CLAUDE.md ground rule 2.
T5. **Metrics + checkpointing.** The record prints to a log; the port records
    the same val trace into the standard results JSON and (optionally)
    checkpoints for the spot tier. No effect on the computation.
T6. **The profiler block** of the validation script (its lines 700-712, a
    second 10-step warmup wrapped in ``torch.profiler``) is dropped; kernel
    warmup runs once. State is restored afterwards exactly as the record does
    (RECORD:714-717), so this is wall-time-only.
T7. **``torch.empty(1, device="cuda").backward()``** (RECORD:15) runs here
    instead of at import.
T9. **Wave-1 tail machinery** (prereg reports/wave1-anneal-decomposition-
    prereg.md §0; config: ``nanogpt.tail`` + ``nanogpt.fork_from``). With the
    default config (tail.mode none, accumulate false, fork_from null) every
    branch added for T9 is dead and the loop is the pre-Wave-1 port. Active
    pieces: cross-config prefix forking (hot-fingerprint + stable-plateau
    guarded), spike-gated streaming iterate means, the schedule-free tail
    (gradients at y, stock optimizer on z, validation at both xbar and z),
    and the batch-ramp tail (variable chunks/step at constant LR, matched
    token budget).
T8. **Process-group bootstrap.** The record (RECORD:592-598) requires
    ``torchrun`` — it reads ``os.environ["RANK"]`` etc. and lets
    ``init_process_group`` rendezvous through the env store. The port defaults
    the triple to ``(0, 1, 0)`` and, at ``world_size == 1`` with no
    ``MASTER_ADDR`` in the environment, rendezvouses through an in-process
    ``dist.HashStore``. A single-device run therefore needs **no** distributed
    environment variables. Under ``torchrun`` (any ``world_size``) the env is
    present and the behaviour is exactly the record's.

Everything else — architecture, optimizer instantiation, LR/momentum/window
schedules, data pipeline, validation protocol — is the record.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from src.nanogpt.config import ConfigError, NanoGPTConfig
from src.nanogpt.data import RecordDataGenerator
from src.nanogpt.model import GPT, next_multiple_of_n
from src.nanogpt.optim import DistAdam, Muon
from src.nanogpt.record_log import steps_to_target
from src.nanogpt.tail import (
    ChunkBuffer,
    ScheduleFreeTail,
    TailAccumulators,
    ramp_chunk_schedule,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ------------------------------------------------------------------ helpers


def get_lr(step: int, cfg: NanoGPTConfig) -> float:
    """RECORD:670-674 — stable then decay, floor at ``min_lr_frac``."""
    x = step / cfg.num_iterations  # progress in training
    assert 0 <= x <= 1
    w = min((1 - x) / cfg.cooldown_frac, 1.0)  # 1 -> 0
    return w * 1.0 + (1 - w) * cfg.min_lr_frac


def window_size_blocks_value(step: int, cfg: NanoGPTConfig) -> int:
    """RECORD:678-684 — linearly increase the block-wise sliding window."""
    x = step / cfg.num_iterations
    assert 0 <= x <= 1
    return next_multiple_of_n(cfg.window_warmup_max * x, n=128) // 128


def _rank_world() -> Tuple[int, int, int]:
    """(rank, world_size, local_rank) from the torchrun env; (0, 1, 0) alone."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_path(cfg: NanoGPTConfig, rank: int) -> Path:
    directory = Path(cfg.checkpoint.dir)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    # cfg.config_fingerprint() keys the file to the trajectory-shaping config:
    # sweep variants sharing (seed, iterations) must never share a checkpoint
    # (program-#7 incident — variants replayed a sibling's finished run).
    tag = (
        f"seed{cfg.seed}_it{cfg.num_iterations}_d{cfg.device_count}"
        f"_{cfg.config_fingerprint()}_rank{rank}"
    )
    return directory / f"nanogpt_{tag}.pt"


def _resolve_fork_path(cfg: NanoGPTConfig) -> Path:
    # "{seed}" lets one sweep config fork each --seed run from its own
    # seed-paired prefix checkpoint (the fork guard cross-checks the seed
    # stored inside the checkpoint either way).
    path = Path(str(cfg.fork_from).format(seed=cfg.seed))
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_fork(state: Dict[str, Any], cfg: NanoGPTConfig, path: Path) -> None:
    """Fork guard (prereg §0.2): a checkpoint from ANOTHER config may seed
    this run only when the shared trajectory is provably identical — same hot
    fingerprint and seed, and every step before the fork on both configs'
    stable-LR plateaus. Extends the program-#7 collision guard rather than
    weakening it: normal resume still requires the full fingerprint."""
    missing = [k for k in ("hot_fingerprint", "seed", "stable_through", "step") if k not in state]
    if missing:
        raise RuntimeError(
            f"fork_from checkpoint {path.name} lacks fork metadata {missing}; "
            "it predates the Wave-1 checkpoint format and cannot be fork-validated"
        )
    if state["hot_fingerprint"] != cfg.hot_fingerprint():
        raise RuntimeError(
            f"fork_from checkpoint {path.name} has hot fingerprint "
            f"{state['hot_fingerprint']!r} but this config is "
            f"{cfg.hot_fingerprint()!r}; refusing to fork a different "
            "stable-phase trajectory (program-#7 collision guard, prereg §0.2)"
        )
    if int(state["seed"]) != cfg.seed:
        raise RuntimeError(
            f"fork_from checkpoint {path.name} was seeded {state['seed']}, "
            f"this run is seeded {cfg.seed}; forked arms must be seed-paired"
        )
    fork_step = int(state["step"])
    if fork_step - 1 > int(state["stable_through"]):
        raise RuntimeError(
            f"fork_from checkpoint {path.name} is at step {fork_step} but its "
            f"source config left the stable LR plateau after step "
            f"{state['stable_through']}; forks must branch on the plateau"
        )
    if fork_step - 1 > cfg.stable_through_step:
        raise RuntimeError(
            f"fork at step {fork_step} but this config's stable plateau ends "
            f"at step {cfg.stable_through_step}; the forked run would "
            "misattribute decayed prefix steps to its own schedule"
        )
    if cfg.tail.mode != "none" and fork_step != cfg.tail.start_step:
        raise RuntimeError(
            f"tail.mode {cfg.tail.mode!r} declares tail.start_step "
            f"{cfg.tail.start_step} but the fork checkpoint is at step "
            f"{fork_step}; the tail must begin exactly at the fork"
        )


# ------------------------------------------------------------------- runner


def run_nanogpt(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """``experiment: nanogpt`` entrypoint (registered in scripts/run.py).

    Multi-GPU: launch under ``torchrun --nproc_per_node=D``; only rank 0
    writes a results JSON (others return ``_no_results_write``).
    """
    cfg = NanoGPTConfig.from_config(config)
    rank, world_size, local_rank = _rank_world()

    if world_size != cfg.device_count:
        raise ConfigError(
            f"nanogpt.device_count={cfg.device_count} but WORLD_SIZE={world_size}. "
            "The token batch per optimizer step is derived from device_count; a "
            "mismatch would silently change it. Launch with "
            f"`torchrun --nproc_per_node={cfg.device_count}` or fix the config."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
        # RECORD:15 (PORT CHANGE T7): prevents a bug on some systems.
        torch.empty(1, device=device, requires_grad=True).backward()
    else:
        backend = "gloo"

    owns_pg = not dist.is_initialized()
    if owns_pg:
        # PORT CHANGE T8 (RECORD:592-598). The record reads RANK/WORLD_SIZE/
        # LOCAL_RANK from ``os.environ[...]`` (KeyError if absent) and calls
        # ``dist.init_process_group(backend="nccl", device_id=device)``, which
        # then resolves rank/world_size/MASTER_* from the env again — i.e. the
        # record's script only runs under ``torchrun``. That is correct for the
        # record (it is always 8 ranks) but it made a *single-device* run of the
        # port fail unless the caller exported RANK/LOCAL_RANK/WORLD_SIZE/
        # MASTER_ADDR/MASTER_PORT by hand.
        #
        # ``_rank_world()`` already defaults to (0, 1, 0) when the env is empty,
        # so the only missing piece is a rendezvous. We build the store
        # explicitly for the single-rank case (no MASTER_ADDR/MASTER_PORT, no
        # free-port race) and pass rank/world_size explicitly in every case.
        # Under torchrun the env is present, ``rank``/``world_size`` are exactly
        # what the env-store would have produced, and behaviour is unchanged.
        pg_kwargs: Dict[str, Any] = dict(backend=backend, rank=rank, world_size=world_size)
        if backend == "nccl":
            pg_kwargs["device_id"] = device
        if world_size == 1 and "MASTER_ADDR" not in os.environ:
            pg_kwargs["store"] = dist.HashStore()
        else:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(**pg_kwargs)
        dist.barrier()

    try:
        metrics = _train(cfg, device, rank, world_size)
    finally:
        if owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    if rank != 0:
        metrics["_no_results_write"] = True
    return metrics


def _build(cfg: NanoGPTConfig, device: torch.device, rank: int, world_size: int):
    """RECORD:627-651 — model, param split, optimizers, initial_lr."""
    _set_seed(cfg.seed)  # PORT CHANGE T4

    model: nn.Module = GPT(
        vocab_size=next_multiple_of_n(cfg.vocab_size, n=128),  # RECORD:627
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        model_dim=cfg.model_dim,
        max_seq_len=max(cfg.train_seq_len, cfg.val_seq_len),
        # PORT CHANGE P2: the RECORD's world size, so `scalars` is the record's
        # tensor at any device count.
        world_size=cfg.record_world_size,
        use_fp8=(cfg.precision_mode == "fp8"),
        attention_impl=cfg.attention_impl,
        head_chunk_rows=cfg.head_chunk_rows,  # PORT CHANGE P5 (32 GB memory path)
    ).to(device)
    for m in model.modules():  # RECORD:628-630
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    if world_size > 1:
        for param in model.parameters():  # RECORD:631-632
            dist.broadcast(param.detach(), 0)

    # RECORD:635-638 — parameter split
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    model._embed_params = embed_params  # for the fp32 grad-accum probe
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # RECORD:644-645 — instantiation values (NOT the class defaults)
    optimizer1 = DistAdam(
        scalar_params + head_params + embed_params,
        lr=cfg.adam_lr, betas=tuple(cfg.adam_betas), eps=cfg.adam_eps,
        weight_decay=cfg.adam_weight_decay, rank=rank, world_size=world_size,
    )
    optimizer2 = Muon(
        hidden_matrix_params,
        lr=cfg.muon_lr, momentum=cfg.muon_momentum, weight_decay=cfg.muon_weight_decay,
        rank=rank, world_size=world_size,
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:  # RECORD:647-649
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return model, optimizers, optimizer2


def _train(cfg: NanoGPTConfig, device: torch.device, rank: int, world_size: int) -> Dict[str, Any]:
    model, optimizers, muon = _build(cfg, device, rank, world_size)
    tempo_probe = None
    if cfg.tempo_probe:
        # PORT CHANGE P6 (program #8): passive serial-alignment measurement
        # inside Muon.step. Each rank probes only the matrices it owns; the
        # metrics carry this rank's rows (world_size 1 locally -> all of them).
        from src.nanogpt.tempo_probe import TempoProbe

        tempo_probe = TempoProbe(
            subset=cfg.tempo_probe_subset,
            flush_every=cfg.tempo_probe_flush_every,
        )
        muon.tempo_probe = tempo_probe
    raw_model = model

    # PORT CHANGE T9 (Wave 1): tail machinery. All None/inert on the default
    # config. `tail_named` is the raw model's parameter dict; in-place writes
    # through it are seen by the compiled wrapper (shared storage).
    tail_cfg = cfg.tail
    tail_named = None
    tail_acc = None
    sf: Optional[ScheduleFreeTail] = None
    if tail_cfg.mode != "none" or tail_cfg.accumulate:
        tail_named = dict(raw_model.named_parameters())
    if tail_cfg.accumulate:
        tail_acc = TailAccumulators(tail_cfg, tail_named)

    @lru_cache(maxsize=None)
    def window_blocks(step: int) -> torch.Tensor:
        return torch.tensor(window_size_blocks_value(step, cfg), dtype=torch.int32, device=device)

    if cfg.compile:
        model = torch.compile(model, dynamic=False)  # RECORD:686

    accum = cfg.accum_factor

    # docs/nanogpt-port.md §6.1 probe (default off, flagged as a deviation).
    # Only meaningful when there is more than one micro-batch to accumulate.
    fp32_embed_accum = None
    if cfg.fp32_embed_grad_accum and accum > 1:
        fp32_embed_accum = {
            p: torch.zeros_like(p, dtype=torch.float32) for p in raw_model._embed_params
        }

    def train_loader() -> RecordDataGenerator:
        return RecordDataGenerator(
            cfg.train_files,
            local_batch_size=cfg.train_seq_len,
            # effective_chunks == record_world_size unless the program-#7
            # chunks_per_step axis is set (the generator is generic in it).
            record_world_size=cfg.effective_chunks,
            device_count=cfg.device_count,
            rank=rank,
            align_to_bos=cfg.train_align_to_bos,
            device=device,
        )

    # ---- kernel warmup (RECORD:688-717; PORT CHANGE T6 drops the profiler)
    start_step = 0
    val_curve: List[Dict[str, float]] = []
    training_time_ms = 0.0
    ckpt_path = _checkpoint_path(cfg, rank)
    resumed = False
    forked_from: Optional[str] = None
    loader_state: Optional[Dict[str, int]] = None

    if cfg.checkpoint.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Belt-and-braces vs the fingerprinted filename: a checkpoint may
        # only continue a run of the identical trajectory-shaping config.
        found = state.get("config_fingerprint")
        if found != cfg.config_fingerprint():
            raise RuntimeError(
                f"checkpoint {ckpt_path.name} carries config fingerprint "
                f"{found!r} but this run is {cfg.config_fingerprint()!r}; "
                "refusing to resume a different configuration's trajectory "
                "(program-#7 collision guard)"
            )
        raw_model.load_state_dict(state["model"])
        for opt, opt_state in zip(optimizers, state["optimizers"]):
            opt.load_state_dict(opt_state)
        start_step = int(state["step"])
        val_curve = list(state["val_curve"])
        training_time_ms = float(state["training_time_ms"])
        loader_state = state["loader"]
        if tail_acc is not None:
            tail_acc.load_state_dict(state.get("tail_accumulators"))
        resumed = True
    elif cfg.fork_from is not None:
        # PORT CHANGE T9: cross-config prefix fork (prereg §0.2). Own resume
        # above takes precedence so a babysitter retry continues this run's
        # own trajectory rather than restarting the tail.
        fork_path = _resolve_fork_path(cfg)
        if not fork_path.exists():
            raise RuntimeError(f"fork_from checkpoint {fork_path} does not exist")
        state = torch.load(fork_path, map_location=device, weights_only=False)
        _validate_fork(state, cfg, fork_path)
        raw_model.load_state_dict(state["model"])
        for opt, opt_state in zip(optimizers, state["optimizers"]):
            opt.load_state_dict(opt_state)
        start_step = int(state["step"])
        val_curve = list(state["val_curve"])
        training_time_ms = float(state["training_time_ms"])
        loader_state = state["loader"]
        forked_from = str(fork_path)
    elif cfg.warmup_steps > 0:
        warm = train_loader()
        # RECORD:690-691 — save the initial state so warmup isn't cheating.
        initial_state = dict(
            model=copy.deepcopy(raw_model.state_dict()),
            optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
        )
        for _ in range(cfg.warmup_steps):
            for inputs, targets in warm.next_step():
                (model(inputs, targets, window_blocks(1)) / accum).backward()
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
        raw_model.load_state_dict(initial_state["model"])
        for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
            opt.load_state_dict(opt_state)
        del warm, initial_state

    loader = train_loader()
    if loader_state is not None:
        loader.load_state_dict(loader_state)

    def validate(step: int, window_step: Optional[int] = None) -> float:
        """RECORD:729-748 — same fixed val_tokens, same fixed chunking.

        ``window_step`` (T9, batch_ramp only): the record-equivalent step for
        the sliding-window schedule when the loop step axis is compressed.
        """
        window_step = step if window_step is None else window_step
        model.eval()
        val_steps = cfg.val_tokens // (cfg.record_world_size * cfg.val_seq_len)
        val_loader = RecordDataGenerator(
            cfg.val_files,
            local_batch_size=cfg.val_seq_len,
            record_world_size=cfg.record_world_size,
            device_count=cfg.device_count,
            rank=rank,
            align_to_bos=cfg.val_align_to_bos,
            device=device,
        )
        val_loss = torch.zeros((), device=device)
        n_chunks = 0
        with torch.no_grad():
            for _ in range(val_steps):
                for inputs, targets in val_loader.next_step():
                    val_loss = val_loss + model(inputs, targets, window_blocks(window_step))
                    n_chunks += 1
        val_loss = val_loss / n_chunks
        del val_loader
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        model.train()
        return float(val_loss.item())

    # ---- training (RECORD:719-779) ---------------------------------------
    # PORT CHANGE T9: batch_ramp compresses the step axis — precompute the
    # deterministic per-step chunk counts, the record-equivalent step before
    # each tail step (for the window schedule + analysis), and which loop
    # steps validate (the record's val marks, mapped onto the ramp by
    # record-equivalent progress so arms compare at matched token counts).
    ramp_ks: Optional[List[int]] = None
    ramp_eq: Optional[List[float]] = None
    ramp_val_steps: set = set()
    chunk_buffer: Optional[ChunkBuffer] = None
    if tail_cfg.mode == "batch_ramp":
        ramp_ks = ramp_chunk_schedule(cfg)
        rws = cfg.effective_chunks
        ramp_eq = [tail_cfg.start_step]
        for k in ramp_ks:
            ramp_eq.append(ramp_eq[-1] + k / rws)
        train_steps = tail_cfg.start_step + len(ramp_ks)
        if cfg.val_loss_every > 0:
            for i in range(1, len(ramp_ks) + 1):
                lo, hi = ramp_eq[i - 1], ramp_eq[i]
                first_mark = (int(lo // cfg.val_loss_every) + 1) * cfg.val_loss_every
                if first_mark <= hi:
                    ramp_val_steps.add(tail_cfg.start_step + i)
    else:
        train_steps = cfg.num_iterations if cfg.max_steps is None else min(cfg.num_iterations, cfg.max_steps)
    cum_tokens = start_step * cfg.tokens_per_step
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(start_step, train_steps + 1):
        last_step = step == train_steps
        in_tail = tail_cfg.mode != "none" and step >= tail_cfg.start_step
        ramp_i = step - tail_cfg.start_step if ramp_ks is not None else None
        # record-equivalent step for the window schedule (ramp only)
        eq_step = int(ramp_eq[ramp_i]) if ramp_eq is not None and in_tail else step

        if ramp_ks is not None:
            do_val = last_step or step in ramp_val_steps
        else:
            do_val = last_step or (cfg.val_loss_every > 0 and step % cfg.val_loss_every == 0)
        if do_val:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            loss_value = validate(step, window_step=eq_step)
            entry = {
                "step": step,
                "tokens": cum_tokens,
                "val_loss": loss_value,
                "train_time_ms": training_time_ms,
            }
            if ramp_eq is not None and in_tail:
                entry["eq_step"] = round(ramp_eq[ramp_i], 2)
            if sf is not None:
                # T9 (schedule_free): the run's primary readout is the Polyak
                # average xbar (prereg §0.4); the raw iterate z is kept as a
                # secondary trace.
                sf.swap_in_xbar()
                entry["val_loss"] = validate(step, window_step=eq_step)
                sf.swap_back()
                entry["val_loss_z"] = loss_value
                loss_value = entry["val_loss"]
            val_curve.append(entry)
            if rank == 0:
                print(
                    f"step:{step}/{train_steps} val_loss:{loss_value:.4f} "
                    f"train_time:{training_time_ms:.0f}ms "
                    f"step_avg:{training_time_ms / max(step, 1):.2f}ms",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # ---- TRAINING SECTION (RECORD:762-776) + PORT CHANGES T1, T9 -----
        if tail_cfg.mode == "schedule_free" and step >= tail_cfg.start_step and sf is None:
            sf = ScheduleFreeTail(tail_cfg, tail_named)  # xbar initialized at z
        sf_active = sf is not None
        if sf_active:
            sf.to_y()  # gradients evaluated at the interpolate y

        track_loss = tail_acc is not None and step >= tail_cfg.start_step
        step_loss = torch.zeros((), device=device) if track_loss else None
        if ramp_ks is not None and in_tail:
            if chunk_buffer is None:
                chunk_buffer = ChunkBuffer(loader)
            k_chunks = ramp_ks[ramp_i]
            micro_iter = chunk_buffer.next_chunks(k_chunks)
            loss_scale = k_chunks
            step_tokens = k_chunks * cfg.train_seq_len
        else:
            micro_iter = loader.next_step()
            loss_scale = accum
            step_tokens = cfg.tokens_per_step
        for inputs, targets in micro_iter:
            # micro-loss scaled by 1/loss_scale: see config.py accumulation
            # math (loss_scale == accum == the record's arithmetic whenever
            # the T9 ramp is inactive).
            loss = model(inputs, targets, window_blocks(eq_step)) / loss_scale
            if step_loss is not None:
                step_loss = step_loss + loss.detach()
            loss.backward()
            # docs/nanogpt-port.md §6.1 probe: drain each micro-batch's bf16
            # embedding grad into an fp32 master buffer and clear `p.grad`, so
            # the running sum never touches bf16.
            if fp32_embed_accum is not None:
                for p in raw_model._embed_params:
                    if p.grad is not None:
                        fp32_embed_accum[p].add_(p.grad.float())
                        p.grad = None
        cum_tokens += step_tokens
        # ...then write the fp32 total back once, in the record's dtype: one
        # rounding for the whole step instead of one per micro-batch. Done
        # after the loop (not on a `micro == accum - 1` index) so the buffer is
        # always drained exactly once per step, whatever the loader yields.
        if fp32_embed_accum is not None:
            for p, buf in fp32_embed_accum.items():
                p.grad = buf.to(p.dtype)
                buf.zero_()
        if sf_active:
            sf.to_z()  # restore the iterate z; the stock optimizer updates z
        for opt in optimizers:  # RECORD:766-768 (+T9 tail LR pin)
            for group in opt.param_groups:
                lr_factor = tail_cfg.kappa if in_tail else get_lr(step, cfg)
                group["lr"] = group["initial_lr"] * lr_factor
        frac = min(step / cfg.momentum_warmup_steps, 1)  # RECORD:769
        for group in muon.param_groups:  # RECORD:770-771
            group["momentum"] = (1 - frac) * cfg.momentum_start + frac * cfg.momentum_end
        for opt in optimizers:  # RECORD:773-774
            opt.step()
        model.zero_grad(set_to_none=True)  # RECORD:776
        if sf_active:
            sf.update_average()
        if track_loss:
            tail_acc.observe(step, float(step_loss.item()))

        if cfg.checkpoint.every_steps and (step + 1) % cfg.checkpoint.every_steps == 0:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(
                {
                    "step": step + 1,
                    "config_fingerprint": cfg.config_fingerprint(),
                    # T9 fork metadata (prereg §0.2)
                    "hot_fingerprint": cfg.hot_fingerprint(),
                    "seed": cfg.seed,
                    "stable_through": cfg.stable_through_step,
                    "model": raw_model.state_dict(),
                    "optimizers": [opt.state_dict() for opt in optimizers],
                    "loader": loader.state_dict(),
                    "val_curve": val_curve,
                    "training_time_ms": training_time_ms + 1000 * (time.perf_counter() - t0),
                    "tail_accumulators": tail_acc.state_dict() if tail_acc is not None else None,
                },
                tmp,
            )
            tmp.rename(ckpt_path)

    # Successful completion: the checkpoint is spent. A surviving completed
    # checkpoint is a resume trap (program-#7 incident) and 2 GB of dead disk.
    if not cfg.checkpoint.keep_on_success:
        ckpt_path.unlink(missing_ok=True)
        ckpt_path.with_suffix(".tmp").unlink(missing_ok=True)

    steps = [int(p["step"]) for p in val_curve]
    losses = [float(p["val_loss"]) for p in val_curve]
    n_steps = steps_to_target(steps, losses, cfg.target_val_loss)

    # ---- PORT CHANGE T9: tail artifact + metrics --------------------------
    tail_metrics: Dict[str, Any] = {}
    if (tail_acc is not None or sf is not None) and rank == 0:
        art_dir = Path(tail_cfg.artifact_dir)
        if not art_dir.is_absolute():
            art_dir = REPO_ROOT / art_dir
        art_dir.mkdir(parents=True, exist_ok=True)
        art_path = art_dir / (
            f"nanogpt_tail_seed{cfg.seed}_{cfg.config_fingerprint()}_{tail_cfg.mode}.pt"
        )
        artifact: Dict[str, Any] = {
            "seed": cfg.seed,
            "config_fingerprint": cfg.config_fingerprint(),
            "hot_fingerprint": cfg.hot_fingerprint(),
            "tail_config": dataclasses.asdict(tail_cfg),
        }
        if tail_acc is not None:
            artifact.update(tail_acc.artifact())
        if sf is not None:
            artifact.update(sf.artifact())
        tmp = art_path.with_suffix(".tmp")
        torch.save(artifact, tmp)
        tmp.rename(art_path)
        try:
            tail_metrics["tail_artifact"] = str(art_path.relative_to(REPO_ROOT))
        except ValueError:  # artifact_dir outside the repo (tests)
            tail_metrics["tail_artifact"] = str(art_path)
    if tail_acc is not None:
        tail_metrics["tail_accumulators"] = tail_acc.summary()
        tail_metrics["tail_gate_log"] = tail_acc.gate_log
    if sf is not None:
        tail_metrics["tail_sf_t"] = sf.t
        z_vals = [p.get("val_loss_z") for p in val_curve if "val_loss_z" in p]
        tail_metrics["final_val_loss_z"] = z_vals[-1] if z_vals else None
    if ramp_ks is not None:
        tail_metrics["tail_ramp_steps"] = len(ramp_ks)
        tail_metrics["tail_ramp_chunks"] = ramp_ks
        tail_metrics["tail_budget_exhausted"] = True
        tail_metrics["tail_final_tokens"] = cum_tokens

    metrics: Dict[str, Any] = {
        "record": "2025-07-12_BosAlign",
        "record_log": "0c5449cc-0b01-4ecc-bec3-f46a09741d60.txt",
        "val_curve": val_curve,
        "final_val_loss": losses[-1] if losses else None,
        "target_val_loss": cfg.target_val_loss,
        "steps_to_target_loss": n_steps,
        "tokens_to_target_loss": None if n_steps is None else n_steps * cfg.tokens_per_step,
        "train_time_s": training_time_ms / 1000.0,
        "num_iterations": cfg.num_iterations,
        "train_steps_run": train_steps,
        "tokens_per_step": cfg.tokens_per_step,
        "accumulation_factor": accum,
        "device_count": cfg.device_count,
        "record_world_size": cfg.record_world_size,
        "precision_mode": cfg.precision_mode,
        "attention_impl": cfg.attention_impl,
        "compiled": cfg.compile,
        "fp32_embed_grad_accum": bool(fp32_embed_accum is not None),
        "record_faithful": cfg.record_faithful,
        "deviations": cfg.deviations(),
        "resumed_from_checkpoint": resumed,
        "forked_from": forked_from,
        **tail_metrics,
        "nanogpt_config": cfg.to_dict(),
    }
    if torch.cuda.is_available():
        metrics["peak_memory_mib"] = torch.cuda.max_memory_allocated() // 1024 // 1024
    if tempo_probe is not None:
        metrics["tempo_probe"] = tempo_probe.to_log()
    return metrics
