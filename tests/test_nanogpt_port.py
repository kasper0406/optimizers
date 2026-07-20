"""WP0.2 — tests for the modded-nanogpt record port (src/nanogpt).

These tests protect the two things that make the port worth having:

1. the **token batch per optimizer step is the record's**, at every supported
   device count (the accumulation arithmetic, unit-tested — not the training);
2. **no silent deviation**: unknown config keys are rejected, deviation flags
   are surfaced in metrics, and the record trace we compare against is parsed
   from the actual vendored log file.
"""

from __future__ import annotations

import importlib.util
import math
import json
import statistics as st
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import results_io
from src.nanogpt.config import (
    RECORD_LOG,
    RECORD_SEED_DISTRIBUTION,
    RECORD_TOKENS_PER_STEP,
    ConfigError,
    NanoGPTConfig,
    accumulation_factor,
    tokens_per_step,
)
from src.nanogpt.data import RecordDataGenerator, data_footprint_gb, find_batch_starts
from src.nanogpt.record_log import (
    censoring,
    collect_record_runs,
    cooldown_start,
    ensemble_deviation,
    parse_record_log,
    phase_decomposition,
    record_validation_traces,
    steps_to_target,
)

CONFIGS = [
    REPO_ROOT / "configs" / "wp02_nanogpt_repro.yaml",
    REPO_ROOT / "configs" / "wp02_nanogpt_repro_3seed.yaml",
]


# ------------------------------------------------- grad-accumulation math


@pytest.mark.parametrize(
    "device_count,expected_accum",
    [(1, 8), (2, 4), (4, 2), (8, 1)],
)
def test_accumulation_factor_reproduces_record_token_batch(device_count, expected_accum):
    """G = 8 / D, and D * G * seq_len is always the record's 393,216."""
    assert accumulation_factor(device_count) == expected_accum
    assert tokens_per_step(device_count) == RECORD_TOKENS_PER_STEP
    assert device_count * expected_accum * (48 * 1024) == RECORD_TOKENS_PER_STEP


def test_record_token_batch_constant_is_the_record_arithmetic():
    # RECORD:692 `world_size * args.seq_len`, RECORD:579 `seq_len = 48*1024`.
    assert RECORD_TOKENS_PER_STEP == 8 * 48 * 1024 == 393_216


@pytest.mark.parametrize("device_count", [3, 5, 6, 7, 16])
def test_non_dividing_device_count_is_refused(device_count):
    """Rounding a non-integer accumulation factor would silently change the
    token batch — the one thing this benchmark cannot tolerate."""
    with pytest.raises(ConfigError):
        accumulation_factor(device_count)


def test_device_count_zero_or_negative_refused():
    with pytest.raises(ConfigError):
        accumulation_factor(0)
    with pytest.raises(ConfigError):
        accumulation_factor(-1)


@pytest.mark.parametrize("device_count", [1, 2, 8])
def test_config_tokens_per_step_invariant(device_count):
    cfg = NanoGPTConfig(device_count=device_count)
    assert cfg.tokens_per_step == RECORD_TOKENS_PER_STEP
    assert cfg.accum_factor * device_count == 8
    # power-of-two accumulation keeps the 1/G loss rescale exact in binary FP
    assert cfg.accum_factor & (cfg.accum_factor - 1) == 0


def test_micro_loss_scaling_reproduces_the_record_gradient():
    """The accumulation identity, on numbers: averaging 8 chunk-sums equals
    accumulating G of them per device, dividing by G, and AVG-ing over D."""
    chunk_grads = [float(i + 1) for i in range(8)]  # per-chunk "sum" gradients
    record = sum(chunk_grads) / 8
    for device_count in (1, 2, 4, 8):
        accum = 8 // device_count
        per_rank = [
            sum(chunk_grads[m * device_count + r] / accum for m in range(accum))
            for r in range(device_count)
        ]
        ported = sum(per_rank) / device_count  # dist.ReduceOp.AVG
        assert ported == pytest.approx(record, rel=1e-12)


# ----------------------------------------------------------- config parsing


def test_defaults_are_the_record():
    cfg = NanoGPTConfig()
    assert cfg.num_iterations == 1750  # RECORD:574
    assert cfg.cooldown_frac == 0.45  # RECORD:575
    assert cfg.min_lr_frac == 0.05  # RECORD:674 — NOT the retiming script's 0.1
    assert cfg.val_tokens == 10485760  # RECORD:572
    assert cfg.train_seq_len == 48 * 1024  # RECORD:579
    assert cfg.val_seq_len == 4 * 64 * 1024  # RECORD:580
    assert (cfg.muon_lr, cfg.muon_momentum, cfg.muon_weight_decay) == (0.05, 0.95, 0.0)
    assert (cfg.adam_lr, tuple(cfg.adam_betas), cfg.adam_eps, cfg.adam_weight_decay) == (
        0.008, (0.8, 0.95), 1e-10, 0.0,
    )
    assert cfg.target_val_loss == 3.28
    # The ML defaults are the record's, so the config IS record-faithful at the
    # record's device count...
    assert NanoGPTConfig(device_count=8).record_faithful
    # ...but the dataclass default is `device_count: 1` (the port's testbed),
    # and D != 8 changes the gradient reduction order, so the bare default is
    # NOT record-faithful. See
    # test_record_faithful_requires_the_records_device_count.
    assert not cfg.record_faithful
    assert set(cfg.deviations()) == {"grad_accumulation"}


def test_unknown_config_key_is_rejected_not_ignored():
    with pytest.raises(ConfigError, match="unknown nanogpt config keys"):
        NanoGPTConfig.from_config({"nanogpt": {"muon_lrr": 0.05}})


def test_run_seed_overrides_nanogpt_seed():
    cfg = NanoGPTConfig.from_config({"seed": 1234, "nanogpt": {"seed": 1000}})
    assert cfg.seed == 1234


def test_deviation_flags_surface():
    cfg = NanoGPTConfig(precision_mode="bf16")
    assert not cfg.record_faithful
    assert "NOT RECORD-FAITHFUL" in cfg.deviations()["precision_mode"]

    cfg = NanoGPTConfig(attention_impl="sdpa")
    assert not cfg.record_faithful
    assert "NOT RECORD-FAITHFUL" in cfg.deviations()["attention_impl"]

    cfg = NanoGPTConfig(max_steps=5)
    assert not cfg.record_faithful
    assert "smoke run" in cfg.deviations()["max_steps"]

    cfg = NanoGPTConfig(device_count=1)
    assert "393216" in cfg.deviations()["grad_accumulation"]

    cfg = NanoGPTConfig(device_count=8, fp32_embed_grad_accum=True)
    assert not cfg.record_faithful
    assert "NOT RECORD-FAITHFUL" in cfg.deviations()["fp32_embed_grad_accum"]


@pytest.mark.parametrize("device_count", [1, 2, 4])
def test_record_faithful_requires_the_records_device_count(device_count):
    """REGRESSION (2026-07-20): `record_faithful` ignored `device_count`.

    A D<8 run reported `record_faithful: True` while its own `deviations` dict
    simultaneously reported `grad_accumulation` — the flag contradicted the
    deviation list. Accumulation preserves the token batch exactly but NOT the
    gradient reduction order, and at D<8 the bf16 embedding grads make that a
    genuine precision difference (docs/nanogpt-port.md §2: "the least-
    controlled numeric deviation in the port"). Numerics-changing means not
    record-faithful. Pinned in BOTH directions.
    """
    cfg = NanoGPTConfig(device_count=device_count)
    assert not cfg.record_faithful, (
        f"D={device_count} != the record's world size 8 must never be "
        f"record-faithful; deviations={cfg.deviations()}"
    )
    # the flag and the deviation list must agree
    assert "grad_accumulation" in cfg.deviations()

    # ...and the other direction: D == 8 with no other flag IS the record.
    faithful = NanoGPTConfig(device_count=8)
    assert faithful.record_faithful, faithful.deviations()
    assert "grad_accumulation" not in faithful.deviations()


def test_record_faithful_never_contradicts_the_deviation_list():
    """Any active *numerics-changing* deviation must clear the flag."""
    numerics_changing = [
        {"device_count": 1},
        {"precision_mode": "bf16"},
        {"attention_impl": "sdpa"},
        {"max_steps": 5},
        {"min_lr_frac": 0.1},
        {"num_iterations": 1000},
        {"fp32_embed_grad_accum": True},
    ]
    for kwargs in numerics_changing:
        cfg = NanoGPTConfig(**kwargs)
        assert not cfg.record_faithful, f"{kwargs} must not be record-faithful"
        assert cfg.deviations(), f"{kwargs} must appear in deviations()"


def test_invalid_precision_and_attention_modes_refused():
    with pytest.raises(ConfigError):
        NanoGPTConfig(precision_mode="fp16")
    with pytest.raises(ConfigError):
        NanoGPTConfig(attention_impl="naive")


def test_val_tokens_must_split_over_devices():
    with pytest.raises(ConfigError):
        NanoGPTConfig(val_tokens=10485760 + 1)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_are_the_record_modulo_declared_deviations(path):
    """The repro configs are the record's ML on a smaller device count.

    They are NOT `record_faithful` — they run at D=1, and D != 8 changes the
    gradient reduction order (see
    `test_record_faithful_requires_the_records_device_count`). What must hold
    is that the token batch, the schedules and the numerics knobs are the
    record's, and that `grad_accumulation` is the ONLY declared deviation.
    """
    raw = yaml.safe_load(path.read_text())
    assert raw["experiment"] == "nanogpt"
    assert raw["seed"] >= 1000, "configs carry dev seeds only (CLAUDE.md rule 2)"
    cfg = NanoGPTConfig.from_config(raw)
    assert cfg.tokens_per_step == RECORD_TOKENS_PER_STEP
    assert cfg.num_iterations == 1750
    assert cfg.precision_mode == "fp8" and cfg.attention_impl == "flex"
    assert set(cfg.deviations()) == {"grad_accumulation"}, cfg.deviations()
    # ...and the same config at the record's device count IS record-faithful.
    at_eight = NanoGPTConfig.from_config(
        {**raw, "nanogpt": {**raw["nanogpt"], "device_count": 8}}
    )
    assert at_eight.record_faithful, at_eight.deviations()


def test_fp32_embed_probe_config_is_one_variable_off_the_repro_config():
    """The §6.1 probe must differ from the repro config in exactly one knob."""
    repro = yaml.safe_load((REPO_ROOT / "configs" / "wp02_nanogpt_repro.yaml").read_text())
    probe_path = REPO_ROOT / "configs" / "wp02_nanogpt_fp32embed.yaml"
    probe = yaml.safe_load(probe_path.read_text())

    assert probe["seed"] == 1701, "the probe re-runs the seed being diagnosed"
    cfg = NanoGPTConfig.from_config(probe)
    assert cfg.fp32_embed_grad_accum is True
    assert cfg.tokens_per_step == RECORD_TOKENS_PER_STEP
    # a diagnostic, never a reproduction: both flags must be declared
    assert not cfg.record_faithful
    assert set(cfg.deviations()) == {"grad_accumulation", "fp32_embed_grad_accum"}

    # exactly one nanogpt knob differs from the repro config (modulo the seed,
    # which lives at run level and is the point of the probe being paired)
    a = NanoGPTConfig.from_config({**repro, "seed": 1701}).to_dict()
    b = cfg.to_dict()
    assert {k for k in a if a[k] != b[k]} == {"fp32_embed_grad_accum"}

    text = probe_path.read_text()
    assert "0.006" in text and "PRE-REGISTERED" in text, (
        "the probe config must carry its pre-registered read"
    )


def test_configs_restate_the_record_optimizer_values():
    """A drift in the pinned optimizer hyperparameters must fail a test, not
    silently ride along in a YAML."""
    raw = yaml.safe_load(CONFIGS[0].read_text())["nanogpt"]
    assert raw["muon_lr"] == 0.05 and raw["muon_momentum"] == 0.95
    assert raw["adam_lr"] == 0.008 and raw["adam_betas"] == [0.8, 0.95]
    assert raw["adam_eps"] == 1e-10
    assert raw["val_tokens"] == 10485760


def test_experiment_is_registered_in_run_py():
    spec = importlib.util.spec_from_file_location(
        "rm_run_py_for_test", REPO_ROOT / "scripts" / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "nanogpt" in module.EXPERIMENT_REGISTRY


# ------------------------------------------------------- record log parsing


def test_record_log_parser_on_the_vendored_log():
    trace = parse_record_log(RECORD_LOG)
    assert trace.total_steps == 1750
    # val every 125 steps (RECORD:576) over 1750 steps, plus step 0 = 15 points
    assert trace.steps == [0] + list(range(125, 1751, 125))
    assert len(trace.val_losses) == 15
    assert trace.val_losses[0] > 10  # untrained model at step 0
    assert trace.val_losses == sorted(trace.val_losses, reverse=True), "loss must decrease"
    assert trace.final_val_loss == pytest.approx(3.2784, abs=1e-4)
    assert trace.tokens[-1] == 1750 * RECORD_TOKENS_PER_STEP
    assert 170 < trace.final_train_time_s < 180  # record: ~173 s on 8xH100


def test_record_directory_splits_into_validation_runs_and_the_retiming_run():
    """The 20 n=20-distribution logs share one script; the 07/13 retiming log
    has a different (ML-differing) script and must not be mistaken for them."""
    groups = collect_record_runs()
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 20], sizes
    traces = record_validation_traces()
    assert len(traces) == 20
    finals = sorted(t.final_val_loss for t in traces)
    assert finals == sorted(RECORD_SEED_DISTRIBUTION), "must match the record README's accs"
    assert st.mean(finals) == pytest.approx(3.2791, abs=5e-5)
    assert st.stdev(finals) == pytest.approx(0.0013, abs=5e-5)


def test_retiming_script_has_the_other_lr_floor():
    """Documents *why* the port follows the n=20 script: the retiming script
    the candidates report cites has min-LR factor 0.1, not the record's 0.05."""
    retiming = RECORD_LOG.parent / "c1fd8a38-bb9f-45c4-8af0-d37f70c993f3.txt"
    assert "(1 - w) * 0.1" in retiming.read_text(errors="replace")
    assert "(1 - w) * 0.05" in RECORD_LOG.read_text(errors="replace")


def test_steps_to_target_interpolates_and_refuses_to_extrapolate():
    steps = [0, 100, 200, 300]
    losses = [5.0, 4.0, 3.5, 3.2]
    got = steps_to_target(steps, losses, 3.28)
    assert 200 < got < 300
    # linear between (200, 3.5) and (300, 3.2): (3.5-3.28)/(3.5-3.2) = 0.7333
    assert got == pytest.approx(200 + 100 * (3.5 - 3.28) / (3.5 - 3.2))
    assert steps_to_target(steps, losses, 3.0) is None  # never reached
    assert steps_to_target(steps, losses, 5.0) == 0.0  # already there at step 0


# ------------------------------------------------------------ data pipeline


def _write_shard(path: Path, num_tokens: int, bos_every: int) -> None:
    """A minimal fineweb .bin: 256-int32 header then uint16 tokens."""
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = 20240520, 1, num_tokens
    tokens = np.arange(num_tokens, dtype=np.uint16) % 1000
    tokens[::bos_every] = 50256
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.tobytes())


def test_find_batch_starts_returns_bos_aligned_non_overlapping_starts(tmp_path):
    shard = tmp_path / "fineweb_train_000001.bin"
    _write_shard(shard, num_tokens=40000, bos_every=64)
    from src.nanogpt.data import _load_data_shard

    tokens = _load_data_shard(shard)
    starts, span = find_batch_starts(tokens, pos=0, world_size=8, local_batch_size=1000, max_batch_span=32000)
    assert len(starts) == 8
    assert all(int(tokens[s]) == 50256 for s in starts), "every start is a BOS token"
    assert all(b - a >= 1000 for a, b in zip(starts, starts[1:])), "chunks do not overlap"
    assert span >= 8 * 1000


@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
def test_ranks_and_micro_steps_cover_the_records_chunks_exactly_once(tmp_path, device_count):
    """The port's chunk→(device, micro-step) map is a permutation of the
    record's 8 chunks — same tokens, same optimizer step."""
    shard = tmp_path / "fineweb_train_000001.bin"
    _write_shard(shard, num_tokens=200_000, bos_every=64)
    pattern = str(tmp_path / "fineweb_train_*.bin")

    def make(dc, rank):
        return RecordDataGenerator(
            pattern, local_batch_size=1000, record_world_size=8,
            device_count=dc, rank=rank, align_to_bos=True,
        )

    def signature(t):
        return (int(t[0]), int(t[-1]), int(t.sum()))

    port = [make(device_count, r) for r in range(device_count)]
    # The record itself is device_count=8, accum=1: one chunk per rank.
    ref = [make(8, r) for r in range(8)]

    for _ in range(3):
        got = []
        for gen in port:
            micro = list(gen.next_step())
            assert len(micro) == 8 // device_count, "accum micro-batches per step"
            got += [signature(inp) for inp, _ in micro]
        want = [signature(next(iter(gen.next_step()))[0]) for gen in ref]
        assert sorted(got) == sorted(want), "port chunks != the record's 8 chunks"
        assert len(set(got)) == 8, "a chunk was consumed twice"
        # every rank tracks the same logical file position as the record
        assert {g.pos for g in port} == {g.pos for g in ref}


def test_loader_state_round_trips(tmp_path):
    shard = tmp_path / "fineweb_train_000001.bin"
    _write_shard(shard, num_tokens=200_000, bos_every=64)
    pattern = str(tmp_path / "fineweb_train_*.bin")
    gen = RecordDataGenerator(pattern, local_batch_size=1000, record_world_size=8,
                              device_count=1, rank=0, align_to_bos=True)
    list(gen.next_step())
    state = gen.state_dict()
    after = [int(i[0]) for i, _ in gen.next_step()]

    gen2 = RecordDataGenerator(pattern, local_batch_size=1000, record_world_size=8,
                               device_count=1, rank=0, align_to_bos=True)
    gen2.load_state_dict(state)
    assert [int(i[0]) for i, _ in gen2.next_step()] == after


def test_missing_data_shards_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_fineweb"):
        RecordDataGenerator(str(tmp_path / "nope_*.bin"), local_batch_size=8,
                            record_world_size=8, device_count=1, rank=0, align_to_bos=False)


def test_data_footprint():
    # 10 shards (9 train + 1 val) of 100M uint16 tokens ~= 2 GB
    assert data_footprint_gb(10) == pytest.approx(2.0, abs=0.02)


def test_fetch_script_shard_plan():
    spec = importlib.util.spec_from_file_location(
        "rm_fetch_fineweb_for_test", REPO_ROOT / "scripts" / "fetch_fineweb.py"
    )
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    cfg = NanoGPTConfig()
    n = fetch.shards_needed(cfg)
    # 1750 steps x 393,216 tokens = 688M; the record repo's own guidance for a
    # run of this length is 9 chunks (vendor/modded-nanogpt/README.md).
    assert n == 9
    names = fetch.shard_names(n)
    assert names[0] == "fineweb_val_000000.bin"
    assert names[1] == "fineweb_train_000001.bin" and len(names) == 10


# ------------------------------------------------------------------- model


def _tiny_model(**kwargs):
    from src.nanogpt.model import GPT

    return GPT(
        # model_dim must equal num_heads * head_dim (128) — the record's
        # 768 = 6 x 128 (RECORD:627); the value-embedding view_as at
        # RECORD:350 assumes it.
        vocab_size=640, num_layers=12, num_heads=1, model_dim=128, max_seq_len=512,
        world_size=8, use_fp8=False, attention_impl="sdpa", **kwargs
    )


def test_tiny_model_forward_backward_on_cpu():
    """Structural smoke test: the record's module graph runs end to end.

    Uses the NOT-RECORD-FAITHFUL sdpa/bf16 paths — FlexAttention has no CPU
    backward and FP8 needs an H100-class GPU. This checks wiring, not numerics.
    """
    torch.manual_seed(1000)
    model = _tiny_model()
    model.train()
    T = 256
    inputs = torch.randint(0, 640, (T,), dtype=torch.int32)
    targets = torch.randint(0, 640, (T,), dtype=torch.int64)
    window = torch.tensor(2, dtype=torch.int32)

    loss = model(inputs, targets, window)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    # the lm_head is zero-initialised (RECORD:417) but must still get a gradient
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_scalars_tensor_matches_the_record_shape_at_any_device_count():
    """PORT CHANGE P2: `pad` uses the RECORD's world size, so the `scalars`
    parameter is the record's 64-entry tensor whatever hardware we run on."""
    model = _tiny_model()
    assert model.scalars.numel() == 12 * 5 + ((-12 * 5) % 8) == 64
    assert model.scalars.lr_mul == 5.0
    assert model.lm_head.weight.lr_mul == 27.5
    assert all(p.lr_mul == 75.0 for p in model.embed.parameters())


def test_train_loss_uses_sum_reduction_eval_uses_mean():
    """The accumulation math assumes sum reduction in training (RECORD:503)."""
    torch.manual_seed(1000)
    model = _tiny_model()
    inputs = torch.randint(0, 640, (256,), dtype=torch.int32)
    targets = torch.randint(0, 640, (256,), dtype=torch.int64)
    window = torch.tensor(2, dtype=torch.int32)
    model.train()
    with torch.no_grad():
        train_loss = float(model(inputs, targets, window))
    model.eval()
    with torch.no_grad():
        eval_loss = float(model(inputs, targets, window))
    assert train_loss == pytest.approx(eval_loss * 256, rel=1e-3)


def test_lr_and_window_schedules_match_the_record():
    from src.nanogpt.train import get_lr, window_size_blocks_value

    cfg = NanoGPTConfig()
    assert get_lr(0, cfg) == pytest.approx(1.0)  # stable phase
    assert get_lr(int(cfg.num_iterations * (1 - cfg.cooldown_frac)), cfg) == pytest.approx(1.0)
    assert get_lr(cfg.num_iterations, cfg) == pytest.approx(cfg.min_lr_frac)  # floor 0.05
    # window: 128 -> 1792 tokens, in 128-token blocks (RECORD:683)
    assert window_size_blocks_value(0, cfg) == 1
    # next_multiple_of_n(1728, n=128) = 1792 tokens = 14 blocks at the end
    assert window_size_blocks_value(cfg.num_iterations, cfg) == 14


# -------------------------------------------------- end-to-end (CPU, stubbed)


class _SingleRankDist:
    """Single-rank stand-ins for the collectives the record's optimizers use.

    The record's Muon/DistAdam call ``reduce_scatter``/``all_gather`` with
    ``ReduceOp.AVG``, which gloo does not implement — so a CPU end-to-end run
    is impossible without stubs. At world_size 1 every one of these collectives
    is the identity, so the stubs are exact, and the optimizers themselves stay
    byte-for-byte the record's code.
    """

    @staticmethod
    def _done():
        fut = torch.futures.Future()
        fut.set_result(None)

        class _Work:
            def get_future(self_inner):
                return fut

        return _Work()

    @classmethod
    def install(cls, monkeypatch, stub_is_initialized: bool = True):
        """Patch the AVG collectives.

        ``stub_is_initialized=False`` leaves ``dist.is_initialized`` alone so
        ``run_nanogpt`` really builds its own single-rank process group — that
        is the code path a 1-GPU cloud run takes, and the only way to test that
        it needs no ``RANK``/``WORLD_SIZE``/``MASTER_ADDR`` in the environment.
        """
        import torch.distributed as dist

        def reduce_scatter(output, input_list, op=None, async_op=False):
            output.copy_(input_list[0])
            return cls._done()

        def reduce_scatter_tensor(output, input_tensor, op=None, async_op=False):
            output.copy_(input_tensor)
            return cls._done()

        def all_gather(tensor_list, tensor, async_op=False):
            tensor_list[0].copy_(tensor)
            return cls._done()

        def all_gather_into_tensor(output, input_tensor, async_op=False):
            output.copy_(input_tensor)
            return cls._done()

        def all_reduce(tensor, op=None, async_op=False):
            return cls._done()

        if stub_is_initialized:
            monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "reduce_scatter", reduce_scatter)
        monkeypatch.setattr(dist, "reduce_scatter_tensor", reduce_scatter_tensor)
        monkeypatch.setattr(dist, "all_gather", all_gather)
        monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather_into_tensor)
        monkeypatch.setattr(dist, "all_reduce", all_reduce)


def test_training_loop_end_to_end_on_cpu(tmp_path, monkeypatch):
    """Two optimizer steps through the real loop: warmup, 8x accumulation,
    validation, checkpoint, metrics. Tiny + NOT record-faithful by construction
    (sdpa/bf16/no-compile); this checks wiring, not numerics."""
    _SingleRankDist.install(monkeypatch)
    # scoped, not global: leaking suppress_errors=True into the rest of the
    # session would silence exactly the dynamo failures
    # test_training_loop_end_to_end_on_cpu_with_compile exists to catch.
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)

    for i in (1, 2):
        _write_shard(tmp_path / f"fineweb_train_{i:06d}.bin", num_tokens=400_000, bos_every=64)
    _write_shard(tmp_path / "fineweb_val_000000.bin", num_tokens=400_000, bos_every=64)

    from src.nanogpt.train import run_nanogpt

    config = {
        "experiment": "nanogpt",
        "seed": 1000,
        "nanogpt": {
            "train_files": str(tmp_path / "fineweb_train_*.bin"),
            "val_files": str(tmp_path / "fineweb_val_*.bin"),
            "device_count": 1,
            "num_layers": 12, "num_heads": 1, "model_dim": 128, "vocab_size": 50257,
            "train_seq_len": 256, "val_seq_len": 256, "val_tokens": 2048,
            "num_iterations": 2, "val_loss_every": 1, "warmup_steps": 1,
            "compile": False, "precision_mode": "bf16", "attention_impl": "sdpa",
            "checkpoint": {"dir": str(tmp_path / "ckpt"), "every_steps": 1, "resume": False},
        },
    }
    metrics = run_nanogpt(config, torch.device("cpu"))

    assert metrics["accumulation_factor"] == 8
    assert metrics["tokens_per_step"] == 8 * 256
    assert [p["step"] for p in metrics["val_curve"]] == [0, 1, 2]
    assert all(math.isfinite(p["val_loss"]) for p in metrics["val_curve"])
    assert metrics["val_curve"][1]["tokens"] == 8 * 256
    assert metrics["record_faithful"] is False
    assert set(metrics["deviations"]) >= {"precision_mode", "attention_impl", "num_iterations"}
    assert (tmp_path / "ckpt").exists(), "checkpoint was not written"

    # a resumed run picks up the loop where the checkpoint left off
    config["nanogpt"]["checkpoint"]["resume"] = True
    resumed = run_nanogpt(config, torch.device("cpu"))
    assert resumed["resumed_from_checkpoint"] is True


@pytest.mark.slow
def test_fp32_embed_grad_accum_probe_runs_and_changes_only_embedding_grads(tmp_path, monkeypatch):
    """The §6.1 probe: fp32 accumulation of the bf16 embedding grads.

    Verifies the flag (a) is off by default, (b) is reported in the metrics,
    (c) produces a finite run, and (d) actually changes the embedding gradient
    path — a run with the flag on must not be bit-identical to one without,
    while remaining close (it is one rounding instead of G).
    """
    _SingleRankDist.install(monkeypatch)
    # same scoping rationale as test_training_loop_end_to_end_on_cpu: DistAdam
    # .step is @torch.compile'd even at `compile: false`, and dynamo's
    # structured logging calls dist.get_rank() against our stubbed group.
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)
    monkeypatch.chdir(tmp_path)
    for i in (1, 2):
        _write_shard(tmp_path / f"fineweb_train_{i:06d}.bin", num_tokens=400_000, bos_every=64)
    _write_shard(tmp_path / "fineweb_val_000000.bin", num_tokens=400_000, bos_every=64)

    from src.nanogpt.train import run_nanogpt

    def config(fp32: bool):
        return {
            "experiment": "nanogpt",
            "seed": 1000,
            "nanogpt": {
                "train_files": str(tmp_path / "fineweb_train_*.bin"),
                "val_files": str(tmp_path / "fineweb_val_*.bin"),
                "device_count": 1,
                "num_layers": 12, "num_heads": 1, "model_dim": 128, "vocab_size": 50257,
                "train_seq_len": 256, "val_seq_len": 256, "val_tokens": 2048,
                "num_iterations": 2, "val_loss_every": 1, "warmup_steps": 1,
                "compile": False, "precision_mode": "bf16", "attention_impl": "sdpa",
                "fp32_embed_grad_accum": fp32,
                "checkpoint": {"dir": str(tmp_path / f"ckpt{int(fp32)}"),
                               "every_steps": 0, "resume": False},
            },
        }

    off = run_nanogpt(config(False), torch.device("cpu"))
    on = run_nanogpt(config(True), torch.device("cpu"))

    assert off["fp32_embed_grad_accum"] is False
    assert on["fp32_embed_grad_accum"] is True
    assert "fp32_embed_grad_accum" not in off["deviations"]
    assert "NOT RECORD-FAITHFUL" in on["deviations"]["fp32_embed_grad_accum"]
    assert all(math.isfinite(p["val_loss"]) for p in on["val_curve"])

    # step 0 is evaluated before any optimizer step -> identical either way
    assert on["val_curve"][0]["val_loss"] == pytest.approx(
        off["val_curve"][0]["val_loss"], abs=1e-9
    )
    # ...but after stepping, the embedding grads took a different arithmetic
    # path, so the curves must differ (and only slightly)
    assert on["final_val_loss"] != off["final_val_loss"]
    assert on["final_val_loss"] == pytest.approx(off["final_val_loss"], rel=0.05)


@pytest.mark.slow
def test_training_loop_end_to_end_on_cpu_with_compile(tmp_path, monkeypatch):
    """The record path's ``compile: true``, on CPU, with dynamo errors FATAL.

    WHY THIS TEST EXISTS. ``test_training_loop_end_to_end_on_cpu`` runs with
    ``compile: False`` and therefore missed a bug that killed *every*
    record-faithful GPU run at the first ``opt.step()``:

        torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: A view was
        created in no_grad mode and its base or another view of its base has
        been modified inplace with grad mode enabled.

    Root cause and fix: ``src/nanogpt/optim.py``, PORT CHANGE O1. The trigger is
    purely an autograd/dynamo aliasing property of ``DistAdam.step`` — it needs
    ``torch.compile``, not CUDA — so CPU reproduces it exactly. Revert O1 and
    this test fails with the message above; that is how it was verified.

    Two more things it pins:
      * ``torch._dynamo.config.suppress_errors`` is forced False, so a dynamo
        failure is an error and not a silent fallback to eager;
      * the run builds its own single-rank process group with **no**
        RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT in the environment
        (PORT CHANGE T8), which is how 1-GPU cloud runs are launched.

    STILL NOT COVERED HERE (GPU-only): FlexAttention block masks
    (``attention_impl: flex`` has no CPU backward), the FP8 head
    (``torch._scaled_mm`` / ``float8_e4m3fn``, H100-class only), inductor's CUDA
    codegen, and NCCL. See docs/nanogpt-port.md §6.
    """
    # Real process group, stubbed AVG collectives only.
    _SingleRankDist.install(monkeypatch, stub_is_initialized=False)
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    torch._dynamo.reset()

    _write_shard(tmp_path / "fineweb_train_000001.bin", num_tokens=20_000, bos_every=64)
    _write_shard(tmp_path / "fineweb_val_000000.bin", num_tokens=20_000, bos_every=64)

    from src.nanogpt.train import run_nanogpt

    config = {
        "experiment": "nanogpt",
        "seed": 1000,
        "nanogpt": {
            "train_files": str(tmp_path / "fineweb_train_*.bin"),
            "val_files": str(tmp_path / "fineweb_val_*.bin"),
            # record_world_size 1 -> accumulation factor 1: the accumulation
            # arithmetic is covered by the tests above; this one is about
            # compile, and 8 micro-batches would only make it 8x slower.
            "device_count": 1, "record_world_size": 1,
            "num_layers": 12, "num_heads": 1, "model_dim": 128, "vocab_size": 50257,
            "train_seq_len": 256, "val_seq_len": 256, "val_tokens": 256,
            "num_iterations": 1, "val_loss_every": 0, "warmup_steps": 1,
            "compile": True, "precision_mode": "bf16", "attention_impl": "sdpa",
            "checkpoint": {"dir": str(tmp_path / "ckpt"), "every_steps": 0, "resume": False},
        },
    }
    metrics = run_nanogpt(config, torch.device("cpu"))

    assert metrics["compiled"] is True
    assert [p["step"] for p in metrics["val_curve"]] == [1]
    assert math.isfinite(metrics["final_val_loss"])
    # the process group was created and torn down by run_nanogpt itself
    import torch.distributed as dist

    assert not dist.is_initialized()


def test_dist_adam_param_slice_is_not_a_no_grad_view(monkeypatch):
    """PORT CHANGE O1, isolated: the fast unit-level guard on the same bug.

    ``DistAdam.step`` must write into the parameter through a handle that is
    NOT an autograd differentiable view of a ``requires_grad`` leaf. If it is,
    the in-place writes rebase the view's history and any later read of
    ``.grad_fn``/``.is_leaf`` (which is what dynamo does when the collective
    graph-breaks) raises the no_grad-view RuntimeError.
    """
    from src.nanogpt.optim import DistAdam

    captured = {}

    def all_gather_into_tensor(output, input_tensor, async_op=False):
        captured["out"] = output
        captured["in"] = input_tensor
        output.copy_(input_tensor)
        return _SingleRankDist._done()

    import torch.distributed as dist

    _SingleRankDist.install(monkeypatch, stub_is_initialized=False)
    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather_into_tensor)
    # DistAdam.step is @torch.compile'd (RECORD:223) and dynamo's structured
    # logging calls dist.get_rank(), so a real single-rank group must exist.
    created_pg = not dist.is_initialized()
    if created_pg:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())

    try:
        p = torch.nn.Parameter(torch.ones(8, 4))
        p.grad = torch.full((8, 4), 0.5)
        opt = DistAdam([p], lr=0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
        opt.step()
    finally:
        if created_pg:
            dist.destroy_process_group()

    for name in ("out", "in"):
        t = captured[name]
        assert t._base is None or not t._base.requires_grad, (
            f"DistAdam handed the collective a differentiable view ({name}); "
            "this is the no_grad-view bug (src/nanogpt/optim.py PORT CHANGE O1)"
        )
        t.is_leaf  # must not raise the view-rebase RuntimeError
    # and the parameter really was updated in place
    assert not torch.allclose(p.detach(), torch.ones(8, 4))


# ------------------------------------------------------------ results schema


def test_synthetic_metrics_dict_is_schema_valid(tmp_path):
    cfg = NanoGPTConfig(device_count=1)
    metrics = {
        "record": "2025-07-12_BosAlign",
        "val_curve": [
            {"step": 0, "tokens": 0, "val_loss": 10.83, "train_time_ms": 0.0},
            {"step": 1750, "tokens": 1750 * RECORD_TOKENS_PER_STEP,
             "val_loss": 3.2788, "train_time_ms": 1.2e6},
        ],
        "final_val_loss": 3.2788,
        "target_val_loss": 3.28,
        "steps_to_target_loss": 1742.3,
        "tokens_to_target_loss": 1742.3 * RECORD_TOKENS_PER_STEP,
        "train_time_s": 1200.0,
        "tokens_per_step": cfg.tokens_per_step,
        "accumulation_factor": cfg.accum_factor,
        "device_count": cfg.device_count,
        "precision_mode": cfg.precision_mode,
        "attention_impl": cfg.attention_impl,
        "record_faithful": cfg.record_faithful,
        "deviations": cfg.deviations(),
        "nanogpt_config": cfg.to_dict(),
    }
    result = {
        "schema_version": results_io.SCHEMA_VERSION,
        "experiment": "nanogpt",
        "config": {"path": "configs/wp02_nanogpt_repro.yaml", "sha256": "0" * 64, "contents": {}},
        "git_sha": "a" * 40,
        "git_dirty": False,
        "seed": 1000,
        "gpu_type": "NVIDIA H100 80GB HBM3",
        "wall_time_s": 1500.0,
        "cost_usd": 1.23,
        "started_at": results_io.utc_now_iso(),
        "finished_at": results_io.utc_now_iso(),
        "metrics": metrics,
    }
    results_io.validate(result)
    out = results_io.write_result(result, tmp_path / "nanogpt_seed1000.json")
    assert json.loads(out.read_text())["metrics"]["tokens_per_step"] == RECORD_TOKENS_PER_STEP


def test_ensemble_deviation_separates_our_offset_from_record_seed_noise():
    """A run planted at the ensemble mean must read ~0 everywhere.

    Against a SINGLE record log the same run would show a deviation equal to
    that log's own seed noise — which is what the pre-fix analysis reported as
    its headline "max |dev|".
    """
    traces = record_validation_traces()
    assert len(traces) == 20
    steps = traces[0].steps
    mean_curve = [st.mean([t.loss_at_step(s) for t in traces]) for s in steps]

    rows = ensemble_deviation((steps, mean_curve), traces)
    assert [r.step for r in rows] == steps
    for r in rows:
        assert abs(r.deviation) < 1e-9

    # ...whereas one record log alone is off the ensemble mean by its own noise
    solo = ensemble_deviation((steps, traces[0].val_losses), traces)
    late = [r for r in solo if r.step == steps[-1]][0]
    assert abs(late.deviation) > 0

    # a planted constant offset is recovered in loss units, and its sigma
    # grows purely because the record's sd shrinks over training
    offset = 0.01
    shifted = ensemble_deviation((steps, [m + offset for m in mean_curve]), traces)
    for r in shifted:
        assert r.deviation == pytest.approx(offset, abs=1e-9)
    sized = [r for r in shifted if r.sigma is not None]
    assert sized[0].record_sd > sized[-1].record_sd  # yardstick shrinks
    assert sized[0].sigma < sized[-1].sigma  # ...so sigma grows at fixed offset


def test_record_ensemble_reproduces_the_reviewed_deviation_trajectory():
    """Pins the headline numbers for our seed-1701 run against the record."""
    results = sorted((REPO_ROOT / "results").glob("nanogpt_seed1701_*.json"))
    if not results:
        pytest.skip("seed-1701 result not synced")
    payload = json.loads(results[0].read_text())
    curve = payload["metrics"]["val_curve"]
    ours = ([int(p["step"]) for p in curve], [float(p["val_loss"]) for p in curve])
    rows = {r.step: r for r in ensemble_deviation(ours, record_validation_traces())}

    expected = {  # step: (deviation, sigma) — the adversarial review's table
        125: (-0.0036, -0.5), 500: (+0.0028, +0.6), 875: (+0.0090, +3.5),
        1250: (+0.0098, +6.4), 1750: (+0.0112, +8.4),
    }
    for step, (dev, sigma) in expected.items():
        assert rows[step].deviation == pytest.approx(dev, abs=5e-5), step
        assert rows[step].sigma == pytest.approx(sigma, abs=0.05), step

    final = rows[1750]
    assert final.n == 20
    assert final.our_loss > final.record_max  # outside the observed support
    assert final.our_loss - final.record_max == pytest.approx(0.0084, abs=5e-5)


def test_steps_to_target_over_the_record_is_censored_at_3_28():
    """6 of the record's 20 runs never reach 3.28 — the disclosure's basis."""
    traces = record_validation_traces()
    cens = censoring(traces, 3.28)
    assert cens.n_total == 20
    assert cens.n_unreached == 6
    assert cens.n_reached == 14
    assert cens.is_censored
    assert min(cens.unreached_finals) == pytest.approx(3.2801, abs=1e-4)
    assert max(cens.unreached_finals) == pytest.approx(3.2819, abs=1e-4)
    # the record's own published statistic is over the survivors only
    assert cens.survivor_mean == pytest.approx(1740.5, abs=0.1)
    assert cens.survivor_sd == pytest.approx(4.5, abs=0.1)
    # every unreached run's final sits ABOVE the target: censoring is one-sided
    assert all(f > 3.28 for f in cens.unreached_finals)

    # a target every run clears is uncensored
    assert not censoring(traces, 3.30).is_censored


def test_cooldown_start_matches_the_records_lr_schedule():
    assert cooldown_start(1750, 0.45) == pytest.approx(962.5)
    assert cooldown_start(1750, 0.0) == pytest.approx(1750.0)


def test_phase_decomposition_localises_a_planted_cooldown_deficit():
    """A deficit planted only after cooldown onset must land only there."""
    traces = record_validation_traces()
    steps = traces[0].steps
    mean_curve = [st.mean([t.loss_at_step(s) for t in traces]) for s in steps]

    # A LEVEL SHIFT that appears at the boundary and then stays constant is a
    # deficit incurred by the boundary and carried, not accrued during
    # cooldown. The decomposition must attribute it to the stable phase and
    # leave cooldown clean — it measures loss *removed per phase*, not the
    # standing offset.
    planted = [m + (0.01 if s >= 1000 else 0.0) for s, m in zip(steps, mean_curve)]
    pd = phase_decomposition((steps, planted), traces, 1750, 0.45)
    assert pd.cooldown_start_exact == pytest.approx(962.5)
    assert pd.cooldown_start_step == 1000  # nearest validation step
    assert pd.stable.deficit == pytest.approx(0.01, abs=1e-9)
    assert pd.cooldown.deficit == pytest.approx(0.0, abs=1e-9)
    assert pd.n_segments_below_record == 0

    # a run that simply removes less loss during cooldown shows it as a deficit
    slow = list(mean_curve)
    for i, s in enumerate(steps):
        if s >= 1000:
            slow[i] = mean_curve[i] + 0.002 * (s - 1000) / 750
    pd2 = phase_decomposition((steps, slow), traces, 1750, 0.45)
    assert pd2.stable.deficit == pytest.approx(0.0, abs=1e-9)
    assert pd2.cooldown.deficit == pytest.approx(0.002, abs=1e-6)
    assert pd2.cooldown.deficit_frac > 0
    # every cooldown segment underperforms -> unanimous sign test
    assert pd2.n_segments_below_record == len(pd2.cooldown_segments)
    assert pd2.sign_test_p == pytest.approx(2 / 2 ** len(pd2.cooldown_segments))
    assert all(s.ratio < 1 for s in pd2.cooldown_segments)

    # ...and an exactly-on-mean run shows no deficit and a balanced sign test
    pd3 = phase_decomposition((steps, mean_curve), traces, 1750, 0.45)
    assert pd3.cooldown.deficit == pytest.approx(0.0, abs=1e-9)
    assert pd3.n_segments_below_record == 0


def test_phase_decomposition_of_our_run_is_cooldown_concentrated():
    results = sorted((REPO_ROOT / "results").glob("nanogpt_seed1701_*.json"))
    if not results:
        pytest.skip("seed-1701 result not synced")
    curve = json.loads(results[0].read_text())["metrics"]["val_curve"]
    ours = ([int(p["step"]) for p in curve], [float(p["val_loss"]) for p in curve])
    pd = phase_decomposition(ours, record_validation_traces(), 1750, 0.45)

    assert pd.stable.deficit == pytest.approx(0.0085, abs=5e-5)
    assert pd.cooldown.deficit == pytest.approx(0.0027, abs=5e-5)
    # the point: as a share of the loss each phase removes, cooldown dominates
    assert pd.stable.deficit_frac == pytest.approx(0.0012, abs=1e-4)
    assert pd.cooldown.deficit_frac == pytest.approx(0.0120, abs=1e-4)
    assert pd.cooldown.deficit_frac > 5 * pd.stable.deficit_frac
    assert pd.n_segments_below_record == 6 == len(pd.cooldown_segments)
    assert pd.sign_test_p == pytest.approx(0.03125)


def test_project_cost_reconciliation_sums_the_results_dir(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "rm_analyze_nanogpt_costs", REPO_ROOT / "scripts" / "analyze_nanogpt.py"
    )
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)

    def write(name, experiment, cost):
        payload = {"experiment": experiment, "metrics": {}}
        if cost is not None:
            payload["cost_usd"] = cost
        (tmp_path / name).write_text(json.dumps(payload))

    write("a.json", "nanogpt", 1.6)
    write("b.json", "nanogpt", 1.6)
    write("c.json", "airbench", 0.8)
    write("d.json", "airbench", None)  # uncosted -> excluded, but counted
    (tmp_path / "not_a_result.json").write_text(json.dumps({"hello": "world"}))

    costs = analyze.project_costs(tmp_path)
    assert costs["total_usd"] == pytest.approx(4.0)
    assert costs["nanogpt_usd"] == pytest.approx(3.2)
    assert costs["n_costed"] == 3
    assert costs["n_missing"] == 1


def test_real_results_dir_cost_totals():
    """Pins the reconciled spend the WP0.2 report publishes."""
    spec = importlib.util.spec_from_file_location(
        "rm_analyze_nanogpt_costs2", REPO_ROOT / "scripts" / "analyze_nanogpt.py"
    )
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    costs = analyze.project_costs(REPO_ROOT / "results")
    assert costs["total_usd"] == pytest.approx(12.00, abs=0.01)
    assert costs["nanogpt_usd"] == pytest.approx(3.20, abs=0.01)


def test_analysis_script_runs_on_a_synthetic_results_file(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "rm_analyze_nanogpt_for_test", REPO_ROOT / "scripts" / "analyze_nanogpt.py"
    )
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)

    trace = parse_record_log(RECORD_LOG)
    payload = {
        "schema_version": 1,
        "experiment": "nanogpt",
        "config": {"path": "configs/wp02_nanogpt_repro.yaml", "sha256": "0" * 64, "contents": {}},
        "git_sha": "b" * 40,
        "git_dirty": False,
        "seed": 1000,
        "gpu_type": "NVIDIA H100 80GB HBM3",
        "wall_time_s": 1500.0,
        "cost_usd": 1.5,
        "started_at": results_io.utc_now_iso(),
        "finished_at": results_io.utc_now_iso(),
        "metrics": {
            "val_curve": [
                {"step": s, "tokens": s * RECORD_TOKENS_PER_STEP,
                 "val_loss": loss + 0.002, "train_time_ms": 1000.0 * s}
                for s, loss in zip(trace.steps, trace.val_losses)
            ],
            "final_val_loss": trace.final_val_loss + 0.002,
            "target_val_loss": 3.28,
            "steps_to_target_loss": 1745.0,
            "train_time_s": 1750.0,
            "tokens_per_step": RECORD_TOKENS_PER_STEP,
            "accumulation_factor": 8,
            "device_count": 1,
            "precision_mode": "fp8",
            "attention_impl": "flex",
            "record_faithful": True,
            "deviations": {},
        },
    }
    path = tmp_path / "nanogpt_seed1000.json"
    path.write_text(json.dumps(payload))
    report = analyze.build_report(analyze.load_runs([path]), record_validation_traces())
    assert "0.0020" in report  # the deviation we injected
    assert "Descriptive only" in report
    assert "steps→3.28" in report
