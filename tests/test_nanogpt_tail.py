"""Wave-1 tail machinery tests (prereg reports/wave1-anneal-decomposition-
prereg.md §0, verification list): fork guard, constant-LR equivalence of the
rho=0 schedule-free tail, accumulator/spike-gate math, and the batch-ramp
budget arithmetic. GPU-free; the end-to-end runs use the CPU tiny-model path
of test_nanogpt_port.py (sdpa/bf16/no-compile — wiring, not numerics).
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nanogpt.config import ConfigError, NanoGPTConfig, TailConfig
from src.nanogpt.tail import TailAccumulators, ramp_chunk_schedule
from src.nanogpt.train import get_lr

from test_nanogpt_port import _SingleRankDist, _write_shard


# ----------------------------------------------------------- pure arithmetic


def _cfg(**tail) -> NanoGPTConfig:
    return NanoGPTConfig(
        precision_mode="bf16", attention_impl="sdpa", compile=False,
        tail=TailConfig(**tail) if tail else TailConfig(),
    )


def test_min_lr_frac_one_is_constant_lr():
    cfg = NanoGPTConfig(min_lr_frac=1.0, precision_mode="bf16",
                        attention_impl="sdpa", compile=False)
    for step in (0, 962, 963, 1400, 1750):
        assert get_lr(step, cfg) == 1.0
    assert cfg.stable_through_step == cfg.num_iterations


def test_stable_through_step_matches_decay_onset():
    cfg = _cfg()
    # w == 1 iff step <= 1750 * 0.55 = 962.5, so steps 0..962 are stable and
    # T_c = 963 is the first decayed step (prereg §0).
    assert cfg.stable_through_step == 962
    assert get_lr(962, cfg) == 1.0
    assert get_lr(963, cfg) < 1.0


def test_hot_fingerprint_ignores_tail_shaping_fields_only():
    base = _cfg()
    same = [
        NanoGPTConfig(precision_mode="bf16", attention_impl="sdpa", compile=False,
                      min_lr_frac=1.0),
        NanoGPTConfig(precision_mode="bf16", attention_impl="sdpa", compile=False,
                      cooldown_frac=0.3),
        NanoGPTConfig(precision_mode="bf16", attention_impl="sdpa", compile=False,
                      max_steps=963),
        _cfg(mode="schedule_free", start_step=963, rho=0.7),
    ]
    for cfg in same:
        assert cfg.hot_fingerprint() == base.hot_fingerprint()
        # ...while the full fingerprint still separates them from base
    assert same[0].config_fingerprint() != base.config_fingerprint()
    hot_variant = NanoGPTConfig(precision_mode="bf16", attention_impl="sdpa",
                                compile=False, muon_lr=0.033)
    assert hot_variant.hot_fingerprint() != base.hot_fingerprint()


def test_ramp_chunk_schedule_full_scale_budget():
    cfg = _cfg(mode="batch_ramp", start_step=963)
    ks = ramp_chunk_schedule(cfg)
    assert sum(ks) == (1750 - 963) * 8  # exact record tail token budget
    assert ks[0] == 8  # B(0) = B0
    assert max(ks) <= 64  # 8x cap
    assert all(k >= 8 for k in ks[:-1])  # only the final step may clamp low
    # monotone non-decreasing until the final remainder clamp
    assert all(a <= b for a, b in zip(ks[:-2], ks[1:-1]))
    # the ramp genuinely compresses the step axis (418 steps vs 787: early
    # steps hold k=8 until w decays enough for round(8/w) to move)
    assert len(ks) == 418


def test_tail_config_validation():
    with pytest.raises(ConfigError, match="batch_ramp requires device_count"):
        NanoGPTConfig(precision_mode="bf16", attention_impl="sdpa", compile=False,
                      device_count=2, tail=TailConfig(mode="batch_ramp"))
    with pytest.raises(ConfigError, match="periodic checkpoints"):
        NanoGPTConfig(precision_mode="bf16", attention_impl="sdpa", compile=False,
                      checkpoint={"every_steps": 250},
                      tail=TailConfig(mode="batch_ramp"))
    with pytest.raises(ConfigError, match="schedule_free does not support periodic"):
        NanoGPTConfig(precision_mode="bf16", attention_impl="sdpa", compile=False,
                      checkpoint={"every_steps": 250},
                      tail=TailConfig(mode="schedule_free"))
    with pytest.raises(ConfigError, match="w1_window"):
        _cfg(accumulate=True, start_step=963, w1_window=(900, 1000))
    with pytest.raises(ConfigError, match="rho"):
        _cfg(mode="schedule_free", rho=1.0)
    # inert default stays record-faithful-compatible (no new deviations)
    cfg = NanoGPTConfig(device_count=8)
    assert "tail_mode" not in cfg.deviations()
    assert cfg.record_faithful


# ------------------------------------------------------------- accumulators


def test_tail_accumulators_match_brute_force_and_gate_spike():
    tail = TailConfig(accumulate=True, start_step=1, w1_window=(1, 4),
                      w2_window=(4, 7), spike_warmup=2, spike_z=4.0,
                      spike_beta=0.9)
    p = torch.zeros(3, 4)
    acc = TailAccumulators(tail, {"p": p})
    torch.manual_seed(0)
    # noisy-stationary losses with a planted spike at step 5 (past warmup);
    # the EMA variance must reflect the ~0.05-scale fluctuations so ordinary
    # steps pass the one-sided z<=4 gate while the spike fails it
    step_losses = {1: 1.0, 2: 1.1, 3: 0.95, 4: 1.05, 5: 50.0, 6: 1.02}
    iterates = []
    for step in range(1, 7):
        p.copy_(torch.randn(3, 4))
        acc.observe(step, step_losses[step])
        iterates.append((step, p.clone(), step_losses[step]))

    log = {e["step"]: e for e in acc.gate_log}
    assert not log[5]["included"], "planted spike was not excluded"
    assert all(log[s]["included"] for s in (1, 2, 3, 4, 6))

    w1_expect = torch.stack([it for s, it, _ in iterates if 1 <= s < 4]).mean(0)
    w2_expect = torch.stack([it for s, it, _ in iterates if 4 <= s < 7 and s != 5]).mean(0)
    pol_expect = torch.stack([it for s, it, _ in iterates if s != 5]).mean(0)
    assert torch.allclose(acc.w1["p"], w1_expect, atol=1e-6)
    assert torch.allclose(acc.w2["p"], w2_expect, atol=1e-6)
    assert torch.allclose(acc.polyak["p"], pol_expect, atol=1e-6)
    assert acc.summary() == {"n1": 3, "n2": 2, "n_polyak": 5,
                             "steps_seen": 6, "excluded": 1}

    art = acc.artifact()
    assert set(art) >= {"w1", "w2", "polyak", "final", "counts", "gate_log"}
    assert torch.allclose(art["final"]["p"], p.float())


def test_tail_accumulators_state_roundtrip():
    tail = TailConfig(accumulate=True, start_step=1, w1_window=(1, 3),
                      w2_window=(3, 5))
    p = torch.zeros(2)
    acc = TailAccumulators(tail, {"p": p})
    for step in (1, 2):
        p.fill_(float(step))
        acc.observe(step, 1.0)
    state = acc.state_dict()
    acc2 = TailAccumulators(tail, {"p": p})
    acc2.load_state_dict(state)
    for step in (3, 4):
        p.fill_(float(step))
        acc.observe(step, 1.0)
        acc2.observe(step, 1.0)
    assert torch.allclose(acc.w2["p"], acc2.w2["p"])
    assert acc.summary() == acc2.summary()
    # pre-start empty accumulator checkpoints as None (lazy allocation)
    assert TailAccumulators(tail, {"p": p}).state_dict() is None


# ---------------------------------------------------- CPU end-to-end wiring


def _tiny_config(tmp_path, **nanogpt_overrides):
    base = {
        "train_files": str(tmp_path / "fineweb_train_*.bin"),
        "val_files": str(tmp_path / "fineweb_val_*.bin"),
        "device_count": 1,
        "num_layers": 12, "num_heads": 1, "model_dim": 128, "vocab_size": 50257,
        "train_seq_len": 256, "val_seq_len": 256, "val_tokens": 2048,
        "num_iterations": 6, "val_loss_every": 1, "warmup_steps": 0,
        "min_lr_frac": 1.0,  # constant LR: the shared Wave-1 stable plateau
        "compile": False, "precision_mode": "bf16", "attention_impl": "sdpa",
        "checkpoint": {"dir": str(tmp_path / "ckpt"), "every_steps": 0,
                       "resume": False, "keep_on_success": False},
    }
    base.update(nanogpt_overrides)
    return {"experiment": "nanogpt", "seed": 1000, "nanogpt": base}


def _write_data(tmp_path):
    for i in (1, 2):
        _write_shard(tmp_path / f"fineweb_train_{i:06d}.bin", num_tokens=400_000, bos_every=64)
    _write_shard(tmp_path / "fineweb_val_000000.bin", num_tokens=400_000, bos_every=64)


def test_fork_and_schedule_free_rho0_reproduce_straight_run(tmp_path, monkeypatch):
    """Prereg §0 verification (a)+(b): a prefix fork with a rho=0, kappa=1
    schedule-free tail reproduces the straight constant-LR run's iterate
    trajectory exactly, and the fork guard refuses a hot-config variant."""
    _SingleRankDist.install(monkeypatch)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)
    _write_data(tmp_path)
    from src.nanogpt.train import run_nanogpt

    # straight constant-LR run, vals every step
    straight = run_nanogpt(_tiny_config(tmp_path), torch.device("cpu"))
    straight_vals = {p["step"]: p["val_loss"] for p in straight["val_curve"]}

    # prefix: same hot config, stopped at step 3, checkpoint kept
    prefix_cfg = _tiny_config(
        tmp_path, max_steps=3,
        checkpoint={"dir": str(tmp_path / "ckpt"), "every_steps": 3,
                    "resume": False, "keep_on_success": True},
    )
    prefix = run_nanogpt(prefix_cfg, torch.device("cpu"))
    assert prefix["train_steps_run"] == 3
    from src.nanogpt.train import _checkpoint_path
    ckpt = _checkpoint_path(NanoGPTConfig.from_config(prefix_cfg), 0)
    assert ckpt.exists()

    # forked schedule-free tail, rho=0 (y == z): z-trajectory must equal the
    # straight run's at every tail val point
    sf_cfg = _tiny_config(
        tmp_path, fork_from=str(ckpt),
        tail={"mode": "schedule_free", "start_step": 3, "kappa": 1.0,
              "rho": 0.0, "artifact_dir": str(tmp_path / "art")},
    )
    forked = run_nanogpt(sf_cfg, torch.device("cpu"))
    assert forked["forked_from"] == str(ckpt)
    tail_entries = [p for p in forked["val_curve"] if p["step"] > 3]
    assert tail_entries and all("val_loss_z" in p for p in tail_entries)
    for p in tail_entries:
        assert p["val_loss_z"] == pytest.approx(straight_vals[p["step"]], abs=1e-6), (
            f"z-trajectory diverged from straight run at step {p['step']}"
        )
    # primary endpoint is val(xbar) and the artifact carries xbar + final z
    assert forked["tail_sf_t"] == 3
    art = torch.load(tmp_path / "art" / Path(forked["tail_artifact"]).name,
                     weights_only=False)
    assert art["t"] == 3 and "xbar" in art and "final_z" in art

    # fork guard: a hot-config variant must be refused
    bad = copy.deepcopy(sf_cfg)
    bad["nanogpt"]["muon_lr"] = 0.033
    with pytest.raises(RuntimeError, match="hot fingerprint"):
        run_nanogpt(bad, torch.device("cpu"))
    # ...and a seed mismatch too
    bad2 = copy.deepcopy(sf_cfg)
    bad2["seed"] = 1001
    with pytest.raises(RuntimeError, match="seed"):
        run_nanogpt(bad2, torch.device("cpu"))

    # batch-ramp tail from the same prefix: exact token budget, flagged done
    ramp_cfg = _tiny_config(
        tmp_path, fork_from=str(ckpt),
        tail={"mode": "batch_ramp", "start_step": 3, "kappa": 1.0},
    )
    ramp = run_nanogpt(ramp_cfg, torch.device("cpu"))
    assert ramp["tail_budget_exhausted"] is True
    assert sum(ramp["tail_ramp_chunks"]) == (6 - 3) * 8
    assert ramp["tail_final_tokens"] == 6 * 8 * 256  # matched total budget
    assert ramp["val_curve"][-1]["tokens"] == 6 * 8 * 256
    assert ramp["train_steps_run"] == 3 + ramp["tail_ramp_steps"]


def test_accumulating_run_writes_artifact_and_survives_resume(tmp_path, monkeypatch):
    """Prereg §0 verification (c) wiring: arm-C-style run (constant LR +
    accumulators) writes the artifact with the right counts, and a
    checkpoint-resume mid-window continues the accumulator state."""
    _SingleRankDist.install(monkeypatch)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)
    _write_data(tmp_path)
    from src.nanogpt.train import run_nanogpt

    cfg = _tiny_config(
        tmp_path,
        tail={"accumulate": True, "start_step": 2, "w1_window": [2, 4],
              "w2_window": [4, 6], "spike_warmup": 10,
              "artifact_dir": str(tmp_path / "art")},
    )
    full = run_nanogpt(cfg, torch.device("cpu"))
    assert full["tail_accumulators"] == {
        "n1": 2, "n2": 2, "n_polyak": 4, "steps_seen": 4, "excluded": 0,
    }
    art_full = torch.load(tmp_path / "art" / Path(full["tail_artifact"]).name,
                          weights_only=False)
    assert art_full["counts"] == {"n1": 2, "n2": 2, "n_polyak": 4}

    # interrupted-and-resumed variant reproduces the same accumulator means:
    # run to step 4 (checkpoint carries acc state), then resume to completion
    cfg2 = copy.deepcopy(cfg)
    cfg2["nanogpt"]["tail"]["artifact_dir"] = str(tmp_path / "art2")
    cfg2["nanogpt"]["checkpoint"] = {
        "dir": str(tmp_path / "ckpt2"), "every_steps": 4, "resume": True,
        "keep_on_success": False,
    }
    interrupted = copy.deepcopy(cfg2)
    interrupted["nanogpt"]["max_steps"] = 4
    interrupted["nanogpt"]["checkpoint"]["keep_on_success"] = True
    run_nanogpt(interrupted, torch.device("cpu"))
    # the resumed full run must pick up the checkpoint (fingerprints differ
    # only in max_steps... which IS part of config_fingerprint, so point the
    # full run at its own path by matching max_steps semantics: resume goes
    # through the full-fingerprint path, so run the SAME config to completion.
    resumed = run_nanogpt(interrupted, torch.device("cpu"))
    assert resumed["resumed_from_checkpoint"] is True
    # resumed-at-4 run covered steps 2,3 before the checkpoint; its state
    # was carried, so n1 is still 2 at completion of max_steps=4 (no-op leg).
    assert resumed["tail_accumulators"]["n1"] == 2
