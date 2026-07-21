"""Unit tests for scripts/analyze_frontier_nanogpt.py (program #7)."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "rm_analyze_frontier_nanogpt", REPO_ROOT / "scripts" / "analyze_frontier_nanogpt.py"
)
afn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(afn)


class TestCrossing:
    def test_upward_crossing_interpolated(self):
        mean_loss = {0.025: 3.310, 0.05: 3.300, 0.1: 3.320}
        e = afn.crossing_from_means(mean_loss)
        assert e["valley_lr"] == 0.05
        assert e["floor"] == pytest.approx(3.310)
        # crossing between 0.05 (3.300 <= floor) and 0.1 (3.320 > floor):
        # frac = (3.310-3.300)/(3.320-3.300) = 0.5 in log-lr
        assert e["lr_cross"] == pytest.approx(
            math.exp((math.log(0.05) + math.log(0.1)) / 2), rel=1e-4
        )
        assert e["shoulder"] == 0.05
        assert e["low_lr_penalty"] == pytest.approx(0.010)

    def test_low_side_never_crosses(self):
        # rising loss on the LOW side must not produce a crossing
        mean_loss = {0.025: 3.40, 0.05: 3.30, 0.1: 3.302, 0.141: 3.303}
        e = afn.crossing_from_means(mean_loss)
        assert e["lr_cross"] is None  # never exceeds floor on the high side
        assert e["shoulder"] == 0.141

    def test_shoulder_extends_below_floor(self):
        mean_loss = {0.025: 3.30, 0.05: 3.305, 0.1: 3.308, 0.141: 3.35}
        e = afn.crossing_from_means(mean_loss)
        assert e["valley_lr"] == 0.025
        assert e["lr_cross"] is not None
        assert 0.1 < e["lr_cross"] < 0.141
        assert e["shoulder"] == 0.1


class TestAlpha:
    def test_planted_exponent(self):
        crossings = {c: 0.08 * (c / 8) ** 0.35 for c in (2, 4, 8, 16)}
        fit = afn.alpha_fit(crossings)
        assert fit["alpha"] == pytest.approx(0.35, abs=1e-9)


class TestEndToEnd:
    def _write(self, d, chunks, lr, seed, final, max_steps=None):
        f = d / f"nanogpt_seed{seed}_2026072{chunks:02d}T{seed}{int(lr*1e4)}.json"
        f.write_text(json.dumps({
            "seed": seed,
            "config": {
                "path": f"sweeps/frontier_nanogpt_c{chunks}/x.yaml",
                "contents": {"nanogpt": {
                    "chunks_per_step": chunks, "muon_lr": lr,
                    "max_steps": max_steps,
                }},
            },
            "metrics": {"final_val_loss": final},
        }))

    def test_full_report(self, tmp_path):
        # planted: valley at rung index growing with chunks; crossing ~B^0.35
        for chunks in (2, 4, 8, 16):
            star = 0.06 * (chunks / 8) ** 0.35
            for lr in afn.RUNGS:
                # V-shaped in log-lr around star, 0.03 loss per log2 unit up
                dist = abs(math.log(lr / star)) / math.log(2)
                final = 3.30 + 0.03 * dist
                for seed in (1720, 1721):
                    self._write(tmp_path, chunks, lr, seed, final)
        # a smoke run that must be excluded
        self._write(tmp_path, 8, 0.05, 1999, 9.9, max_steps=12)
        rep = afn.build_report(tmp_path)
        assert set(rep["arms"]) == {"2", "4", "8", "16"}
        fit = rep["alpha_fit"]
        assert fit["n_points"] == 4
        assert fit["alpha"] == pytest.approx(0.35, abs=0.08)  # rung-limited
        assert rep["valley_monotone_nondecreasing"] is True
        assert rep["p3_no_cliff"] is True
        md = afn.to_markdown(rep)
        assert "alpha" in md and "tokens/step" in md

    def test_dedupe_and_missing_finals(self, tmp_path):
        self._write(tmp_path, 8, 0.05, 1720, 3.30)
        self._write(tmp_path, 8, 0.05, 1720, 3.31)  # later file wins
        table = afn.load_runs(tmp_path)
        assert table[8][0.05] == [pytest.approx(3.31)]
