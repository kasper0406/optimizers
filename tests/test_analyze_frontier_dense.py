"""Unit tests for scripts/analyze_frontier_dense.py (program #6b)."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "rm_analyze_frontier_dense", REPO_ROOT / "scripts" / "analyze_frontier_dense.py"
)
afd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(afd)


class TestInterpolatedCrossing:
    def test_exact_midpoint_in_log_lr(self):
        # acc falls from floor+0.2 to floor-0.2 across the pair: frac = 0.5
        mean_acc = {0.4: 93.2, 0.6: 92.8}
        got = afd.interpolated_crossing(mean_acc, floor=93.0)
        assert got == pytest.approx(math.exp((math.log(0.4) + math.log(0.6)) / 2))

    def test_crossing_at_rung_returns_rung(self):
        mean_acc = {0.4: 93.0, 0.6: 92.0}
        assert afd.interpolated_crossing(mean_acc, floor=93.0) == pytest.approx(0.4)

    def test_first_crossing_wins_over_recovery(self):
        mean_acc = {0.4: 93.5, 0.5: 92.5, 0.6: 93.4, 0.7: 91.0}
        got = afd.interpolated_crossing(mean_acc, floor=93.0)
        assert got < 0.5  # interpolated inside (0.4, 0.5), not the later pair

    def test_lowest_rung_below_floor_undefined(self):
        assert afd.interpolated_crossing({0.4: 92.0, 0.6: 91.0}, floor=93.0) is None

    def test_no_crossing_in_band_undefined(self):
        assert afd.interpolated_crossing({0.4: 94.0, 0.6: 93.5}, floor=93.0) is None


class TestPeakReferencedShoulder:
    def test_scans_down_from_peak(self):
        mean_acc = {0.12: 63.0, 0.24: 70.6, 0.48: 72.7, 0.6: 70.0, 0.72: 68.3}
        # peak 72.7 @ 0.48, floor 71.7; 0.6 is below -> shoulder stays 0.48
        assert afd.peak_referenced_shoulder(mean_acc) == 0.48

    def test_extends_past_peak_while_above_floor(self):
        mean_acc = {0.24: 92.7, 0.36: 92.8, 0.48: 92.7, 0.6: 92.5, 0.72: 92.1, 0.96: 91.0}
        # peak 92.8 @ 0.36, floor 91.8; rungs up to 0.72 stay above
        assert afd.peak_referenced_shoulder(mean_acc) == 0.72

    def test_low_rungs_below_floor_do_not_matter(self):
        mean_acc = {0.12: 60.0, 0.48: 72.7, 0.6: 72.5}
        assert afd.peak_referenced_shoulder(mean_acc) == 0.6


class TestAlpha:
    def test_recovers_planted_exponent_exactly(self):
        crossings = {b: 0.5 * (b / 2000.0) ** 0.42 for b in (1000, 2000, 4000)}
        fit = afd.alpha_from_crossings(crossings)
        assert fit["alpha"] == pytest.approx(0.42, abs=1e-9)

    def test_undefined_crossings_dropped(self):
        assert afd.alpha_from_crossings({1000: None, 2000: 0.5}) is None


class TestEndToEnd:
    def _write(self, d, tag, seed, lr, acc, i):
        # tag folded into the fake timestamp so cells from different ladders
        # sharing a rung value cannot collide on filename
        u = abs(hash(tag)) % 1000
        f = d / f"airbench_instrumented_seed{seed}_2026072{i:02d}T{u:03d}{seed}{int(lr*100):03d}.json"
        f.write_text(json.dumps({
            "seed": seed,
            "config": {"path": f"sweeps/{tag}/{tag}__lr{lr}.yaml",
                       "contents": {"train": {"batch_size": 0, "epochs": 8},
                                    "probe_overrides": {"lr": lr}}},
            "metrics": {"tta_val_acc": acc, "steps": 200},
        }))

    def test_full_report_with_planted_surface(self, tmp_path):
        # plant acc(lr) = 94 - 2 * (lr / lr_star) pp with lr_star = c * B^0.5.
        # ref(B) is the anchor rung's acc, so the analytic floor crossing is
        # lr_star/2 + anchor per B; the machinery must recover those exactly
        # (piecewise-linear interpolation of a linear-in-lr surface is exact
        # only in lr, and the estimator interpolates in log lr — hence the
        # small tolerance) and alpha must match the OLS of the true crossings.
        ladders = {
            "frontier_dense_b1000": (1000, [0.24, 0.32, 0.37, 0.42, 0.48, 0.55, 0.64]),
            "frontier_dense_b2000": (2000, [0.24, 0.42, 0.48, 0.55, 0.64, 0.73, 0.84]),
            "frontier_dense_b4000": (4000, [0.36, 0.55, 0.64, 0.73, 0.84, 0.97, 1.11]),
        }
        expected = {}
        for tag, (bsz, lrs) in ladders.items():
            lr_star = 0.9 * (bsz / 2000.0) ** 0.5
            expected[bsz] = lr_star / 2 + lrs[0]
            for lr in lrs:
                acc = 0.94 - 0.02 * (lr / lr_star)  # fraction; slope 2pp per lr_star
                for seed in range(1410, 1415):
                    self._write(tmp_path, tag, seed, lr, acc, i=0)
        for lr, acc in ((0.24, 0.79), (0.48, 0.80), (0.72, 0.798), (0.96, 0.77)):
            for seed in (1410, 1411):
                self._write(tmp_path, afd.STEPMATCH_TAG, seed, lr, acc, i=1)
        report = afd.build_report(tmp_path)
        for bsz, want in expected.items():
            got = report["part1"][str(bsz)]["lr_cross"]
            assert got == pytest.approx(want, rel=0.02)
        fit = report["alpha_fit"]
        assert fit["n_points"] == 3
        true_fit = afd.alpha_from_crossings(expected)
        assert fit["alpha"] == pytest.approx(true_fit["alpha"], abs=0.02)
        boot = report["alpha_bootstrap"]
        assert boot and boot["alpha_ci95"][0] <= fit["alpha"] <= boot["alpha_ci95"][1]
        p2 = report["part2_stepmatched_b8000"]
        assert p2["peak_ref_shoulder"] == 0.72  # 79.8 within 1pp of 80.0 peak
        md = afd.to_markdown(report)
        assert "step-matched B=8000" in md

    def test_dedupe_keeps_latest(self, tmp_path):
        self._write(tmp_path, "frontier_dense_b1000", 1410, 0.24, 0.90, i=0)
        self._write(tmp_path, "frontier_dense_b1000", 1410, 0.24, 0.94, i=1)
        dense, _ = afd.load_runs(tmp_path)
        assert dense[1000][0.24] == [pytest.approx(94.0)]
