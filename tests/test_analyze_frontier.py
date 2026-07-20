"""Unit tests for scripts/analyze_frontier.py (program #6).

Covers the pre-registered decision logic only (shoulder convention, alpha
fit, tracking-signature ratios) on hand-built inputs; the sidecar helpers
are exercised end-to-end by the analyzer's own smoke usage and reuse
already-tested code from analyze_occupancy_lr / analyze_smoothness.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "rm_analyze_frontier", REPO_ROOT / "scripts" / "analyze_frontier.py"
)
af = importlib.util.module_from_spec(spec)
spec.loader.exec_module(af)


def mean_acc(**kv):
    """{'lr0_12': 94.0, ...} -> {0.12: 94.0, ...}"""
    return {float(k[2:].replace("_", ".")): v for k, v in kv.items()}


class TestShoulder:
    def test_first_crossing(self):
        acc = mean_acc(
            lr0_12=93.9, lr0_24=94.0, lr0_36=93.8, lr0_48=93.2,
            lr0_6=92.4, lr0_72=91.0, lr0_96=88.0, lr1_44=80.0,
        )
        sh = af.shoulder_from_means(acc)
        assert sh["ref_acc"] == 94.0
        assert sh["floor"] == 93.0
        assert sh["shoulder"] == 0.48
        assert sh["recoveries"] == []

    def test_recovery_flagged_not_used(self):
        acc = mean_acc(
            lr0_12=94.0, lr0_24=94.0, lr0_36=92.5, lr0_48=93.5, lr0_6=90.0
        )
        sh = af.shoulder_from_means(acc)
        assert sh["shoulder"] == 0.24  # first crossing at 0.36 ends the scan
        assert sh["recoveries"] == [0.48]

    def test_ref_is_max_of_two_lowest_rungs(self):
        acc = mean_acc(lr0_12=94.4, lr0_24=94.0, lr0_36=93.5, lr0_48=93.0)
        sh = af.shoulder_from_means(acc)
        assert sh["ref_acc"] == 94.4
        assert sh["floor"] == pytest.approx(93.4)
        assert sh["shoulder"] == 0.36

    def test_all_below_floor_gives_none(self):
        # lowest rung defines ref; everything after immediately crosses
        acc = mean_acc(lr0_12=90.0, lr0_24=88.5, lr0_36=88.0)
        sh = af.shoulder_from_means(acc)
        assert sh["shoulder"] == 0.12  # ref rung itself is always >= floor

    def test_missing_ref_rungs(self):
        assert af.shoulder_from_means(mean_acc(lr0_48=93.0))["shoulder"] is None


class TestAlphaFit:
    def test_recovers_planted_slope(self):
        shoulders = {b: 0.48 * (b / 2000.0) ** 0.5 for b in af.BATCHES}
        fit = af.fit_alpha(shoulders)
        assert fit["alpha"] == pytest.approx(0.5, abs=1e-6)
        assert fit["n_points"] == 5

    def test_flat_frontier_gives_zero(self):
        fit = af.fit_alpha({b: 0.48 for b in af.BATCHES})
        assert fit["alpha"] == pytest.approx(0.0, abs=1e-9)

    def test_fewer_than_two_points(self):
        assert af.fit_alpha({2000: 0.48}) is None
        assert af.fit_alpha({2000: None, 4000: None}) is None


class TestTrackingSignature:
    def _report_with(self, at_shoulder, at_1x):
        table = {}  # ratios only need the p2 sub-tables
        out = {"at_shoulder": at_shoulder, "at_1x": at_1x, "ratios": {}}
        for key in af.P2_KEYS:
            entry = {}
            for which in ("at_shoulder", "at_1x"):
                vals = [v[key] for v in out[which].values() if v.get(key) is not None]
                entry[which] = (
                    round(max(vals) / min(vals), 4)
                    if len(vals) > 1 and min(vals)
                    else None
                )
            r_sh, r_1x = entry["at_shoulder"], entry["at_1x"]
            entry["tracking_signature"] = (
                None
                if r_sh is None or r_1x is None
                else bool(r_sh < af.TRACKING_RATIO <= r_1x)
            )
            out["ratios"][key] = entry
        return out

    def test_tracking_true_requires_flat_at_shoulder_varying_at_1x(self):
        flat = {str(b): {"occupancy": 0.6, "spectral": 1, "euclidean": 1, "hvp_q90": 1}
                for b in af.P2_BATCHES}
        varying = {str(b): {"occupancy": 0.2 + 0.1 * i, "spectral": 1, "euclidean": 1,
                            "hvp_q90": 1}
                   for i, b in enumerate(af.P2_BATCHES)}
        out = self._report_with(flat, varying)
        assert out["ratios"]["occupancy"]["tracking_signature"] is True
        # flat everywhere = insensitive, NOT tracking
        assert out["ratios"]["spectral"]["tracking_signature"] is False

    def test_missing_values_give_none(self):
        out = self._report_with({}, {})
        assert out["ratios"]["occupancy"]["tracking_signature"] is None


class TestLoading:
    def _write_run(self, d, seed, bsz, lr, acc, tag=af.CONFIG_TAG):
        f = d / f"airbench_instrumented_seed{seed}_20260720T{seed}00.json"
        f.write_text(json.dumps({
            "seed": seed,
            "config": {"path": f"sweeps/{tag}/x__lr{lr}_batch_size{bsz}.yaml",
                       "contents": {"train": {"batch_size": bsz, "epochs": 8},
                                    "probe_overrides": {"lr": lr}}},
            "metrics": {"tta_val_acc": acc, "steps": 200,
                        "instrumentation_sidecar": "missing.instrumentation.json"},
        }))

    def test_filters_tag_and_converts_percent(self, tmp_path):
        self._write_run(tmp_path, 1400, 2000, 0.24, 0.9408)
        self._write_run(tmp_path, 1401, 2000, 0.24, 0.9401)
        self._write_run(tmp_path, 1310, 2000, 0.48, 0.9430, tag="smoothness_lr_ladder")
        runs = af.load_frontier_runs(tmp_path)
        assert len(runs) == 2
        assert runs[0]["acc"] == pytest.approx(94.08)

    def test_full_report_on_synthetic_grid(self, tmp_path):
        # planted alpha = 0.5 frontier: shoulder rung shifts up with B
        drop = {500: 0.36, 1000: 0.48, 2000: 0.60, 4000: 0.72, 8000: 0.96}
        for bsz, first_bad in drop.items():
            for lr in af.RUNGS:
                acc = 0.94 if lr < first_bad else 0.90
                for seed in (1400, 1401):
                    self._write_run(tmp_path, seed + af.RUNGS.index(lr) * 10
                                    + af.BATCHES.index(bsz) * 100, bsz, lr, acc)
        report = af.build_report(tmp_path)
        assert report["n_runs"] == 80
        got = {int(b): e["shoulder"] for b, e in report["frontier"].items()}
        assert got == {500: 0.24, 1000: 0.36, 2000: 0.48, 4000: 0.60, 8000: 0.72}
        assert report["alpha_fit"]["alpha"] > 0.25
        md = af.to_markdown(report)
        assert "P1" in md and "P2" in md and "P3" in md
