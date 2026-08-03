"""Tests for the two mechanism-experiment analyzers of the flow-first program
(docs/litreview/j-theory-theorem-sweep.md §6):

  * ``scripts/analyze_teleport_gate.py`` (program item 4 / T6) on a synthetic
    `airbench_teleport_gate` results directory — per-snapshot means, the
    stdlib percentile helper, the overall min over snapshots, NaN filtering,
    duplicate-seed dedupe, and the SystemExit refusals;
  * ``scripts/analyze_centralflow.py`` (program item 2) on a synthetic
    `airbench_centralflow` 2x2 — arm identification, seed pairing across all
    four arms, the four paired deltas, recovery_fraction, term telemetry, and
    the SystemExit refusals.

Both fixtures plant hand-computable numbers on two seeds, so every mean, CI
and ratio asserted below is exact by construction. Style mirrors
tests/test_anneal_branch.py; CPU-only, no torch.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze_teleport_gate = _load("rm_analyze_teleport_gate", "analyze_teleport_gate.py")
analyze_centralflow = _load("rm_analyze_centralflow", "analyze_centralflow.py")


# ============================================================ percentile helper

percentile = analyze_teleport_gate.percentile


def test_percentile_interpolates_between_ranks():
    # unsorted on purpose: the helper sorts a copy
    values = [1.10, 1.00, 1.04, 1.02]
    assert percentile(values, 0) == pytest.approx(1.00, abs=1e-12)
    assert percentile(values, 100) == pytest.approx(1.10, abs=1e-12)
    # p50 with n=4: position 1.5 -> midpoint of the two middle entries
    assert percentile(values, 50) == pytest.approx(1.03, abs=1e-12)
    # p90 with n=4: position 2.7 -> 1.04 + 0.7 * (1.10 - 1.04)
    assert percentile(values, 90) == pytest.approx(1.082, abs=1e-12)
    assert values == [1.10, 1.00, 1.04, 1.02]  # input untouched


def test_percentile_exact_rank_and_degenerate_inputs():
    # position lands exactly on an index -> that entry, no interpolation
    assert percentile([0.0, 1.0, 2.0, 3.0, 4.0], 25) == pytest.approx(1.0, abs=1e-12)
    assert percentile([5.0], 90) == pytest.approx(5.0, abs=1e-12)
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)


# ================================================================ gate fixture

# Per-seed tilt: +0.01 on the ratios of seed 1001, so every across-seed CI is
# non-degenerate and hand-computable.
GATE_SEEDS = (1000, 1001)
SNAP_STEPS = [50, 100]
SAMPLE_NUC = {50: [1.00, 1.02, 1.04, 1.10], 100: [0.90, 0.95, 1.00, 1.20]}
BEST_RANDOM_NUC = {50: 1.10, 100: 1.20}
BEST_REFINED_NUC = {50: 1.25, 100: 1.40}
BEST_REFINED_GRAD = {50: 1.30, 100: 1.50}
N_FEASIBLE = {50: 60, 100: 62}
MAX_DLOSS = {(1000, 50): 4e-4, (1000, 100): 5e-4, (1001, 50): 6e-4, (1001, 100): 3e-4}
PAIR_NAMES = [f"conv{i}" for i in range(6)]


def _sample(nuc, dloss):
    return {
        "rel_dloss": dloss,
        "total_grad_ratio": nuc + 0.005,
        "nuclear_sum_ratio": nuc,
        "fro_ratio": [nuc] * 6,
        "nuc_ratio": [nuc] * 6,
    }


def _snapshot(seed, step):
    tilt = 0.01 * (seed - 1000)
    samples = [
        _sample(nuc + tilt, 1e-5 * (i + 1)) for i, nuc in enumerate(SAMPLE_NUC[step])
    ]
    best_random = _sample(BEST_RANDOM_NUC[step] + tilt, 2e-5)
    best_refined = dict(
        _sample(BEST_REFINED_NUC[step] + tilt, 3e-5),
        n_accepted=7,
        n_infeasible=11,
    )
    best_refined["total_grad_ratio"] = BEST_REFINED_GRAD[step] + tilt
    return {
        "step": step,
        "base_loss": 1234.5,
        "base_total_grad": 10.0,
        "base_nuclear_sum": 20.0,
        "base_fro": [1.0] * 6,
        "base_nuclear": [2.0] * 6,
        "samples": samples,
        "n_feasible": N_FEASIBLE[step] + 2 * (seed - 1000),
        "max_abs_rel_dloss": MAX_DLOSS[(seed, step)],
        "best_random": best_random,
        "best_refined": best_refined,
    }


def _gate_payload(seed, started="2026-08-03T00:00:00"):
    return {
        "experiment": "airbench_teleport_gate",
        "seed": seed,
        "started_at": started,
        "gpu_type": "NVIDIA L40",
        "metrics": {
            "optimizer": "vendor_muon",
            "steps": 200,
            "invariance_tol": 1e-3,
            "pair_names": PAIR_NAMES,
            "snapshots": {str(t): _snapshot(seed, t) for t in SNAP_STEPS},
            "final_val_acc": 0.9400 + 0.001 * (seed - 1000),
            "probe_seconds": 12.0,
            "time_seconds": 30.0,
        },
    }


def _write(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload))


def _populate_gate(tmp_path):
    for seed in GATE_SEEDS:
        _write(tmp_path, f"gate_{seed}.json", _gate_payload(seed))
    # an unrelated experiment in the same directory must be ignored
    _write(
        tmp_path,
        "other.json",
        {
            "experiment": "airbench_ema",
            "seed": 1000,
            "started_at": "2026-08-03T00:00:00",
            "gpu_type": "NVIDIA RTX A6000",
            "metrics": {"val_acc": 0.5},
        },
    )


# ================================================================ gate analyzer


def test_gate_analyze_per_snapshot_summaries(tmp_path):
    _populate_gate(tmp_path)
    result = analyze_teleport_gate.analyze(tmp_path)

    assert result["n_seeds"] == 2
    assert result["seeds"] == [1000, 1001]
    assert result["gpu_type"] == "NVIDIA L40"
    assert result["invariance_tol"] == 1e-3
    assert result["snapshot_steps"] == [50, 100]

    block = result["per_snapshot"]["50"]
    assert block["n_snapshot_seeds"] == 2
    # n_feasible 60 and 62 -> 61; worst |rel_dloss| is seed 1001's 6e-4
    assert block["n_feasible_mean"] == pytest.approx(61.0, abs=1e-12)
    assert block["max_abs_rel_dloss_max"] == pytest.approx(6e-4, abs=1e-12)
    # per-seed p50 of [1.00,1.02,1.04,1.10] (+tilt) -> 1.03 and 1.04
    assert block["random_draw_nuclear_p50"] == pytest.approx(1.035, abs=1e-9)
    # per-seed p90 -> 1.082 and 1.092
    assert block["random_draw_nuclear_p90"] == pytest.approx(1.087, abs=1e-9)
    assert block["best_random_nuclear"]["mean"] == pytest.approx(1.105, abs=1e-9)
    assert block["best_refined_nuclear"]["n"] == 2
    assert block["best_refined_nuclear"]["mean"] == pytest.approx(1.255, abs=1e-9)
    # ci95 = t(df=1) * sd / sqrt(2), sd = 0.01/sqrt(2) over the two seeds
    assert block["best_refined_nuclear"]["ci95"] == pytest.approx(
        12.706 * 0.005, abs=1e-9
    )
    assert block["best_refined_total_grad"]["mean"] == pytest.approx(1.305, abs=1e-9)

    late = result["per_snapshot"]["100"]
    assert late["n_feasible_mean"] == pytest.approx(63.0, abs=1e-12)
    assert late["max_abs_rel_dloss_max"] == pytest.approx(5e-4, abs=1e-12)
    # p50 of [0.90,0.95,1.00,1.20] -> 0.975 / 0.985 ; p90 -> 1.14 / 1.15
    assert late["random_draw_nuclear_p50"] == pytest.approx(0.98, abs=1e-9)
    assert late["random_draw_nuclear_p90"] == pytest.approx(1.145, abs=1e-9)
    assert late["best_refined_nuclear"]["mean"] == pytest.approx(1.405, abs=1e-9)
    assert late["best_refined_total_grad"]["mean"] == pytest.approx(1.505, abs=1e-9)


def test_gate_overall_min_and_markdown(tmp_path):
    _populate_gate(tmp_path)
    result = analyze_teleport_gate.analyze(tmp_path)
    # min over snapshots of the best-refined nuclear means (1.255 vs 1.405)
    assert result["overall"][
        "min_over_snapshots_best_refined_nuclear_mean"
    ] == pytest.approx(1.255, abs=1e-9)

    md = analyze_teleport_gate.to_markdown(result)
    assert "Teleportation gate" in md
    assert "| 50 |" in md and "| 100 |" in md
    assert "Smallest best-refined nuclear-sum ratio" in md
    assert "1.2550" in md
    # every cell has two seeds, so every summary carries a real CI
    assert "(n=1)" not in md


def test_gate_missing_best_random_drops_that_seed(tmp_path):
    _populate_gate(tmp_path)
    payload = json.loads((tmp_path / "gate_1001.json").read_text())
    payload["metrics"]["snapshots"]["50"]["best_random"] = None
    _write(tmp_path, "gate_1001.json", payload)
    block = analyze_teleport_gate.analyze(tmp_path)["per_snapshot"]["50"]
    # only seed 1000 reported a feasible random draw -> n=1, no CI
    assert block["best_random_nuclear"]["n"] == 1
    assert block["best_random_nuclear"]["mean"] == pytest.approx(1.10, abs=1e-12)
    assert block["best_random_nuclear"]["ci95"] is None
    # the refined record is untouched
    assert block["best_refined_nuclear"]["n"] == 2


def test_gate_nan_samples_are_filtered(tmp_path):
    """``_ratio`` yields NaN on a zero base norm; percentiles must skip it."""
    _populate_gate(tmp_path)
    for seed in GATE_SEEDS:
        payload = json.loads((tmp_path / f"gate_{seed}.json").read_text())
        payload["metrics"]["snapshots"]["50"]["samples"].append(
            _sample(float("nan"), 1e-5)
        )
        _write(tmp_path, f"gate_{seed}.json", payload)
    block = analyze_teleport_gate.analyze(tmp_path)["per_snapshot"]["50"]
    assert block["random_draw_nuclear_p50"] == pytest.approx(1.035, abs=1e-9)
    assert block["random_draw_nuclear_p90"] == pytest.approx(1.087, abs=1e-9)


def test_gate_duplicate_seed_keeps_earliest(tmp_path):
    _populate_gate(tmp_path)
    payload = _gate_payload(1000, started="2026-08-04T00:00:00")
    for step in SNAP_STEPS:
        payload["metrics"]["snapshots"][str(step)]["best_refined"][
            "nuclear_sum_ratio"
        ] = 9.0
    _write(tmp_path, "gate_1000_rerun.json", payload)
    result = analyze_teleport_gate.analyze(tmp_path)
    # the later rerun must not displace the original run's numbers
    assert result["n_seeds"] == 2
    assert result["per_snapshot"]["50"]["best_refined_nuclear"][
        "mean"
    ] == pytest.approx(1.255, abs=1e-9)


def test_gate_main_writes_outputs(tmp_path):
    _populate_gate(tmp_path)
    out_json = tmp_path / "gate.json"
    out_md = tmp_path / "gate.md"
    code = analyze_teleport_gate.main(
        [str(tmp_path), "--json", str(out_json), "--md", str(out_md)]
    )
    assert code == 0
    written = json.loads(out_json.read_text())
    assert written["overall"][
        "min_over_snapshots_best_refined_nuclear_mean"
    ] == pytest.approx(1.255, abs=1e-9)
    assert "Teleportation gate" in out_md.read_text()


def test_gate_empty_dir_refused(tmp_path):
    with pytest.raises(SystemExit, match="no airbench_teleport_gate runs"):
        analyze_teleport_gate.analyze(tmp_path)
    # a run without snapshots is not a gate result either
    payload = _gate_payload(1000)
    payload["metrics"]["snapshots"] = {}
    _write(tmp_path, "empty_gate.json", payload)
    with pytest.raises(SystemExit, match="no airbench_teleport_gate runs"):
        analyze_teleport_gate.analyze(tmp_path)


def test_gate_mixed_gpu_types_refused(tmp_path):
    _populate_gate(tmp_path)
    payload = json.loads((tmp_path / "gate_1001.json").read_text())
    payload["gpu_type"] = "NVIDIA RTX A6000"
    _write(tmp_path, "gate_1001.json", payload)
    with pytest.raises(SystemExit, match="mixed GPU types"):
        analyze_teleport_gate.analyze(tmp_path)


def test_gate_mixed_invariance_tol_refused(tmp_path):
    _populate_gate(tmp_path)
    payload = json.loads((tmp_path / "gate_1001.json").read_text())
    payload["metrics"]["invariance_tol"] = 1e-2
    _write(tmp_path, "gate_1001.json", payload)
    with pytest.raises(SystemExit, match="disagree on invariance_tol"):
        analyze_teleport_gate.analyze(tmp_path)


# ========================================================== central-flow fixture

# Arm deltas are scaled per seed (x1.0, x1.2) so the CIs are non-degenerate
# while the ratio (C-B)/(A-B) stays exactly 0.75 on both seeds — and therefore
# on their mean.
CF_SEEDS = (1000, 1001)
A_VAL = {1000: 0.9400, 1001: 0.9420}
SEED_SCALE = {1000: 1.0, 1001: 1.2}
ARM_DELTA = {"A": 0.0, "B": -0.0200, "C": -0.0050, "D": 0.0010}
TTA_OFFSET = {"A": 0.010, "B": 0.010, "C": 0.012, "D": 0.012}
CF_SETTINGS = {
    "A": (1.0, False),
    "B": (0.25, False),
    "C": (0.25, True),
    "D": (1.0, True),
}
PEN_BASE = {"C": 0.10, "D": 0.30}
CURV_BASE = {"C": 2.0, "D": 5.0}
N_TS = 3


def _cf_val(arm, seed, deltas=None):
    deltas = ARM_DELTA if deltas is None else deltas
    return A_VAL[seed] + deltas[arm] * SEED_SCALE[seed]


def _cf_timeseries(arm, seed):
    if arm not in PEN_BASE:
        return []
    tilt = seed - 1000
    return [
        {
            "step": 10 * (i + 1),
            "muon_lr": 0.24 * CF_SETTINGS[arm][0],
            "n_directions": 4,
            "penalty_grad_norm": PEN_BASE[arm] + 0.10 * i + 0.01 * tilt,
            "curvature_mean": CURV_BASE[arm] + i + 0.5 * tilt,
            "curvature_max": 2 * (CURV_BASE[arm] + i + 0.5 * tilt),
        }
        for i in range(N_TS)
    ]


def _cf_payload(arm, seed, started="2026-08-03T00:00:00", deltas=None):
    lr_scale, enabled = CF_SETTINGS[arm]
    val = _cf_val(arm, seed, deltas)
    return {
        "experiment": "airbench_centralflow",
        "seed": seed,
        "started_at": started,
        "gpu_type": "NVIDIA L40",
        "metrics": {
            "optimizer": "vendor_muon",
            "epochs": 8,
            "steps": 200,
            "cf": {
                "lr_scale": lr_scale,
                "enabled": enabled,
                "refresh_every": 10,
                "k_directions": 4,
                "beta_scale": 1.0,
            },
            "val_accs": [val - 0.01 * (7 - i) for i in range(8)],
            "val_acc": val,
            "tta_val_acc": val + TTA_OFFSET[arm],
            "cf_timeseries": _cf_timeseries(arm, seed),
            "time_seconds": 40.0,
        },
    }


def _populate_cf(tmp_path, arms=("A", "B", "C", "D"), deltas=None):
    for arm in arms:
        for seed in CF_SEEDS:
            _write(
                tmp_path,
                f"cf_{arm}_{seed}.json",
                _cf_payload(arm, seed, deltas=deltas),
            )
    # a cf block outside the 2x2 must be ignored, not mistaken for an arm
    stray = _cf_payload("A", 1000)
    stray["seed"] = 1002
    stray["metrics"]["cf"]["lr_scale"] = 0.5
    _write(tmp_path, "cf_stray_1002.json", stray)


# ========================================================= central-flow analyzer


def test_cf_arm_identification_and_means(tmp_path):
    _populate_cf(tmp_path)
    result = analyze_centralflow.analyze(tmp_path)

    assert result["n_paired_seeds"] == 2
    assert result["seeds"] == [1000, 1001]
    assert result["gpu_type"] == "NVIDIA L40"
    assert sorted(result["per_arm"]) == ["A", "B", "C", "D"]

    means = {arm: result["per_arm"][arm]["final_val_mean"] for arm in "ABCD"}
    assert means["A"] == pytest.approx(0.9410, abs=1e-9)
    assert means["B"] == pytest.approx(0.9190, abs=1e-9)  # 0.9200, 0.9180
    assert means["C"] == pytest.approx(0.9355, abs=1e-9)  # 0.9350, 0.9360
    assert means["D"] == pytest.approx(0.9421, abs=1e-9)  # 0.9410, 0.9432
    # TTA offsets: +0.010 on the term-off arms, +0.012 on the term-on arms
    assert result["per_arm"]["A"]["final_tta_mean"] == pytest.approx(0.9510, abs=1e-9)
    assert result["per_arm"]["C"]["final_tta_mean"] == pytest.approx(0.9475, abs=1e-9)

    curve = result["per_arm"]["A"]["val_accs_mean"]
    assert len(curve) == 8
    assert curve[-1] == pytest.approx(means["A"], abs=1e-9)
    assert curve[0] == pytest.approx(means["A"] - 0.07, abs=1e-9)


def test_cf_paired_deltas_and_recovery_fraction(tmp_path):
    _populate_cf(tmp_path)
    result = analyze_centralflow.analyze(tmp_path)
    deltas = result["paired_deltas"]
    assert sorted(deltas) == ["B_minus_A", "C_minus_A", "C_minus_B", "D_minus_A"]

    # C - B: +0.0150 and +0.0180 -> mean 0.0165
    mech = deltas["C_minus_B"]["val_delta"]
    assert mech["n"] == 2
    assert mech["mean"] == pytest.approx(0.0165, abs=1e-9)
    # ci95 = t(df=1) * sd / sqrt(2), sd = 0.0030/sqrt(2)
    assert mech["ci95"] == pytest.approx(12.706 * 0.0015, abs=1e-9)
    assert deltas["C_minus_A"]["val_delta"]["mean"] == pytest.approx(
        -0.0055, abs=1e-9
    )
    assert deltas["B_minus_A"]["val_delta"]["mean"] == pytest.approx(
        -0.0220, abs=1e-9
    )
    assert deltas["D_minus_A"]["val_delta"]["mean"] == pytest.approx(
        0.0011, abs=1e-9
    )
    # TTA deltas: +0.002 wherever a term-on arm is compared to a term-off one
    assert deltas["C_minus_B"]["tta_delta"]["mean"] == pytest.approx(
        0.0185, abs=1e-9
    )
    assert deltas["C_minus_A"]["tta_delta"]["mean"] == pytest.approx(
        -0.0035, abs=1e-9
    )
    assert deltas["B_minus_A"]["tta_delta"]["mean"] == pytest.approx(
        -0.0220, abs=1e-9
    )
    assert deltas["D_minus_A"]["tta_delta"]["mean"] == pytest.approx(
        0.0031, abs=1e-9
    )
    # 0.0165 / 0.0220 = 0.75 exactly by construction
    assert result["recovery_fraction"] == pytest.approx(0.75, abs=1e-9)


def test_cf_recovery_fraction_none_when_gap_unresolvable(tmp_path):
    # cold LR costs nothing -> mean(A - B) = 0, the ratio is undefined
    flat = dict(ARM_DELTA, B=0.0)
    _populate_cf(tmp_path, deltas=flat)
    result = analyze_centralflow.analyze(tmp_path)
    assert result["paired_deltas"]["B_minus_A"]["val_delta"]["mean"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result["recovery_fraction"] is None
    assert "n/a" in analyze_centralflow.to_markdown(result)

    # ... and likewise when the cold arm is ABOVE stock (A - B negative)
    inverted = dict(ARM_DELTA, B=0.0200)
    _populate_cf(tmp_path, deltas=inverted)
    assert analyze_centralflow.analyze(tmp_path)["recovery_fraction"] is None


def test_cf_telemetry_first_last_and_midpoint(tmp_path):
    _populate_cf(tmp_path)
    telemetry = analyze_centralflow.analyze(tmp_path)["cf_telemetry"]
    assert sorted(telemetry) == ["C", "D"]

    arm_c = telemetry["C"]
    assert arm_c["n_seeds_with_timeseries"] == 2
    assert arm_c["penalty_norm_first"] == pytest.approx(0.105, abs=1e-9)
    assert arm_c["penalty_norm_last"] == pytest.approx(0.305, abs=1e-9)
    # midpoint of a 3-entry series is index 1: 3.0 and 3.5
    assert arm_c["curvature_mean_mid"] == pytest.approx(3.25, abs=1e-9)

    arm_d = telemetry["D"]
    assert arm_d["penalty_norm_first"] == pytest.approx(0.305, abs=1e-9)
    assert arm_d["penalty_norm_last"] == pytest.approx(0.505, abs=1e-9)
    assert arm_d["curvature_mean_mid"] == pytest.approx(6.25, abs=1e-9)


def test_cf_markdown_and_main(tmp_path):
    _populate_cf(tmp_path)
    result = analyze_centralflow.analyze(tmp_path)
    md = analyze_centralflow.to_markdown(result)
    assert "Central-flow Muon v0" in md
    assert "mechanism effect (term at cold LR)" in md
    assert "C − B" in md and "D − A" in md
    assert "recovery_fraction" in md and "0.750" in md
    assert "Arm C" in md and "Arm D" in md
    assert "+0.0165" in md

    out_json = tmp_path / "cf.json"
    out_md = tmp_path / "cf.md"
    code = analyze_centralflow.main(
        [str(tmp_path), "--json", str(out_json), "--md", str(out_md)]
    )
    assert code == 0
    written = json.loads(out_json.read_text())
    assert written["recovery_fraction"] == pytest.approx(0.75, abs=1e-9)
    assert "Central-flow Muon v0" in out_md.read_text()


def test_cf_empty_or_incomplete_dir_refused(tmp_path):
    with pytest.raises(SystemExit, match="no airbench_centralflow runs"):
        analyze_centralflow.analyze(tmp_path)
    # three of the four arms is not a pairable 2x2
    _populate_cf(tmp_path, arms=("A", "B", "C"))
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_centralflow.analyze(tmp_path)
    # ... and neither is a fourth arm on a seed the others do not have
    payload = _cf_payload("D", 1000)
    payload["seed"] = 1009
    _write(tmp_path, "cf_D_1009.json", payload)
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_centralflow.analyze(tmp_path)


def test_cf_mixed_gpu_types_refused(tmp_path):
    _populate_cf(tmp_path)
    payload = json.loads((tmp_path / "cf_C_1001.json").read_text())
    payload["gpu_type"] = "NVIDIA RTX A6000"
    _write(tmp_path, "cf_C_1001.json", payload)
    with pytest.raises(SystemExit, match="mixed GPU types"):
        analyze_centralflow.analyze(tmp_path)


def test_cf_duplicate_arm_seed_keeps_earliest(tmp_path):
    _populate_cf(tmp_path)
    payload = _cf_payload("C", 1000, started="2026-08-04T00:00:00")
    payload["metrics"]["val_acc"] = 0.5
    payload["metrics"]["tta_val_acc"] = 0.5
    _write(tmp_path, "cf_C_1000_rerun.json", payload)
    result = analyze_centralflow.analyze(tmp_path)
    assert result["per_arm"]["C"]["final_val_mean"] == pytest.approx(0.9355, abs=1e-9)
    assert result["recovery_fraction"] == pytest.approx(0.75, abs=1e-9)


def test_both_analyzers_ignore_each_others_results(tmp_path):
    """The two experiments land in the same results/ directory."""
    _populate_gate(tmp_path)
    _populate_cf(tmp_path)
    gate = analyze_teleport_gate.analyze(tmp_path)
    cf = analyze_centralflow.analyze(tmp_path)
    assert gate["n_seeds"] == 2 and gate["snapshot_steps"] == [50, 100]
    assert cf["n_paired_seeds"] == 2
    assert math.isclose(cf["recovery_fraction"], 0.75, abs_tol=1e-9)
