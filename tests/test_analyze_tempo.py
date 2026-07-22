"""Tests for scripts/analyze_tempo.py on synthetic results JSONs."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_tempo import analyze_compare, analyze_passive, spearman


def fake_run(tmp_path, name, seed, lr, opt="tempomuon", kappa=0.0, scope="per_matrix",
             acc=0.94, rho=-0.1, gain=1.0):
    hist = [
        {"step": s, "rho": {"matrix0_8x12": rho}, "gain": {"matrix0_8x12": gain}}
        for s in range(10, 201, 10)
    ]
    d = {
        "seed": seed,
        "config": {
            "contents": {
                "optimizer": {"name": opt, "lr": lr, "kappa": kappa, "scope": scope}
            },
            "path": f"{name}.yaml",
            "sha256": "0" * 64,
        },
        "metrics": {"tta_val_acc": acc, "tempo_stats": {"history": hist}},
    }
    p = tmp_path / f"{name}_seed{seed}.json"
    p.write_text(json.dumps(d))
    return d


def test_passive_monotonicity_and_table(tmp_path):
    runs = []
    for lr, rho in [(0.24, -0.05), (0.48, -0.15), (0.96, -0.30)]:
        for seed in (1420, 1421):
            runs.append(
                fake_run(tmp_path, f"p{lr}", seed, lr, rho=rho + 0.001 * (seed % 2))
            )
    out = analyze_passive(runs)
    assert out["spearman_lr_vs_agg_rho"] < -0.9
    assert "0.24" in out["per_lr"]
    assert out["per_lr"]["0.96"]["agg_rho_mean"] < out["per_lr"]["0.24"]["agg_rho_mean"]


def test_compare_paired_deltas(tmp_path):
    runs = []
    for seed in (1420, 1421, 1422):
        runs.append(fake_run(tmp_path, "m", seed, 0.72, opt="muon", acc=0.926))
        runs.append(
            fake_run(tmp_path, "t", seed, 0.72, kappa=0.3, acc=0.936, gain=0.5)
        )
    out = analyze_compare(runs)
    row = out["per_lr"]["0.72"]
    assert row["muon"]["n"] == 3
    delta = row["delta_tempomuon-per_matrix"]
    assert delta["n"] == 3
    assert abs(delta["mean_pp"] - 1.0) < 1e-9


def test_nanogpt_passive_fixed_step_table():
    from scripts.analyze_tempo import analyze_nanogpt_passive

    def run(lr, base):
        rows = []
        for step in range(2, 301):
            for m in (0, 2):
                rows.append({"step": step, "matrix": m,
                             "cos_gg": base + 0.001 * m, "cos_gm": base / 2})
        return {
            "seed": 1440,
            "config": {"contents": {"nanogpt": {"muon_lr": lr}}},
            "metrics": {"tempo_probe": {"rows": rows}, "final_val_loss": 4.0},
        }

    out = analyze_nanogpt_passive([run(0.035, -0.5), run(0.15, -0.2)],
                                  steps=(100, 200))
    assert out["per_lr"]["0.035"]["cos_gg"]["100"] < out["per_lr"]["0.15"]["cos_gg"]["100"]
    assert abs(out["per_lr"]["0.15"]["cos_gm"]["200"] - (-0.1)) < 1e-9


def test_spearman_perfect_orders():
    assert spearman([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman([1, 2, 3], [30, 20, 10]) == -1.0
