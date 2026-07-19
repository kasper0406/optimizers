"""scripts/analyze_mechanism.py tests (Gate-1 amendment A4 analysis).

CPU-only, synthetic sidecars (same raw-sidecar reading convention as
scripts/analyze_disambiguation.py, whose statistics core it imports).

Seeds: dev seeds only (>= 1000).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.instrument.schema import INSTRUMENTATION_SCHEMA_VERSION  # noqa: E402


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


am = _load_module("analyze_mechanism", "scripts/analyze_mechanism.py")


# ---------------------------------------------------------------- fixtures


def _direction(kind, index, rho_value, steps, total_steps=200):
    n = len(steps)
    return {
        "index": index,
        "kind": kind,
        "s": [0.0] * total_steps,
        "reset_steps": [],
        "refresh_alignment": {"step": [], "value": []},
        "sigma": {"step": [], "value": []},
        "lambda_hvp": {"step": [], "value": []},
        "per_beta": {
            "0.9": {
                "step": list(steps),
                "regime": ["noise"] * n,
                "mu": [0.0] * n,
                "var": [1.0] * n,
                "rho": [rho_value] * n,
                "t_stat": [0.0] * n,
                "amplitude_ratio": [1.0] * n,
                "implied_eta_lambda": [2.0] * n,
                "ess": [10.0] * n,
                "n_since_reset": [100] * n,
            }
        },
    }


def _sidecar(rho_top, rho_bulk):
    steps = list(range(5, 201, 5))
    return {
        "instrumentation_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
        "betas": ["0.9"],
        "hvp_enabled": False,
        "matrices": {
            "layers.0.conv1.weight": {
                "shape": [8, 8],
                "k1": 1,
                "k2": 1,
                "t_refresh": 50,
                "align_min": 0.9,
                "snapshot_every": 5,
                "steps": list(range(1, 201)),
                "grad_fro_norm": [1.0] * 200,
                "top_sigma_m": [1.0] * 200,
                "refresh_steps": [1, 51, 101, 151],
                "directions": [
                    _direction("top", 0, rho_top, steps),
                    _direction("bulk", 1, rho_bulk, steps),
                ],
            }
        },
    }


@pytest.fixture()
def probe_dirs(tmp_path):
    """Baseline: top AND bulk negative-rho (all-frac 1.0). mom0: none (0.0).
    lrhalf: bulk only (0.5). lrquarter: bulk only (0.5) -- a planted
    'large drop at momentum 0, partial drop down the LR ladder' pattern."""
    spec = {
        "baseline": ((1000, 1001), -0.5, -0.5),
        "mom0": ((1300, 1301), 0.1, 0.1),
        "lrhalf": ((1302, 1303), 0.1, -0.5),
        "lrquarter": ((1304, 1305), 0.1, -0.5),
    }
    dirs = {}
    for name, (seeds, rho_top, rho_bulk) in spec.items():
        d = tmp_path / name
        d.mkdir()
        for seed in seeds:
            (d / f"run_seed{seed}.instrumentation.json").write_text(
                json.dumps(_sidecar(rho_top, rho_bulk))
            )
        dirs[name] = d
    return dirs


# ------------------------------------------------------------------- tests


def test_probe_comparison_fractions_and_differences(probe_dirs, tmp_path):
    out_json = tmp_path / "mech.json"
    out_md = tmp_path / "mech.md"
    rc = am.main(
        [
            "--baseline", str(probe_dirs["baseline"]),
            "--mom0", str(probe_dirs["mom0"]),
            "--lrhalf", str(probe_dirs["lrhalf"]),
            "--lrquarter", str(probe_dirs["lrquarter"]),
            "--n-boot", "200",
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ]
    )
    assert rc == 0
    report = json.loads(out_json.read_text())
    comp = report["mechanism_comparison"]
    assert comp["sets"] == ["baseline", "mom0", "lrhalf", "lrquarter"]
    assert comp["n_runs"] == {
        "baseline": 2, "mom0": 2, "lrhalf": 2, "lrquarter": 2
    }

    cells = comp["variants"]["raw"]["cells"]
    by_key = {(c["beta"], c["phase"], c["kind"]): c for c in cells}
    assert len(by_key) == len(cells)
    for phase in (1, 2, 3, 4):
        c = by_key[("0.9", phase, "all")]
        assert c["sets"]["baseline"]["mean_frac"] == pytest.approx(1.0)
        assert c["sets"]["mom0"]["mean_frac"] == pytest.approx(0.0)
        assert c["sets"]["lrhalf"]["mean_frac"] == pytest.approx(0.5)
        assert c["sets"]["lrquarter"]["mean_frac"] == pytest.approx(0.5)
        # Probe - baseline differences (degenerate CIs: identical runs).
        d = c["difference_probe_minus_baseline"]
        assert d["mom0"]["mean"] == pytest.approx(-1.0)
        assert d["mom0"]["ci"] == [pytest.approx(-1.0), pytest.approx(-1.0)]
        assert d["lrhalf"]["mean"] == pytest.approx(-0.5)
        # Bulk-vs-top split per config.
        top = by_key[("0.9", phase, "top")]
        bulk = by_key[("0.9", phase, "bulk")]
        assert top["sets"]["lrhalf"]["mean_frac"] == pytest.approx(0.0)
        assert bulk["sets"]["lrhalf"]["mean_frac"] == pytest.approx(1.0)
        assert top["sets"]["baseline"]["mean_frac"] == pytest.approx(1.0)

    md = out_md.read_text()
    assert "descriptive; no pass/fail" in md
    assert "bulk-vs-top split" in md
    assert "momentum-overshoot predicts" in md  # context stated, not judged


def test_subset_of_probes_is_allowed(probe_dirs, tmp_path):
    out_json = tmp_path / "subset.json"
    rc = am.main(
        [
            "--baseline", str(probe_dirs["baseline"]),
            "--mom0", str(probe_dirs["mom0"]),
            "--n-boot", "100",
            "--out-json", str(out_json),
        ]
    )
    assert rc == 0
    comp = json.loads(out_json.read_text())["mechanism_comparison"]
    assert comp["sets"] == ["baseline", "mom0"]


def test_requires_at_least_one_probe(probe_dirs):
    with pytest.raises(SystemExit):
        am.main(["--baseline", str(probe_dirs["baseline"])])


def test_output_is_deterministic(probe_dirs, tmp_path):
    outs = []
    for tag in ("a", "b"):
        oj = tmp_path / f"{tag}.json"
        om = tmp_path / f"{tag}.md"
        am.main(
            [
                "--baseline", str(probe_dirs["baseline"]),
                "--mom0", str(probe_dirs["mom0"]),
                "--lrhalf", str(probe_dirs["lrhalf"]),
                "--n-boot", "100",
                "--out-json", str(oj),
                "--out-md", str(om),
            ]
        )
        outs.append((oj.read_bytes(), om.read_bytes()))
    assert outs[0] == outs[1]
