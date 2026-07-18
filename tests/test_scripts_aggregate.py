"""Tests for scripts/aggregate.py: mean/std/95% CI correctness on synthetic
results JSONs, rejection of invalid / provenance-less files, and refusal of
mixed-gpu_type comparison tables.

All seeds in synthetic results are dev seeds (>= 1000) per CLAUDE.md rule 2.
"""

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import results_io


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = _load_module("aggregate_module", "scripts/aggregate.py")


def make_result(
    seed,
    accuracy=0.94,
    wall_time_s=3.0,
    gpu_type="cpu",
    config_path="configs/synthetic_a.yaml",
    config_sha="a" * 64,
    git_sha="0123456789abcdef0123456789abcdef01234567",
    cost_usd=None,
    experiment="synthetic",
):
    metrics = {"loss_last": 1.0}
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    return {
        "schema_version": results_io.SCHEMA_VERSION,
        "experiment": experiment,
        "config": {
            "path": config_path,
            "sha256": config_sha,
            "contents": {"experiment": experiment},
        },
        "git_sha": git_sha,
        "git_dirty": False,
        "seed": seed,
        "gpu_type": gpu_type,
        "wall_time_s": wall_time_s,
        "cost_usd": cost_usd,
        "started_at": "2026-07-18T00:00:00+00:00",
        "finished_at": "2026-07-18T00:00:03+00:00",
        "metrics": metrics,
    }


def write_results(tmp_path, results):
    paths = []
    for i, result in enumerate(results):
        path = tmp_path / f"r{i:03d}_seed{result['seed']}.json"
        results_io.write_result(result, path)
        paths.append(path)
    return paths


# ------------------------------------------------------------------- stats


def test_mean_std_ci95_matches_reference_formula():
    values = [0.94, 0.95, 0.96, 0.93]
    stats = aggregate.mean_std_ci95(values)
    arr = np.asarray(values)
    n = len(values)
    expected_std = arr.std(ddof=1)
    expected_ci = scipy_stats.t.ppf(0.975, n - 1) * expected_std / math.sqrt(n)
    assert stats["n"] == n
    assert stats["mean"] == pytest.approx(arr.mean())
    assert stats["std"] == pytest.approx(expected_std)
    assert stats["ci95"] == pytest.approx(expected_ci)


def test_mean_std_ci95_single_value_has_nan_ci():
    stats = aggregate.mean_std_ci95([0.94])
    assert stats["mean"] == pytest.approx(0.94)
    assert math.isnan(stats["std"]) and math.isnan(stats["ci95"])


def test_summarize_computes_group_stats(tmp_path):
    accs_a = [0.94, 0.95, 0.96]
    walls_a = [3.0, 3.2, 3.4]
    results = [
        make_result(1000 + i, accuracy=a, wall_time_s=w)
        for i, (a, w) in enumerate(zip(accs_a, walls_a))
    ] + [
        make_result(
            1000 + i,
            accuracy=0.90 + 0.01 * i,
            config_path="configs/synthetic_b.yaml",
            config_sha="b" * 64,
        )
        for i in range(2)
    ]
    write_results(tmp_path, results)

    loaded, warnings = aggregate.load_results([tmp_path])
    assert warnings == []
    rows = aggregate.summarize(loaded, metric="accuracy")
    assert len(rows) == 2

    row_a = next(r for r in rows if r["config"] == "configs/synthetic_a.yaml")
    assert row_a["n"] == 3
    assert row_a["seed_min"] == 1000 and row_a["seed_max"] == 1002
    assert row_a["metric"]["mean"] == pytest.approx(np.mean(accs_a))
    assert row_a["metric"]["std"] == pytest.approx(np.std(accs_a, ddof=1))
    expected_ci = scipy_stats.t.ppf(0.975, 2) * np.std(accs_a, ddof=1) / math.sqrt(3)
    assert row_a["metric"]["ci95"] == pytest.approx(expected_ci)
    assert row_a["wall_time_s"]["mean"] == pytest.approx(np.mean(walls_a))

    row_b = next(r for r in rows if r["config"] == "configs/synthetic_b.yaml")
    assert row_b["n"] == 2
    assert row_b["metric"]["mean"] == pytest.approx(0.905)


def test_summarize_rejects_duplicate_seeds(tmp_path):
    write_results(tmp_path, [make_result(1000), make_result(1000, accuracy=0.9)])
    loaded, _ = aggregate.load_results([tmp_path])
    with pytest.raises(aggregate.AggregateError, match="duplicate seed"):
        aggregate.summarize(loaded, metric="accuracy")


def test_summarize_rejects_partially_missing_metric(tmp_path):
    write_results(
        tmp_path, [make_result(1000), make_result(1001, accuracy=None)]
    )
    loaded, _ = aggregate.load_results([tmp_path])
    with pytest.raises(aggregate.AggregateError, match="some but not all"):
        aggregate.summarize(loaded, metric="accuracy")


def test_summarize_allows_wholly_missing_metric(tmp_path):
    write_results(
        tmp_path,
        [make_result(1000, accuracy=None), make_result(1001, accuracy=None)],
    )
    loaded, _ = aggregate.load_results([tmp_path])
    rows = aggregate.summarize(loaded, metric="accuracy")
    assert rows[0]["metric"] is None
    assert rows[0]["wall_time_s"]["n"] == 2


# ------------------------------------------------------- validation/rejection


def test_load_rejects_schema_invalid_file(tmp_path):
    write_results(tmp_path, [make_result(1000)])
    bad = make_result(1001)
    del bad["git_sha"]  # schema violation
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(aggregate.AggregateError, match="invalid results file"):
        aggregate.load_results([tmp_path])
    # --skip-invalid path: valid file survives, warning emitted
    loaded, warnings = aggregate.load_results([tmp_path], skip_invalid=True)
    assert len(loaded) == 1
    assert any("bad.json" in w for w in warnings)


def test_load_rejects_unknown_git_sha(tmp_path):
    bad = make_result(1000, git_sha="unknown")
    (tmp_path / "unknown_sha.json").write_text(json.dumps(bad))
    with pytest.raises(aggregate.AggregateError, match="git_sha"):
        aggregate.load_results([tmp_path])


def test_load_warns_on_null_cloud_cost(tmp_path):
    write_results(
        tmp_path,
        [make_result(1000, gpu_type="NVIDIA A100", cost_usd=None)],
    )
    loaded, warnings = aggregate.load_results([tmp_path])
    assert len(loaded) == 1
    assert any("cost_usd" in w for w in warnings)


def test_load_errors_on_empty_input(tmp_path):
    with pytest.raises(aggregate.AggregateError, match="no results"):
        aggregate.load_results([tmp_path])


# ------------------------------------------------------------ gpu-type rule


def test_refuses_mixed_gpu_type(tmp_path):
    write_results(
        tmp_path,
        [
            make_result(1000, gpu_type="cpu"),
            make_result(1001, gpu_type="NVIDIA A100", cost_usd=1.0),
        ],
    )
    loaded, _ = aggregate.load_results([tmp_path])
    with pytest.raises(aggregate.AggregateError, match="mixed gpu_type"):
        aggregate.enforce_single_gpu_type(loaded)


def test_main_refuses_mixed_gpu_but_filter_works(tmp_path, capsys):
    write_results(
        tmp_path,
        [
            make_result(1000, gpu_type="cpu"),
            make_result(1001, gpu_type="NVIDIA A100", cost_usd=1.0),
        ],
    )
    rc = aggregate.main([str(tmp_path)])
    assert rc == 2
    assert "mixed gpu_type" in capsys.readouterr().err

    rc = aggregate.main([str(tmp_path), "--gpu-type", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "`cpu`" in out


# ------------------------------------------------------------------ outputs


def test_main_writes_markdown_and_csv(tmp_path, capsys):
    results_dir = tmp_path / "res"
    results_dir.mkdir()
    accs = [0.94, 0.95, 0.96]
    write_results(
        results_dir, [make_result(1000 + i, accuracy=a) for i, a in enumerate(accs)]
    )
    out_md = tmp_path / "reports" / "table.md"
    out_csv = tmp_path / "reports" / "table.csv"
    rc = aggregate.main(
        [str(results_dir), "--out-md", str(out_md), "--out-csv", str(out_csv)]
    )
    assert rc == 0
    assert "| configs/synthetic_a.yaml |" in out_md.read_text()

    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert float(rows[0]["metric_mean"]) == pytest.approx(np.mean(accs))
    assert int(rows[0]["n"]) == 3
    expected_ci = scipy_stats.t.ppf(0.975, 2) * np.std(accs, ddof=1) / math.sqrt(3)
    assert float(rows[0]["metric_ci95"]) == pytest.approx(expected_ci)


def test_main_refuses_output_into_results_dir(tmp_path, capsys):
    results_dir = tmp_path / "res"
    results_dir.mkdir()
    write_results(results_dir, [make_result(1000)])
    rc = aggregate.main(
        [
            str(results_dir),
            "--out-md",
            str(results_io.RESULTS_DIR / "table.md"),
        ]
    )
    assert rc == 2
    assert "append-only" in capsys.readouterr().err
    assert not (results_io.RESULTS_DIR / "table.md").exists()
