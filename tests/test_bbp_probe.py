"""Program #22 probe tests (prereg reports/bbp-prereg.md): doubling-tree merge
counts, curve structure, and checkpoint/artifact weight loading, on the CPU
tiny-model path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from test_nanogpt_port import _SingleRankDist, _write_shard
from test_nanogpt_tail import _tiny_config, _write_data


def test_bbp_probe_tree_counts_and_curves(tmp_path, monkeypatch):
    _SingleRankDist.install(monkeypatch)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)
    _write_data(tmp_path)
    from src.nanogpt.train import run_nanogpt, _checkpoint_path
    from src.nanogpt.config import NanoGPTConfig
    from src.nanogpt.bbp import run_bbp_probe

    # make a checkpoint at step 3
    pre = _tiny_config(tmp_path, max_steps=3,
                       checkpoint={"dir": str(tmp_path / "ckpt"), "every_steps": 3,
                                   "resume": False, "keep_on_success": True})
    run_nanogpt(pre, torch.device("cpu"))
    ckpt = _checkpoint_path(NanoGPTConfig.from_config(pre), 0)

    probe_cfg = _tiny_config(tmp_path)
    probe_cfg["experiment"] = "bbp_probe"
    probe_cfg["bbp"] = {"checkpoint": str(ckpt), "window_step": 3,
                        "n_chunks": 8, "data_file_index": 0}
    m = run_bbp_probe(probe_cfg, torch.device("cpu"))
    assert m["bbp_source"].startswith("checkpoint:")
    some_curve = next(iter(m["curves"].values()))
    assert some_curve["b_chunks"] == [1, 2, 4]
    assert some_curve["n_merges"] == [4, 2, 1]
    for name, c in m["curves"].items():
        assert all(0.0 <= a <= 1.0 for a in c["a_hat"]), name
    # raw cos preserved per level
    assert set(m["raw_cos"]) == {"1", "2", "4"}
    assert len(next(iter(m["raw_cos"]["1"].values()))) == 4
