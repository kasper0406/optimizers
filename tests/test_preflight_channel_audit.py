"""Tests for the program #23 channel-audit launch gate.

scripts/preflight_channel_audit.py is the thing that decides whether the 15
GPU runs may start. Two properties matter and are asserted here against the
real documents on disk:

* it BLOCKS while `reports/channel-audit-preregistration.md` says DRAFT or
  carries an unfrozen threshold row, and stops blocking on a REGISTERED copy
  with the table filled (so the gate is not permanently red by construction);
* it CONFIRMS that the five configs on disk are the run set section 3 of that
  document registers, and turns red on any drift between them -- the check
  that a stale header comment cannot provide.

No GPU, no training, no writes outside tmp_path.
"""

import importlib.util
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
import yaml


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_module("preflight_channel_audit", "scripts/preflight_channel_audit.py")

PREREG_TEXT = preflight.PREREG.read_text()


@pytest.fixture
def registered_prereg(tmp_path, monkeypatch):
    """A copy of the pre-registration with the status flipped and table filled."""
    text = PREREG_TEXT.replace(
        "Status: **DRAFT — not registered.**", "Status: **REGISTERED.**", 1
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and re.match(r"\|\s*\d+\s*\|", stripped):
            if stripped.endswith("| |") or stripped.endswith("|  |"):
                line = line.rstrip()[:-1] + " frozen |"
        lines.append(line)
    path = tmp_path / "channel-audit-preregistration.md"
    path.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(preflight, "PREREG", path)
    return path


@pytest.fixture
def config_copy(tmp_path, monkeypatch):
    """The five configs copied to tmp_path so a test may mutate one."""
    out = tmp_path / "dev"
    out.mkdir()
    for src in sorted(preflight.CONFIG_DIR.glob(f"{preflight.CONFIG_STEM}*.yaml")):
        shutil.copy(src, out / src.name)
    monkeypatch.setattr(preflight, "CONFIG_DIR", out)
    return out


def test_parsers_read_the_real_documents():
    rows = preflight.parse_freeze_table(PREREG_TEXT)
    assert [r["row"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)]
    run_set = preflight.parse_run_set(PREREG_TEXT)
    assert sum(cell["runs"] for cell in run_set.values()) == 15
    instr = preflight.parse_registered_instrumentation(PREREG_TEXT)
    assert instr["frozen_probes"] == {
        "enabled": True,
        "k3": 64,
        "max_lag": 32,
        "decimate": 1,
    }
    assert instr["recipe.compile"] is False


def test_gate_blocks_while_the_prereg_is_a_draft():
    ok, detail = preflight.check_prereg_status(PREREG_TEXT)
    assert not ok and "DRAFT" in detail
    ok, detail = preflight.check_threshold_freeze_table(PREREG_TEXT)
    # Row count is not pinned here: the checklist grows whenever a repair adds
    # a threshold (revision R1 took it from 19 to 23). What is pinned is that
    # EVERY row is still unfrozen, which is the property the gate turns on.
    n_rows = len(preflight.parse_freeze_table(PREREG_TEXT))
    assert n_rows >= 19
    assert not ok and f"{n_rows} of {n_rows}" in detail


def test_gate_clears_on_a_registered_prereg_with_a_frozen_table(registered_prereg):
    text = registered_prereg.read_text()
    assert preflight.check_prereg_status(text)[0]
    assert preflight.check_threshold_freeze_table(text)[0]


def test_configs_on_disk_are_the_registered_run_set():
    sweep = preflight._load_sweep_module()
    ok, detail = preflight.check_run_set(PREREG_TEXT, sweep)
    assert ok, detail
    assert "15 runs" in detail
    ok, detail = preflight.check_instrumentation(PREREG_TEXT)
    assert ok, detail


def test_run_set_check_catches_config_drift(config_copy):
    sweep = preflight._load_sweep_module()
    path = config_copy / f"{preflight.CONFIG_STEM}.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["sweep"]["seeds"] = [1800, 1801, 1802]
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    ok, detail = preflight.check_run_set(PREREG_TEXT, sweep)
    assert not ok
    assert "1800" in detail and "HUMAN decision" in detail


def test_instrumentation_check_catches_a_probe_setting_drift(config_copy):
    path = config_copy / f"{preflight.CONFIG_STEM}_b8000.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["instrumentation"]["frozen_probes"]["k3"] = 16
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    ok, detail = preflight.check_instrumentation(PREREG_TEXT)
    assert not ok
    assert "frozen_probes" in detail


def test_main_exits_nonzero_while_blocked(capsys):
    assert preflight.main([]) == 1
    out = capsys.readouterr().out
    assert "BLOCKED prereg_status" in out
    assert "OK      run_set_matches_prereg" in out


def test_a_failing_parse_blocks_rather_than_raises(monkeypatch, tmp_path):
    path = tmp_path / "channel-audit-preregistration.md"
    path.write_text("# not a pre-registration\n")
    monkeypatch.setattr(preflight, "PREREG", path)
    results = dict((name, ok) for name, ok, _ in preflight.run_checks())
    assert results["prereg_status"] is False
    assert results["threshold_freeze_table"] is False
