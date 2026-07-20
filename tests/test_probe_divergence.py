"""Twin-trajectory divergence probe (brainstorm program #1: compounding).

CPU-verifiable coverage of scripts/probe_divergence.py:

* the tiny-MLP twin harness runs end to end through the full probe path;
* DETERMINISM CONTROL: stock-vs-stock twins (identical optimizer, identical
  init, identical batches) diverge exactly zero at every step -- if this
  breaks, no divergence number from the probe is trustworthy;
* a state-side intervention twin actually moves the trajectory (rel_dist > 0);
* the experiment is registered in the standard runner and writes a
  schema-valid results JSON;
* the four airbench probe configs parse, carry dev seeds (>= 1000), and their
  twin_b specs resolve in the optimizer registry.

All seeds here are dev seeds (>= 1000).
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import results_io  # noqa: E402
from src.optim import build_optimizer  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PD = _load("probe_divergence_mod", "scripts/probe_divergence.py")

MLP_BASE = dict(
    experiment="probe_divergence",
    harness="mlp",
    steps=30,
    model=dict(in_dim=16, hidden_dim=24, out_dim=4),
    train=dict(batch_size=32),
)
MUON_SPEC = dict(
    name="muon", lr=1e-3, momentum=0.6, nesterov=True, ns_steps=3, ns_dtype="float32"
)


def test_stock_vs_stock_twins_do_not_diverge():
    """Determinism control: identical twins stay bit-identical (rel_dist == 0,
    update cosine == 1) at every step. torch CPU float32 is deterministic here;
    if a future torch breaks that, replace the exact check with a stated
    tolerance and document it -- do not silently loosen it."""
    cfg = dict(MLP_BASE, seed=1700, twin_a=dict(MUON_SPEC), twin_b=dict(MUON_SPEC))
    m = PD.run_probe_divergence(cfg, torch.device("cpu"))
    assert m["steps"] == MLP_BASE["steps"]
    assert all(d["rel_dist"] == 0.0 for d in m["divergence"])
    assert all(d["update_cosine"] == pytest.approx(1.0) for d in m["divergence"])
    assert m["final"]["rel_dist"] == 0.0
    assert m["final"]["max_rel_dist"] == 0.0


def test_state_side_intervention_moves_the_trajectory():
    """A state-damping twin against a stock-Muon twin: the buffer edits
    accumulate, so the trajectory genuinely diverges (rel_dist > 0, monotone
    nonzero by the end)."""
    cfg = dict(
        MLP_BASE,
        seed=1701,
        twin_a=dict(MUON_SPEC),
        twin_b=dict(
            name="routed",
            lr=1e-3,
            momentum=0.6,
            nesterov=True,
            ns_steps=3,
            ns_dtype="float32",
            k=4,
            n_min=5,
            t_refresh=20,
            beta=0.9,
            state_damping=True,
            seed=1500,
        ),
    )
    m = PD.run_probe_divergence(cfg, torch.device("cpu"))
    rels = [d["rel_dist"] for d in m["divergence"]]
    assert m["final"]["rel_dist"] > 0.0
    assert max(rels) > 0.0
    # Divergence appears only once routing activates (after the n_min gate),
    # so early steps are still zero -- the LAST step must be nonzero.
    assert rels[-1] > 0.0
    assert m["prediction"] == PD.PREDICTION


def test_probe_registered_and_writes_schema_valid_json(tmp_path):
    """The experiment is reachable through the standard runner and its output
    validates against the results schema (provenance, seed, metrics)."""
    run_mod = _load("run_mod_probe", "scripts/run.py")
    assert "probe_divergence" in run_mod.EXPERIMENT_REGISTRY

    cfg = dict(MLP_BASE, seed=1702, device="cpu",
               twin_a=dict(MUON_SPEC), twin_b=dict(MUON_SPEC))
    cfg_path = tmp_path / "probe_mlp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc = run_mod.main([str(cfg_path), "--out-dir", str(tmp_path)])
    assert rc == 0
    out = list(tmp_path.glob("probe_divergence_seed1702_*.json"))
    assert len(out) == 1
    result = results_io.load_result(out[0])  # validates on load
    assert result["experiment"] == "probe_divergence"
    assert result["seed"] == 1702
    assert result["metrics"]["harness"] == "mlp"


PROBE_CONFIGS = [
    "probe_divergence_output.yaml",
    "probe_divergence_state.yaml",
    "probe_divergence_goscconst050_output.yaml",
    "probe_divergence_goscconst050_state.yaml",
]


@pytest.mark.parametrize("fname", PROBE_CONFIGS)
def test_airbench_probe_configs_parse_and_resolve(fname):
    with open(REPO_ROOT / "configs" / "dev" / fname) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["experiment"] == "probe_divergence"
    assert cfg["harness"] == "airbench"
    assert cfg["steps"] == 200
    assert isinstance(cfg["seed"], int) and cfg["seed"] >= 1000
    # twin_b resolves and constructs on an airbench-shaped filter.
    name, kw = PD._resolve_spec(cfg["twin_b"])
    assert name == "routed"
    assert kw["seed"] >= 1000
    opt = build_optimizer(name, [torch.nn.Parameter(torch.randn(8, 4, 3, 3))], kw)
    from src.optim.routed import RoutedMuon

    assert isinstance(opt, RoutedMuon)


def test_output_and_state_goscconst_configs_differ_only_in_application_point():
    """The matched-gain pair must be identical except the state_damping flag
    (and the run seed) -- otherwise the comparison is confounded."""
    def load(f):
        with open(REPO_ROOT / "configs" / "dev" / f) as fh:
            return yaml.safe_load(fh)

    out = load("probe_divergence_goscconst050_output.yaml")
    st = load("probe_divergence_goscconst050_state.yaml")
    assert out["twin_b"]["state_damping"] is False
    assert st["twin_b"]["state_damping"] is True
    assert out["twin_b"]["g_osc_const"] == st["twin_b"]["g_osc_const"] == 0.5
    # Every twin_b key except state_damping matches.
    ob = {k: v for k, v in out["twin_b"].items() if k != "state_damping"}
    sb = {k: v for k, v in st["twin_b"].items() if k != "state_damping"}
    assert ob == sb
    assert out["twin_a"] == st["twin_a"]
