"""WP2.1 structural invariants (plan "Distributed scalability" section).

Enforced here, not just documented:

1. The optimizer update path (every module in src/optim except the
   experiment-harness adapter airbench_zoo.py) contains no curvature-probe
   code: CI grep for the forbidden estimator names ("hvp", "hessian").
   Curvature probes are a full extra pipeline pass at scale; the routing
   path must read eta*lambda from amplitude ratios only. The harness
   adapter is excluded because it legitimately wires the Phase-1
   *instrumentation* probes (validation-only, src.instrument-owned) into
   the instrumented experiment -- never into an optimizer update.
2. src.optim never imports src.instrument (same exclusion, same reason):
   routed.py owns its statistics via src.stats and a local copy of the
   power-iteration core.
3. Single-writer, per-matrix state: all tracked-tier state (subspace,
   classifier, gating RNG) lives in optimizer.state[param]; distinct
   matrices share no objects.
4. Projections are k rank-1 contractions -- k scalars per matrix per step,
   never a full-gradient gather.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim.routed import RoutedMuon

OPTIM_DIR = REPO_ROOT / "src" / "optim"
# The experiment-harness adapter (not an optimizer; wires the WP1.2
# instrumented experiment, whose probes live in src.instrument).
HARNESS_EXCLUDED = {"airbench_zoo.py"}


def _update_path_sources():
    files = [
        p
        for p in sorted(OPTIM_DIR.glob("*.py"))
        if p.name not in HARNESS_EXCLUDED
    ]
    assert any(p.name == "routed.py" for p in files)
    return files


def test_no_curvature_probe_names_in_update_path():
    forbidden = ("hvp", "hessian")
    for path in _update_path_sources():
        text = path.read_text().lower()
        for word in forbidden:
            assert word not in text, (
                f"forbidden curvature-probe reference '{word}' in "
                f"src/optim/{path.name}: the routing/update path must be "
                f"trajectory-only (distributed invariant 3)"
            )


def test_no_instrument_import_in_update_path():
    # Import statements only (docstrings may credit src/instrument as the
    # origin of the copy-adapted power-iteration core).
    pattern = re.compile(
        r"^\s*(from\s+src\.instrument|import\s+src\.instrument"
        r"|from\s+src\s+import\s+.*\binstrument\b)",
        re.MULTILINE,
    )
    for path in _update_path_sources():
        match = pattern.search(path.read_text())
        assert match is None, (
            f"src/optim/{path.name} imports src.instrument "
            f"({match.group(0).strip()!r}); routed statistics must come "
            f"from src.stats"
        )


def test_importing_routed_does_not_load_instrument():
    """Runtime check in a fresh interpreter: importing the routed optimizer
    (and the full registry) must not pull in src.instrument."""
    code = (
        "import sys; import src.optim.routed; import src.optim.registry; "
        "bad = [m for m in sys.modules if 'instrument' in m]; "
        "assert not bad, bad"
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, check=True, timeout=120
    )


def _stepped_optimizer():
    torch.manual_seed(1400)  # dev-seed range
    p1 = torch.nn.Parameter(torch.randn(9, 7))
    p2 = torch.nn.Parameter(torch.randn(8, 6))
    opt = RoutedMuon(
        [p1, p2], lr=1e-3, momentum=0.6, ns_dtype=torch.float32, k=3, n_min=5,
        seed=1400,
    )
    for _ in range(3):
        p1.grad = torch.randn(9, 7)
        p2.grad = torch.randn(8, 6)
        opt.step()
    return opt, p1, p2


def test_all_routing_state_is_per_matrix():
    opt, p1, p2 = _stepped_optimizer()
    t1 = opt.state[p1]["routing"]
    t2 = opt.state[p2]["routing"]
    # Distinct owner-rank state objects per matrix; nothing shared.
    assert t1 is not t2
    assert t1.classifier is not t2.classifier
    assert t1.classifier.stats is not t2.classifier.stats
    assert t1.gating_rng is not t2.gating_rng
    assert t1.U is not t2.U and t1.V is not t2.V
    # Momentum lives beside the routing state in the same per-matrix dict.
    assert "momentum_buffer" in opt.state[p1]
    assert "momentum_buffer" in opt.state[p2]
    # No transient raw-gradient reference is left behind after a step
    # (state_dict / checkpoint hygiene).
    assert "_routed_raw_grad" not in opt.state[p1]
    assert "_routed_raw_grad" not in opt.state[p2]


def test_projections_are_k_scalars_not_gathers():
    opt, p1, _ = _stepped_optimizer()
    tier = opt.state[p1]["routing"]
    s = tier.project(torch.randn(9, 7))
    assert s.shape == (tier.k,)  # k scalars per matrix per step
    assert tier.k == 3
    # last_gains mirrors the same k-vector.
    assert isinstance(tier.last_gains, np.ndarray)
    assert tier.last_gains.shape == (tier.k,)


def test_routing_decision_uses_only_src_stats_machinery():
    """The classifier object owned by the tier is the WP0.5-validated
    src.stats BatchRegimeClassifier -- no reimplementation."""
    from src.stats import BatchRegimeClassifier

    opt, p1, _ = _stepped_optimizer()
    tier = opt.state[p1]["routing"]
    assert type(tier.classifier) is BatchRegimeClassifier
