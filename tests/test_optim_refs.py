"""WP0.4: reference Newton-Schulz implementations + NS primitive tests.

This module serves two purposes:

1. It holds *verbatim* copies of the vendored Newton-Schulz reference
   implementations (decorators stripped, nothing else changed). The other
   test_optim_* files import these as the independent expectation for any
   NS-based update step, per the WP0.4 DoD ("assert against a reference NS
   implementation copied verbatim from vendor").
2. It tests the src/optim/newton_schulz.py primitives against those copies
   (bitwise) and against numpy SVD where the rule is exact (DynMuon's
   sigma -> sigma^p via SVD), plus the near-orthonormality property tests.

All fixed matrices are literal values; no randomness is used anywhere in the
expectations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim.newton_schulz import (
    DYNMUON_NS_COEFFS,
    MUON_NS_COEFFS,
    dynmuon_fast_spectral,
    dynmuon_newtonschulz,
    spectral_power_via_svd,
    zeropower_via_newtonschulz5,
)

# --------------------------------------------------------------------------
# Verbatim vendor copies (the independent NS references for all optim tests)
# --------------------------------------------------------------------------


def airbench_ns_reference(G, steps=3, eps=1e-7):
    """Copied verbatim from vendor/airbench/airbench94_muon.py:32-54
    (function body of ``zeropower_via_newtonschulz5``; the ``@torch.compile``
    decorator and the docstring are stripped, nothing else changed)."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def dynmuon_ns_reference(G, epsilon=1e-7):
    """Copied verbatim from vendor/DynMuon/dynmuon/newton_schulz_triton.py:
    306-333 (``zeropower_via_newtonschulz5``; ``@torch.compile`` decorator and
    docstring stripped, nothing else changed)."""
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# Fixed literal test matrices (4x3 tall, 3x4 wide, well conditioned).
FIXED_4x3 = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [2.0, 1.0, 0.0],
    ]
)
FIXED_3x4 = FIXED_4x3.T.contiguous()


# --------------------------------------------------------------------------
# Bitwise agreement with the vendored references
# --------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [3, 5])
@pytest.mark.parametrize("mat", [FIXED_4x3, FIXED_3x4], ids=["tall", "wide"])
def test_zeropower_matches_airbench_reference_bitwise(mat, steps):
    ours = zeropower_via_newtonschulz5(mat, steps=steps)
    ref = airbench_ns_reference(mat.clone(), steps=steps)
    assert ours.dtype == torch.bfloat16
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("mat", [FIXED_4x3, FIXED_3x4], ids=["tall", "wide"])
def test_dynmuon_ns_matches_vendor_reference_bitwise(mat):
    ours = dynmuon_newtonschulz(mat)
    ref = dynmuon_ns_reference(mat.clone())
    assert ours.dtype == torch.bfloat16
    assert torch.equal(ours, ref)


def test_ns_coefficient_constants_match_vendor():
    # airbench94_muon.py:43
    assert MUON_NS_COEFFS == (3.4445, -4.7750, 2.0315)
    # newton_schulz_triton.py:311-317
    assert DYNMUON_NS_COEFFS == [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]


def test_zeropower_does_not_modify_input():
    mat = FIXED_4x3.clone()
    zeropower_via_newtonschulz5(mat, steps=5)
    dynmuon_newtonschulz(mat)
    assert torch.equal(mat, FIXED_4x3)


# --------------------------------------------------------------------------
# Property: NS output has near-orthonormal singular structure
# --------------------------------------------------------------------------


def test_muon_ns_output_near_orthonormal():
    """Quintic Muon NS: singular values land in ~U(0.5, 1.5) (reference
    docstring), and the singular *directions* of the input are preserved
    (the iteration is an odd polynomial in X, so U, V are exactly preserved
    in exact arithmetic)."""
    out = zeropower_via_newtonschulz5(FIXED_4x3, steps=5, dtype=torch.float32)
    svals = torch.linalg.svdvals(out)
    assert torch.all(svals > 0.3) and torch.all(svals < 1.7), svals

    U, _, Vh = torch.linalg.svd(FIXED_4x3, full_matrices=False)
    D = U.T @ out @ Vh.T
    offdiag = D - torch.diag(torch.diagonal(D))
    assert offdiag.abs().max() < 1e-3


def test_dynmuon_ns_output_near_orthonormal():
    """DynMuon's coefficient schedule converges tighter than the Muon quintic:
    singular values within a few % of 1 (measured ~[1.00, 1.02] on this
    matrix in float32)."""
    out = dynmuon_newtonschulz(FIXED_4x3, dtype=torch.float32)
    svals = torch.linalg.svdvals(out)
    assert torch.all(svals > 0.9) and torch.all(svals < 1.1), svals


# --------------------------------------------------------------------------
# DynMuon spectral shaping: exact SVD rule vs numpy; poly rule vs formula
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [-0.25, -0.5, -1.0])
@pytest.mark.parametrize("mat", [FIXED_4x3, FIXED_3x4], ids=["tall", "wide"])
def test_spectral_power_via_svd_matches_numpy(mat, p):
    """sigma -> sigma^p is exact via SVD; expectation independently in numpy
    float64: G = U S V^T maps to U S^p V^T."""
    ours = spectral_power_via_svd(mat, p)
    u, s, vt = np.linalg.svd(mat.numpy().astype(np.float64), full_matrices=False)
    expected = (u * s**p) @ vt
    np.testing.assert_allclose(ours.numpy(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("p", [-0.25, -0.5])
def test_fast_spectral_matches_written_formula(p, order):
    """Recompute DynMuon's fast_spectral (dynmuon.py:717-772) from the written
    rule, using the verbatim vendor NS copy for the Muon base:

        scale = ||X||_F + eps;  Xn = X/scale;  Y = NS(Xn)
        E = Xn Xn^T - I; delta = p/2
        C = I + delta E [+ 0.5 delta (delta-1) E^2]
        out = C Y * scale^p        (transpose dance for tall matrices)
    """
    eps = 1e-7
    X = FIXED_4x3.to(torch.float32)
    Xt = X.T  # reference operates on the wide orientation for tall inputs
    scale = torch.norm(Xt) + eps
    Xn = Xt / scale
    Y = dynmuon_ns_reference(Xn).to(torch.float32)
    E = Xn @ Xn.T - torch.eye(3)
    delta = 0.5 * p
    C = torch.eye(3) + delta * E
    if order == 2:
        C = C + 0.5 * delta * (delta - 1.0) * (E @ E)
    expected = ((C @ Y) * scale**p).T

    ours = dynmuon_fast_spectral(FIXED_4x3, p=p, eps=eps, order=order)
    torch.testing.assert_close(ours, expected, atol=1e-5, rtol=1e-5)


def test_fast_spectral_p_zero_is_pure_ns():
    """Reference keeps an exact p == 0 branch: plain NS on the normalized
    input (note: normalized by ||X||_F + eps *before* NS, so this differs
    from NS(X) only through the eps in the norm)."""
    eps = 1e-7
    ours = dynmuon_fast_spectral(FIXED_4x3, p=0.0, eps=eps)
    Xt = FIXED_4x3.T
    Xn = Xt / (torch.norm(Xt) + eps)
    expected = dynmuon_ns_reference(Xn).to(torch.float32).T
    torch.testing.assert_close(ours, expected, atol=1e-6, rtol=1e-6)
