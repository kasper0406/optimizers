"""Newton-Schulz orthogonalization / spectral-shaping primitives (WP0.4).

Two NS variants are used by the baseline zoo, each matching a vendored
reference implementation coefficient-for-coefficient:

1. ``zeropower_via_newtonschulz5`` -- classic quintic Muon iteration with the
   single coefficient tuple (3.4445, -4.7750, 2.0315), matching
   ``vendor/airbench/airbench94_muon.py:32-54`` (which uses steps=3) and the
   AdaMuon paper (arXiv:2507.11005, Sec. 2.1: same a, b, c with T=5).

2. ``dynmuon_newtonschulz`` -- DynMuon's 5-tuple coefficient schedule,
   matching the reference (non-Triton) implementation in
   ``vendor/DynMuon/dynmuon/newton_schulz_triton.py:306-333``
   (``zeropower_via_newtonschulz5`` there; the Triton kernel at lines 337-373
   uses the identical constants).

Also ports DynMuon's spectral shaping for p != 0:

3. ``dynmuon_fast_spectral`` -- polynomial correction for sigma -> sigma^p,
   port of ``vendor/DynMuon/dynmuon/dynmuon.py:717-772`` (``fast_spectral``).

4. ``spectral_power_via_svd`` -- exact sigma -> sigma^p via SVD, port of
   ``vendor/DynMuon/dynmuon/dynmuon.py:684-711`` (``shape_with_p_lam_via_svd``).

All functions are pure (no in-place modification of their input). ``dtype``
defaults to bfloat16 to match the references; unit/property tests may pass
float32 for tight tolerances.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Classic Muon quintic coefficients -- identical to
# vendor/airbench/airbench94_muon.py:43 and modded-nanogpt's historical Muon.
MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)

# DynMuon coefficient schedule -- identical to
# vendor/DynMuon/dynmuon/newton_schulz_triton.py:311-317.
DYNMUON_NS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def zeropower_via_newtonschulz5(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Quintic Newton-Schulz orthogonalization of a 2D matrix.

    Port of vendor/airbench/airbench94_muon.py:32-54 with the iteration count
    exposed (airbench calls it with steps=3; AdaMuon/modded-nanogpt-era Muon
    use steps=5). Coefficients are kept identical to the reference. As the
    reference docstring notes, the output is US'V^T with S'_ii ~ U(0.5, 1.5),
    not exactly UV^T.
    """
    assert len(G.shape) == 2
    a, b, c = MUON_NS_COEFFS
    X = G.to(dtype)
    X = X / (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def dynmuon_newtonschulz(
    G: Tensor,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """DynMuon's Newton-Schulz iteration (their 5-tuple coefficient schedule).

    Port of vendor/DynMuon/dynmuon/newton_schulz_triton.py:306-333
    (``zeropower_via_newtonschulz5``). Supports batched (..., m, n) input like
    the reference.
    """
    X = G.to(dtype=dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b, c in DYNMUON_NS_COEFFS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def dynmuon_fast_spectral(
    G: Tensor,
    p: float,
    eps: float = 1e-7,
    order: int = 2,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """DynMuon's fast spectral shaping sigma -> sigma^p (polynomial correction).

    Port of vendor/DynMuon/dynmuon/dynmuon.py:717-772 (``fast_spectral``),
    with ``newton_schulz_triton`` replaced by :func:`dynmuon_newtonschulz`
    (same constants, reference non-Triton path). ``dtype`` controls the NS
    working precision (reference: bfloat16 inside the Triton/NS kernel; the
    surrounding polynomial runs in float32 exactly as in the reference).
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    orig_dtype = G.dtype
    X = G.to(torch.float32)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    scale = X.norm(dim=(-2, -1), keepdim=True) + eps
    Xn = X / scale

    # Muon base on normalized input
    Y_mu = dynmuon_newtonschulz(Xn, eps=eps, dtype=dtype).to(torch.float32)

    # Pure Muon path (reference keeps this exact-equality branch)
    if p == 0.0:
        U = Y_mu
        if transposed:
            U = U.mT
        return U.to(orig_dtype)

    A = Xn @ Xn.mT
    m = A.size(-1)
    I = torch.eye(m, device=A.device, dtype=A.dtype).view(
        (1,) * (A.ndim - 2) + (m, m)
    )

    # Polynomial correction
    delta = 0.5 * p
    E = A - I

    if order == 1:
        C = I + delta * E
    else:
        E2 = E @ E
        C = I + delta * E + 0.5 * delta * (delta - 1.0) * E2

    U = C @ Y_mu
    U = U * scale.pow(p)

    if transposed:
        U = U.mT

    return U.to(orig_dtype)


def spectral_power_via_svd(G: Tensor, p: float, eps: float = 1e-7) -> Tensor:
    """Exact spectral shaping sigma -> sigma^p via SVD.

    Port of vendor/DynMuon/dynmuon/dynmuon.py:684-711
    (``shape_with_p_lam_via_svd``): G = U S V^T maps to U S^p V^T
    (S_new = S * (S^2)^((p-1)/2) = S^p).
    """
    orig_dtype = G.dtype
    X = G.to(torch.float32)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        if not torch.isfinite(S).all():
            raise RuntimeError("Non-finite singular values from SVD")
    except RuntimeError:
        U, S, Vh = torch.linalg.svd(
            X + eps * torch.randn_like(X),
            full_matrices=False,
        )

    s2 = S * S
    exp = 0.5 * (p - 1.0)
    S_new = S * (s2).pow(exp)

    Q = (U * S_new.unsqueeze(-2)) @ Vh

    if transposed:
        Q = Q.mT
    return Q.to(orig_dtype)
