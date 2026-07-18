"""WP1.1 subspace-iteration tests: planted singular structure recovery,
warm-start behavior, bulk-probe orthogonality, alignment semantics.

Seeds: dev seeds only (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


import torch

from src.instrument.subspace import TrackedSubspace


def _planted_matrix(m, n, sigmas, gen):
    """M = U0 diag(sigmas) V0^T with random orthonormal factors."""
    k = len(sigmas)
    U0, _ = torch.linalg.qr(torch.randn(m, k, generator=gen, dtype=torch.float32))
    V0, _ = torch.linalg.qr(torch.randn(n, k, generator=gen, dtype=torch.float32))
    S = torch.tensor(sigmas, dtype=torch.float32)
    return U0 @ torch.diag(S) @ V0.T, U0, V0, S


def _subspace_gap(A, B):
    """max principal-angle sin between span(A) and span(B) (orthonormal cols)."""
    s = torch.linalg.svdvals(A.T @ B)
    return float(torch.sqrt(torch.clamp(1.0 - s.min() ** 2, min=0.0)))


def test_recovers_planted_top_pairs():
    gen = torch.Generator().manual_seed(1234)
    m, n, k1 = 40, 30, 4
    M, U0, V0, S = _planted_matrix(m, n, [10.0, 8.0, 6.0, 4.0, 1.0, 0.5], gen)
    sub = TrackedSubspace(m, n, k1=k1, k2=0, iters=3, generator=gen)
    for _ in range(3):  # warm-started refreshes on a static matrix
        res = sub.refresh(M)

    # Subspace angle between recovered and true top-k1 blocks.
    assert _subspace_gap(sub.U, U0[:, :k1]) < 5e-3
    assert _subspace_gap(sub.V, V0[:, :k1]) < 5e-3
    # Per-pair singular value estimates.
    assert torch.allclose(res.sigma[:k1], S[:k1], atol=1e-2)
    # Individual pairs align with the planted ones (up to joint sign).
    for i in range(k1):
        du = float(sub.U[:, i] @ U0[:, i])
        dv = float(sub.V[:, i] @ V0[:, i])
        assert abs(du * dv) > 0.999
        assert du * dv > 0  # joint sign consistency: s_i is preserved


def test_warm_start_alignment_static_matrix():
    gen = torch.Generator().manual_seed(1235)
    m, n = 30, 20
    M, *_ = _planted_matrix(m, n, [9.0, 6.0, 3.0], gen)
    sub = TrackedSubspace(m, n, k1=3, k2=2, iters=2, generator=gen)
    first = sub.refresh(M)
    assert first.first
    assert torch.all(first.alignment == 1.0)
    second = sub.refresh(M)
    assert not second.first
    # Static matrix -> nothing rotates; alignment stays ~ +1 everywhere.
    assert torch.all(second.alignment > 0.999)


def test_rotation_produces_low_alignment():
    gen = torch.Generator().manual_seed(1236)
    m, n = 30, 20
    M1, U0, V0, _ = _planted_matrix(m, n, [9.0, 6.0], gen)
    sub = TrackedSubspace(m, n, k1=2, k2=0, iters=3, generator=gen)
    sub.refresh(M1)
    sub.refresh(M1)
    # Replace the 2nd pair with fresh directions orthogonal to the old ones.
    u_new = torch.randn(m, generator=gen)
    u_new -= U0 @ (U0.T @ u_new)
    u_new /= u_new.norm()
    v_new = torch.randn(n, generator=gen)
    v_new -= V0 @ (V0.T @ v_new)
    v_new /= v_new.norm()
    M2 = 9.0 * torch.outer(U0[:, 0], V0[:, 0]) + 6.0 * torch.outer(u_new, v_new)
    res = sub.refresh(M2)
    assert float(res.alignment[0]) > 0.99  # untouched pair stays aligned
    assert float(res.alignment[1]) < 0.5  # rotated pair flagged


def test_bulk_probes_orthonormal_and_orthogonal_to_top():
    gen = torch.Generator().manual_seed(1237)
    m, n = 40, 25
    M, *_ = _planted_matrix(m, n, [10.0, 5.0, 2.5], gen)
    sub = TrackedSubspace(m, n, k1=3, k2=4, iters=3, generator=gen)
    sub.refresh(M)
    for block, probes in ((sub.U, sub.Up), (sub.V, sub.Vp)):
        # Orthonormal probes.
        gram = probes.T @ probes
        assert torch.allclose(gram, torch.eye(4), atol=1e-5)
        # Orthogonal to the top block.
        cross = block.T @ probes
        assert float(cross.abs().max()) < 1e-5


def test_bulk_probes_persistent_across_refreshes():
    gen = torch.Generator().manual_seed(1238)
    m, n = 30, 20
    M, *_ = _planted_matrix(m, n, [8.0, 4.0], gen)
    sub = TrackedSubspace(m, n, k1=2, k2=3, iters=2, generator=gen)
    sub.refresh(M)
    up_before = sub.Up.clone()
    res = sub.refresh(M)  # static matrix: probes should barely move
    assert torch.allclose(sub.Up, up_before, atol=1e-4)
    assert torch.all(res.alignment[2:] > 0.999)


def test_projection_matches_planted_coefficients():
    gen = torch.Generator().manual_seed(1239)
    m, n = 30, 20
    M, U0, V0, _ = _planted_matrix(m, n, [9.0, 6.0], gen)
    sub = TrackedSubspace(m, n, k1=2, k2=1, iters=3, generator=gen)
    sub.refresh(M)
    # G with known coefficients along the planted pairs.
    G = 3.0 * torch.outer(U0[:, 0], V0[:, 0]) - 2.0 * torch.outer(U0[:, 1], V0[:, 1])
    s = sub.project(G)
    # Sign of each tracked pair is arbitrary but consistent between u and v,
    # so the projection magnitude must match; the bulk probe sees ~0.
    assert abs(abs(float(s[0])) - 3.0) < 1e-3
    assert abs(abs(float(s[1])) - 2.0) < 1e-3
    assert abs(float(s[2])) < 1e-3


def test_k_budget_validation():
    gen = torch.Generator().manual_seed(1240)
    try:
        TrackedSubspace(10, 8, k1=6, k2=4, generator=gen)
    except ValueError as e:
        assert "exceeds" in str(e)
    else:
        raise AssertionError("expected ValueError for k1 + k2 > min(m, n)")
