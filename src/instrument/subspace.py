"""Tracked-pair subspace machinery (WP1.1, plan section 1.1).

For one Muon-managed weight matrix (flattened to 2-D, shape m x n) this module
maintains:

* the top-k1 singular pairs (u_i, v_i) of the momentum matrix M_t, refreshed
  every T_refresh steps via subspace iteration warm-started from the previous
  vectors (Dion-style amortized power iteration; 2-3 iterations per refresh),
  and
* k2 "bulk probes": persistent random Gaussian direction pairs orthogonalized
  against the top block (and against each other) on the same cadence -- the
  bulk is where noise lives; it must be observed directly.

Refresh returns a per-direction *alignment score*

    align_i = <u_new, u_old> * <v_new, v_old>  in [-1, 1]

with the sign of the product preserved: a joint sign flip of (u, v) leaves the
projection s_i = u^T G v invariant (align approx +1, no innovation), whereas a
flip of only one factor negates s_i and counts as a rotation.  Callers treat
``align_i < align_min`` as an innovation and reset that direction's statistics
(tracker.py).  After each refresh the top pairs are jointly sign-canonicalized
(<u_new, u_old> >= 0) so warm starts stay continuous.

Everything is deterministic given the ``torch.Generator`` passed in; no global
RNG state is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["RefreshResult", "TrackedSubspace"]


@dataclass
class RefreshResult:
    """Outcome of one subspace refresh.

    ``alignment[i]`` is the signed alignment score of direction ``i`` (top
    block first, then bulk probes) with its pre-refresh predecessor; +1 on the
    very first refresh (nothing rotated -- there was no predecessor to
    disagree with, and the tracker starts fresh anyway).
    ``sigma[i]`` is the singular-value estimate for top directions and the
    Rayleigh-like projection |u_i^T M v_i| for bulk probes.
    """

    alignment: torch.Tensor  # (k1 + k2,) float
    sigma: torch.Tensor  # (k1 + k2,) float
    first: bool


class TrackedSubspace:
    """Warm-started subspace iteration + orthogonalized bulk probes."""

    def __init__(
        self,
        m: int,
        n: int,
        *,
        k1: int = 16,
        k2: int = 16,
        iters: int = 2,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if k1 < 1:
            raise ValueError(f"k1 must be >= 1, got {k1}")
        if k2 < 0:
            raise ValueError(f"k2 must be >= 0, got {k2}")
        max_rank = min(m, n)
        if k1 + k2 > max_rank:
            raise ValueError(
                f"k1 + k2 = {k1 + k2} exceeds min(m, n) = {max_rank} "
                f"for a {m}x{n} matrix"
            )
        if iters < 1:
            raise ValueError(f"iters must be >= 1, got {iters}")
        self.m, self.n = int(m), int(n)
        self.k1, self.k2 = int(k1), int(k2)
        self.iters = int(iters)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self._gen = generator

        # Top block (None until the first refresh).
        self.U: Optional[torch.Tensor] = None  # (m, k1)
        self.V: Optional[torch.Tensor] = None  # (n, k1)
        self.sigma: Optional[torch.Tensor] = None  # (k1,)

        # Persistent Gaussian bases for the bulk probes (drawn once).
        if self.k2 > 0:
            self._probe_base_u = self._randn(self.m, self.k2)
            self._probe_base_v = self._randn(self.n, self.k2)
        else:
            self._probe_base_u = torch.zeros(self.m, 0, dtype=dtype, device=self.device)
            self._probe_base_v = torch.zeros(self.n, 0, dtype=dtype, device=self.device)
        self.Up: Optional[torch.Tensor] = None  # (m, k2)
        self.Vp: Optional[torch.Tensor] = None  # (n, k2)

        self.n_refreshes = 0

    # ------------------------------------------------------------------ utils

    @property
    def k_total(self) -> int:
        return self.k1 + self.k2

    def _randn(self, rows: int, cols: int) -> torch.Tensor:
        # Draw on CPU with the instance generator (device-independent
        # determinism), then move to the compute device.
        x = torch.randn(rows, cols, generator=self._gen, dtype=self.dtype)
        return x.to(self.device)

    @staticmethod
    def _orthonormalize(X: torch.Tensor) -> torch.Tensor:
        """Thin QR; falls back gracefully on rank deficiency via jitter-free QR."""
        Q, _ = torch.linalg.qr(X, mode="reduced")
        return Q

    def all_u(self) -> torch.Tensor:
        """(m, k1 + k2) left vectors, top block first. Refresh must have run."""
        assert self.U is not None and self.Up is not None
        return torch.cat([self.U, self.Up], dim=1)

    def all_v(self) -> torch.Tensor:
        """(n, k1 + k2) right vectors, top block first."""
        assert self.V is not None and self.Vp is not None
        return torch.cat([self.V, self.Vp], dim=1)

    def kinds(self) -> list:
        """Per-direction kind labels: 'top' for the k1 block, 'bulk' probes."""
        return ["top"] * self.k1 + ["bulk"] * self.k2

    # ---------------------------------------------------------------- refresh

    def refresh(self, M: torch.Tensor) -> RefreshResult:
        """One warm-started subspace refresh against the momentum matrix M.

        2-3 subspace iterations (per plan section 1.1), then a k1 x k1 SVD of
        the projected matrix B = U^T M V rotates the block onto individual
        singular-pair estimates (sorted by decreasing sigma).  Bulk probes are
        re-orthogonalized against the new top block per-probe (modified
        Gram-Schmidt -- QR would mix probe identities across columns).
        """
        if M.shape != (self.m, self.n):
            raise ValueError(f"expected M of shape {(self.m, self.n)}, got {tuple(M.shape)}")
        M = M.detach().to(dtype=self.dtype, device=self.device)

        first = self.U is None
        old_u = None if first else self.all_u()
        old_v = None if first else self.all_v()

        U = self._randn(self.m, self.k1) if first else self.U.clone()
        U = self._orthonormalize(U)
        n_iters = self.iters if not first else max(self.iters, 8)
        for _ in range(n_iters):
            V = self._orthonormalize(M.T @ U)
            U = self._orthonormalize(M @ V)

        # Rotate onto singular-pair estimates: B = P S Q^T -> U P, V Q.
        B = U.T @ M @ V  # (k1, k1)
        P, S, Qh = torch.linalg.svd(B)
        self.U = U @ P
        self.V = V @ Qh.T
        self.sigma = S  # sorted descending by construction

        if not first:
            # Joint sign canonicalization for warm-start continuity: flip
            # (u_i, v_i) together so <u_new, u_old> >= 0. s_i is invariant.
            sign = torch.sign((self.U * old_u[:, : self.k1]).sum(dim=0))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            self.U = self.U * sign
            self.V = self.V * sign

        # Bulk probes: per-probe Gram-Schmidt against the top block and the
        # previously accepted probes (identity-preserving, unlike QR).
        if self.k2 > 0:
            self.Up = self._gram_schmidt_probes(self._probe_base_u, self.U)
            self.Vp = self._gram_schmidt_probes(self._probe_base_v, self.V)
        else:
            self.Up = torch.zeros(self.m, 0, dtype=self.dtype, device=self.device)
            self.Vp = torch.zeros(self.n, 0, dtype=self.dtype, device=self.device)

        new_u, new_v = self.all_u(), self.all_v()
        if first:
            alignment = torch.ones(self.k_total, dtype=self.dtype, device=self.device)
        else:
            du = (new_u * old_u).sum(dim=0)
            dv = (new_v * old_v).sum(dim=0)
            alignment = du * dv

        # sigma for bulk probes: |u_p^T M v_p| (their "position" in the bulk).
        if self.k2 > 0:
            sig_bulk = torch.abs(((self.Up.T @ M) * self.Vp.T).sum(dim=1))
            sigma_all = torch.cat([self.sigma, sig_bulk])
        else:
            sigma_all = self.sigma

        self.n_refreshes += 1
        return RefreshResult(alignment=alignment, sigma=sigma_all, first=first)

    def _gram_schmidt_probes(self, base: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        """Orthogonalize each persistent probe against ``block`` and earlier
        probes, then normalize. Redraws a probe (deterministically, from the
        instance generator) only if it degenerates to (near) zero norm."""
        out = torch.empty_like(base)
        cols = [block]
        for j in range(base.shape[1]):
            p = base[:, j].clone()
            for C in cols:
                p = p - C @ (C.T @ p)
            norm = torch.linalg.norm(p)
            while float(norm) < 1e-6:
                p = self._randn(base.shape[0], 1)[:, 0]
                for C in cols:
                    p = p - C @ (C.T @ p)
                norm = torch.linalg.norm(p)
            p = p / norm
            out[:, j] = p
            cols.append(p.unsqueeze(1))
        return out

    # ------------------------------------------------------------ projections

    def project(self, G: torch.Tensor) -> torch.Tensor:
        """Per-direction scalar projections s_i = u_i^T G v_i, (k1 + k2,).

        Used on the RAW pre-momentum gradient each step (plan section 1.1) and
        on M for the per-step top-singular-value estimate.
        """
        U, V = self.all_u(), self.all_v()
        G = G.detach().to(dtype=self.dtype, device=self.device)
        return ((U.T @ G) * V.T).sum(dim=1)
