"""FIRMuon: Muon with a process-matched momentum filter (program #9).

Idea (novelty-verified open 2026-07-22; nearest neighbors: Greedy Alignment
2512.06370 — the optimizer-as-filter framing, but selects beta within the
EMA family; SGDF/DMR 2311.02818/2603.06120 — "Wiener" gain from lag-0
variance ratios under a white-noise assumption; Kuehn & Rosenow 2306.05300 —
the anti-correlation phenomenon, analysis only): the momentum EMA is a
fixed low-pass filter, but the measured per-direction gradient process on
this substrate is approximately MA(1) with negative lag-1 autocorrelation
(ACF ~ (1, -0.29, ~0, ...) for top directions). At matched mean lag
(staleness), the variance-optimal linear filter beats the truncated EMA by
1.4-1.8x and the record's Nesterov-EMA by 2.5-4.7x (median, offline
kill-tests on 3,456 directions/tau — scratchpad wiener_killtest.py,
2026-07-22). FIRMuon synthesizes that filter online:

    rho1_hat = EMA[cos(G_t, G_{t-1})] per matrix (fp32; program-#8 signal)
    Gamma    = tridiagonal Toeplitz(1, rho1_hat) + ridge*I
    taps w   = argmin w' Gamma w   s.t.  sum w = 1,  sum i*w_i = tau
    M_t      = sum_i w_i G_{t-i}   -> Newton-Schulz -> update

tau (target mean lag) replaces momentum beta as the smoothing knob. During
warm-up (buffer filling / rho1_hat immature) the optimizer takes stock
Nesterov-Muon steps, bit-identical to Muon (unit-tested).

Controls: ``force_rho`` pins rho1_hat to a constant — force_rho=0 gives the
white-noise-optimal kernel at the same tau, isolating "exploits *measured*
anti-correlation" from "different kernel family" (the program's placebo).

Memory: n_taps gradient buffers per matrix (airbench: trivial; an IIR
realization is the path to LM scale and is out of scope for Phase B).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.optim.muon import Muon

RHO_CLIP = 0.45  # keep the tridiagonal Toeplitz comfortably positive-definite


def synthesize_taps(
    rho1: float, tau: float, n_taps: int, ridge: float
) -> torch.Tensor:
    """Closed-form solution of min w'(Gamma+ridge I)w s.t. [1,i]'w=[1,tau]."""
    L = n_taps
    G = torch.eye(L, dtype=torch.float64) * (1.0 + ridge)
    idx = torch.arange(L - 1)
    G[idx, idx + 1] = rho1
    G[idx + 1, idx] = rho1
    A = torch.stack(
        [torch.ones(L, dtype=torch.float64), torch.arange(L, dtype=torch.float64)],
        dim=1,
    )
    b = torch.tensor([1.0, tau], dtype=torch.float64)
    Gi_A = torch.linalg.solve(G, A)
    lam = torch.linalg.solve(A.T @ Gi_A, b)
    return (Gi_A @ lam).to(torch.float32)


class FIRMuon(Muon):
    """Muon with online process-matched FIR momentum."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr: Optional[str] = None,
        ns_dtype: torch.dtype = torch.bfloat16,
        n_taps: int = 12,
        tau: float = 1.5,
        ridge: float = 0.05,
        rho_beta: float = 0.9,
        warmup_steps: int = 15,
        force_rho: Optional[float] = None,
        history_every: int = 1,
    ) -> None:
        if n_taps < 2:
            raise ValueError(f"n_taps must be >= 2, got {n_taps}")
        if not 0.0 <= tau <= n_taps - 1:
            raise ValueError(f"tau must be in [0, n_taps-1], got {tau}")
        if ridge < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        if not 0.0 < rho_beta < 1.0:
            raise ValueError(f"rho_beta must be in (0, 1), got {rho_beta}")
        if warmup_steps < 2:
            raise ValueError(f"warmup_steps must be >= 2, got {warmup_steps}")
        if force_rho is not None and not -1.0 <= force_rho <= 1.0:
            raise ValueError(f"force_rho must be in [-1, 1], got {force_rho}")
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr=adjust_lr,
            ns_dtype=ns_dtype,
        )
        extra = dict(
            n_taps=n_taps,
            tau=tau,
            ridge=ridge,
            rho_beta=rho_beta,
            warmup_steps=warmup_steps,
        )
        self.defaults.update(extra)
        for group in self.param_groups:
            for k, v in extra.items():
                group.setdefault(k, v)
        self.force_rho = force_rho
        self.history_every = int(history_every)
        self._step_count = 0
        self._history: List[Dict[str, Any]] = []
        self._labels: Dict[int, str] = {}

    # ---------------------------------------------------------------- helpers

    def _label(self, param: torch.Tensor) -> str:
        key = id(param)
        if key not in self._labels:
            shape = "x".join(str(s) for s in param.shape)
            self._labels[key] = f"matrix{len(self._labels)}_{shape}"
        return self._labels[key]

    def _rho1(self, state: Dict[str, Any], group: Dict[str, Any]) -> Optional[float]:
        if self.force_rho is not None:
            return float(self.force_rho)
        obs = state.get("fir_obs", 0)
        if obs == 0:
            return None
        raw = state.get("fir_rho_raw", 0.0)
        rho = raw / (1.0 - group["rho_beta"] ** obs)
        return min(max(rho, -RHO_CLIP), RHO_CLIP)

    def _observe(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> None:
        buf: List[torch.Tensor] = state.setdefault("fir_grads", [])
        if buf:
            g32 = G.detach().reshape(-1).float()
            p32 = buf[0].reshape(-1).float()
            num = torch.dot(g32, p32)
            den = (g32.norm() * p32.norm()).clamp_min(1e-30)
            cos = float((num / den).item())
            if math.isfinite(cos):
                beta = group["rho_beta"]
                state["fir_rho_raw"] = (
                    beta * state.get("fir_rho_raw", 0.0) + (1.0 - beta) * cos
                )
                state["fir_obs"] = state.get("fir_obs", 0) + 1
        buf.insert(0, G.detach().clone())
        del buf[group["n_taps"]:]

    # ------------------------------------------------------------------ hooks

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        self._observe(G, state, group)
        buf: List[torch.Tensor] = state["fir_grads"]
        rho = self._rho1(state, group)
        ready = (
            len(buf) >= group["n_taps"]
            and rho is not None
            and (self.force_rho is not None or state.get("fir_obs", 0) >= group["warmup_steps"])
        )
        # Stock Nesterov momentum is maintained throughout so the warm-up
        # path is bit-identical to Muon and the buffer survives resumes.
        stock = super().pre_step(G, state, group)
        if not ready:
            state["fir_active"] = False
            return stock
        state["fir_active"] = True
        taps = synthesize_taps(rho, group["tau"], group["n_taps"], group["ridge"])
        state["fir_taps"] = taps
        out32 = torch.zeros(G.shape, dtype=torch.float32, device=G.device)
        for w, g in zip(taps.tolist(), buf):
            out32.add_(g.float(), alpha=w)
        return out32.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self._step_count += 1
        if self.history_every and self._step_count % self.history_every == 0:
            snap: Dict[str, Any] = {"step": self._step_count, "rho": {}, "active": {}}
            for group in self.param_groups:
                for param in group["params"]:
                    st = self.state.get(param)
                    if not st or "fir_grads" not in st:
                        continue
                    lbl = self._label(param)
                    snap["rho"][lbl] = self._rho1(st, group)
                    snap["active"][lbl] = bool(st.get("fir_active", False))
            self._history.append(snap)
        return loss

    # ---------------------------------------------------------------- logging

    def tempo_stats(self) -> Dict[str, Any]:
        """Telemetry via the harness's tempo_stats hook."""
        final: Dict[str, Any] = {}
        for group in self.param_groups:
            for param in group["params"]:
                st = self.state.get(param)
                if not st or "fir_grads" not in st:
                    continue
                lbl = self._label(param)
                taps = st.get("fir_taps")
                final[lbl] = {
                    "rho": self._rho1(st, group),
                    "obs": st.get("fir_obs", 0),
                    "active": bool(st.get("fir_active", False)),
                    "taps": None if taps is None else [round(float(t), 5) for t in taps],
                }
        return {
            "scope": "fir",
            "config": {
                k: self.defaults[k]
                for k in ("n_taps", "tau", "ridge", "rho_beta", "warmup_steps")
            },
            "force_rho": self.force_rho,
            "final": final,
            "history": self._history,
        }
