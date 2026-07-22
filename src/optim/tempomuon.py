"""TempoMuon: Muon with a per-matrix temporal trust ratio (program #8).

The idea (docs/litreview/b-layer-temporal-trust-ratio.md, re-verified open
2026-07-22): a whole-matrix learning-rate gain driven by the *serial*
structure of that matrix's gradient stream — the "temporal third column" of
the trust-ratio family. Existing per-matrix gains in the Muon family are
driven by spatial norms (OrScale), noise magnitude (LANTON/NAMO/MoLS), or
smoothness models (Gluon); none by lag-1 serial correlation, which those
signal families provably cannot see (equal-variance AR(+rho) vs AR(-rho)).

Signal
    a_t = cos(G_t, G_{t-1}) per matrix (raw pre-momentum gradients, one
    extra buffer + one dot product per matrix per step), accumulated into a
    bias-corrected EMA rho_hat (the matrix-level, energy-weighted aggregate
    of the per-direction lag-1 autocorrelation population measured in
    WP1.2: LR-monotone, bulk-participating).

Controller (delta-bar-delta lifted to matrix granularity, in Muon)
    gain <- clip(gain * exp(kappa * (rho_hat - rho_star)),
                 gain_min, gain_max)
    applied as ``W -= lr_adjusted * gain * O``. The setpoint rho_star is
    *negative*: the healthy record regime carries substantial negative-rho
    occupancy (WP1.2), so driving rho_hat to 0 would fight a well-tuned run.
    kappa = 0 makes the optimizer bit-identical to stock Muon while still
    measuring rho_hat (passive/Phase-A mode).

Scope
    scope="per_matrix": one (rho_hat, gain) pair per parameter matrix.
    scope="global": per-step cosines from all matrices are pooled (mean)
    into a single EMA and a single shared gain — the GALA/CLARA-style
    global-control ablation that tests whether per-matrix granularity adds
    anything.

State & resumability: per-matrix quantities live in ``self.state[param]``
(prev-grad tensor + python floats), so ``state_dict`` round-trips. Global-
scope scalars are mirrored into ``param_groups[0]["tempo_global"]`` so they
serialize with the param groups.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.optim.muon import Muon, adjusted_lr_for_shape

_SCOPES = ("per_matrix", "global")


class TempoMuon(Muon):
    """Muon + per-matrix (or global) temporal trust-ratio gain."""

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
        rho_beta: float = 0.9,
        kappa: float = 0.0,
        rho_star: float = -0.2,
        gain_min: float = 0.25,
        gain_max: float = 1.0,
        warmup_steps: int = 25,
        scope: str = "per_matrix",
        history_every: int = 1,
    ) -> None:
        if not 0.0 < rho_beta < 1.0:
            raise ValueError(f"rho_beta must be in (0, 1), got {rho_beta}")
        if kappa < 0.0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        if not -1.0 <= rho_star <= 1.0:
            raise ValueError(f"rho_star must be in [-1, 1], got {rho_star}")
        if not 0.0 < gain_min <= gain_max:
            raise ValueError(f"need 0 < gain_min <= gain_max, got {gain_min}, {gain_max}")
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        if scope not in _SCOPES:
            raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
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
            rho_beta=rho_beta,
            kappa=kappa,
            rho_star=rho_star,
            gain_min=gain_min,
            gain_max=gain_max,
            warmup_steps=warmup_steps,
        )
        self.defaults.update(extra)
        for group in self.param_groups:
            for k, v in extra.items():
                group.setdefault(k, v)
        self.scope = scope
        self.history_every = int(history_every)
        self._step_count = 0
        self._pool: List[float] = []  # this step's cosines (global scope)
        self._history: List[Dict[str, Any]] = []
        self._labels: Dict[int, str] = {}  # id(param) -> stable label

    # ------------------------------------------------------------ global state

    @property
    def _gstate(self) -> Dict[str, float]:
        # Kept inside param_groups[0] so torch's state_dict serializes it.
        return self.param_groups[0].setdefault(
            "tempo_global", {"rho_raw": 0.0, "obs": 0, "gain": 1.0}
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _rho_hat(raw: float, obs: int, beta: float) -> Optional[float]:
        if obs == 0:
            return None
        return raw / (1.0 - beta**obs)

    @staticmethod
    def _advance_gain(
        gain: float, rho_hat: Optional[float], group: Dict[str, Any]
    ) -> float:
        if rho_hat is None or not math.isfinite(rho_hat):
            return gain
        factor = math.exp(group["kappa"] * (rho_hat - group["rho_star"]))
        return min(max(gain * factor, group["gain_min"]), group["gain_max"])

    def _observe(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> None:
        prev = state.get("tempo_prev_grad")
        if prev is not None:
            # fp32 throughout: the airbench model trains in fp16, where the
            # elementwise products of sum-reduction CE gradients overflow
            # (fp16 max 65504) — observed as the program-#8 Phase-A NaN
            # collapse at low LR. flatten() also normalizes conv shapes.
            g32 = G.detach().reshape(-1).float()
            p32 = prev.reshape(-1).float()
            num = torch.dot(g32, p32)
            den = g32.norm() * p32.norm()
            cos = float((num / den.clamp_min(1e-30)).item())
            if math.isfinite(cos):
                beta = group["rho_beta"]
                state["tempo_rho_raw"] = (
                    beta * state.get("tempo_rho_raw", 0.0) + (1.0 - beta) * cos
                )
                state["tempo_obs"] = state.get("tempo_obs", 0) + 1
                if self.scope == "global":
                    self._pool.append(cos)
                elif (
                    group["kappa"] != 0.0
                    and state["tempo_obs"] > group["warmup_steps"]
                ):
                    rho = self._rho_hat(
                        state["tempo_rho_raw"], state["tempo_obs"], beta
                    )
                    state["tempo_gain"] = self._advance_gain(
                        state.get("tempo_gain", 1.0), rho, group
                    )
            state["tempo_prev_grad"].copy_(G)
        else:
            state["tempo_prev_grad"] = G.detach().clone()
        state.setdefault("tempo_gain", 1.0)
        state.setdefault("tempo_rho_raw", 0.0)
        state.setdefault("tempo_obs", 0)

    def _flush_global_pool(self) -> None:
        if not self._pool:
            return
        cos_mean = sum(self._pool) / len(self._pool)
        self._pool = []
        g = self._gstate
        beta = self.defaults["rho_beta"]
        g["rho_raw"] = beta * g["rho_raw"] + (1.0 - beta) * cos_mean
        g["obs"] = g["obs"] + 1
        rho = self._rho_hat(g["rho_raw"], g["obs"], beta)
        if g["obs"] > self.defaults["warmup_steps"]:
            g["gain"] = self._advance_gain(g["gain"], rho, self.defaults)

    def _label(self, param: torch.Tensor) -> str:
        key = id(param)
        if key not in self._labels:
            shape = "x".join(str(s) for s in param.shape)
            self._labels[key] = f"matrix{len(self._labels)}_{shape}"
        return self._labels[key]

    # ------------------------------------------------------------------ hooks

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        self._observe(G, state, group)
        return super().pre_step(G, state, group)

    def post_step(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
    ) -> None:
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)
        if wd != 0.0:
            param.mul_(1.0 - lr * wd)
        alpha = adjusted_lr_for_shape(lr, param.shape, group.get("adjust_lr"))
        if self.scope == "global":
            gain = self._gstate["gain"]
        else:
            gain = state.get("tempo_gain", 1.0)
        param.add_(update, alpha=-alpha * gain)

    # -------------------------------------------------------------- step loop

    @torch.no_grad()
    def step(self, closure=None):
        if self.scope == "global":
            # Pool from the *previous* forward/backward step; the resulting
            # shared gain applies uniformly to every matrix this step.
            self._flush_global_pool()
        loss = super().step(closure)
        self._step_count += 1
        if self.history_every and self._step_count % self.history_every == 0:
            self._history.append(self._snapshot())
        return loss

    # ---------------------------------------------------------------- logging

    def _per_matrix_rows(self) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        for group in self.param_groups:
            beta = group["rho_beta"]
            for param in group["params"]:
                state = self.state.get(param)
                if not state or "tempo_obs" not in state:
                    continue
                rho = self._rho_hat(
                    state.get("tempo_rho_raw", 0.0), state.get("tempo_obs", 0), beta
                )
                rows[self._label(param)] = {
                    "rho": rho,
                    "gain": state.get("tempo_gain", 1.0),
                    "obs": state.get("tempo_obs", 0),
                }
        return rows

    def _snapshot(self) -> Dict[str, Any]:
        rows = self._per_matrix_rows()
        snap: Dict[str, Any] = {
            "step": self._step_count,
            "rho": {k: v["rho"] for k, v in rows.items()},
        }
        if self.scope == "global":
            g = self._gstate
            snap["global_rho"] = self._rho_hat(
                g["rho_raw"], g["obs"], self.defaults["rho_beta"]
            )
            snap["global_gain"] = g["gain"]
        else:
            snap["gain"] = {k: v["gain"] for k, v in rows.items()}
        return snap

    def tempo_stats(self) -> Dict[str, Any]:
        """Full telemetry for the results JSON (harness hook)."""
        out: Dict[str, Any] = {
            "scope": self.scope,
            "config": {
                k: self.defaults[k]
                for k in (
                    "rho_beta",
                    "kappa",
                    "rho_star",
                    "gain_min",
                    "gain_max",
                    "warmup_steps",
                )
            },
            "final": self._per_matrix_rows(),
            "history": self._history,
        }
        if self.scope == "global":
            g = self._gstate
            out["final_global"] = {
                "rho": self._rho_hat(g["rho_raw"], g["obs"], self.defaults["rho_beta"]),
                "gain": g["gain"],
                "obs": g["obs"],
            }
        return out
