#!/usr/bin/env python
"""Pre-launch anchors for program #23 (per-direction channel audit).

Regenerates every zero-GPU number that ``reports/channel-audit-preregistration.md``
commits BEFORE any Phase B run: the AR(1) surrogate nulls at each series length
the design produces, the published-DC-versus-null inflation that motivates the
band contrast (prereg repair R1), the per-matrix ``phi_hat`` implied by the
published frozen ESS, the two-channel phi sensitivity (repair R7), the
integrated-autocorrelation-time references and their branch table (repair R2),
and the P1 amplitude/power table.

WHY THIS EXISTS.  The pre-registration freezes thresholds whose consequences
have to be visible before they are frozen, and several of those consequences
are Monte-Carlo quantities rather than arithmetic.  Committing them as prose
alone makes them unauditable, so every table in prereg sections 2 and 4 is
reproduced here from one deterministic entry point.  Running it and checking
its output against those tables is launch precondition K0(h).

NOT A MEASUREMENT.  Nothing here reads a Phase B sidecar; nothing here
evaluates a criterion or emits a verdict (CLAUDE.md ground rule 1).  The only
observed data it touches is ``reports/frozen-probes.json``, the already-peeked
published aggregate that prereg section 0 discloses in full.

Deterministic: NumPy only, seeded RNGs only, no timestamps, sorted keys, no
GPU, no network -- identical inputs produce byte-identical outputs.

Usage:
    uv run --no-sync python scripts/channel_audit_anchors.py \
        --out-json reports/channel-audit-anchors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.stats.spectral import (  # noqa: E402
    _ar1_streams,
    ar1_surrogate_null,
    block_bootstrap_ci,
    channel_t,
    lag_ladder,
)

# Registered constants (prereg sections 3 and 5); mirrored here, never redefined.
BASE_SEED = 4242
TAU_REF_SEED = 4243  # deliberately != BASE_SEED, so the white control is not a tautology
BURN_IN = 5
NW_MAX_LAG = 8  # the Newey-West window's cap; the bandwidth rule gives L = 4 anyway
LADDER_MAX_LAG = 64
TAU_LAGS = (8, 16, 32, 64)
TAU_PRIMARY_K = 32
BOOTSTRAP_BLOCK = 64
BOOTSTRAP_REPS = 2000
CORE_PROBES = 3456  # 9-run B = 2000 core, prereg section 3 (repair R8)
PHASE_A_PHI = -0.34  # the peeked tracked-tier fit; the registered sensitivity value
N_OBS_PUBLISHED = 200  # the published frozen tier's series length

# (n after burn-in, where it occurs) -- every distinct series length in the design.
SERIES_LENGTHS: Tuple[Tuple[int, str], ...] = (
    (195, "frozen probe, 200-step run"),
    (187, "frozen probe, B = 8000 (192 steps)"),
    (45, "tracked refresh segment"),
    (37, "tracked last segment, B = 8000"),
)
PHI_GRID = (-0.20, -0.30, -0.34, -0.385, -0.40, -0.50)
PLANTED_AMPLITUDES = (0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
TAU_STREAMS: Tuple[Tuple[str, float, float], ...] = (
    ("white", 0.0, 1.0),
    ("ar1_phi_p0.20", 0.20, 1.5),
    ("ar1_phi_p0.50", 0.50, 3.0),
    ("ar1_phi_m0.34", -0.34, 0.66 / 1.34),
    ("ar1_phi_m0.385", -0.385, 0.615 / 1.385),
)


def _f(x: Any, nd: int = 5) -> float:
    """Round for stable JSON across platforms without losing the digits used."""
    return float(np.round(float(x), nd))


def _ratio(num: float, den: float) -> float:
    """``num / den``, NaN on an empty denominator.

    Every ratio here has a Monte-Carlo denominator, and at reduced ``--reps``
    (the test path) a 1e-4 rate can round to exactly zero.  A NaN is a reported
    non-number; a ZeroDivisionError is a crash in a script whose whole job is
    to be runnable before launch.
    """
    return float(num) / float(den) if float(den) != 0.0 else float("nan")


# --------------------------------------------------------------- null tables


def _null(phi: float, n: int, reps: int, seed: int, burn_in: int) -> Dict[str, Any]:
    return ar1_surrogate_null(
        phi=phi,
        n=n,
        reps=reps,
        seed=seed,
        burn_in=burn_in,
        max_lag=NW_MAX_LAG,
    )


def _abs_t(null: Dict[str, Any], channel: str) -> np.ndarray:
    return np.asarray(null[channel]["samples"]["abs_t_nw"], dtype=np.float64)


def _channel_summary(null: Dict[str, Any], channel: str) -> Dict[str, Any]:
    s = _abs_t(null, channel)
    q25, med, q75, q90 = np.quantile(s, [0.25, 0.50, 0.75, 0.90])
    return {
        "ess_over_n_median": _f(null[channel]["ess_over_n"]["median"]),
        "frac_ge_2": _f(float(np.mean(s >= 2.0))),
        "frac_ge_3": _f(float(np.mean(s >= 3.0))),
        "frac_ge_4": _f(float(np.mean(s >= 4.0))),
        "median_abs_t_nw": _f(med),
        "q25_abs_t_nw": _f(q25),
        "q75_abs_t_nw": _f(q75),
        "q90_abs_t_nw": _f(q90),
    }


def null_anchor_table(reps: int) -> Dict[str, Any]:
    """Prereg section 2's null-anchor table, at the Phase A phi."""
    out: Dict[str, Any] = {}
    for n, where in SERIES_LENGTHS:
        null = _null(PHASE_A_PHI, n + BURN_IN, reps, BASE_SEED, BURN_IN)
        out[str(n)] = {
            "alt": _channel_summary(null, "alt"),
            "dc": _channel_summary(null, "dc"),
            "phi": PHASE_A_PHI,
            "reps": reps,
            "where": where,
        }
    return out


def phi_sensitivity_table(reps: int) -> Dict[str, Any]:
    """Prereg section 2's two-channel phi-sensitivity table (repair R7).

    The previous DRAFT disclosed the DC channel's insensitivity only.  P1 is
    read on ``alt``, which moves the other way and moves further.
    """
    out: Dict[str, Any] = {}
    for phi in PHI_GRID:
        null = _null(phi, N_OBS_PUBLISHED, reps, BASE_SEED, BURN_IN)
        out["%.3f" % phi] = {
            "alt": _channel_summary(null, "alt"),
            "dc": _channel_summary(null, "dc"),
        }
    lo, hi = out["%.3f" % PHI_GRID[0]], out["%.3f" % PHI_GRID[-1]]
    out["_span"] = {
        "alt_frac_ge_4_ratio": _f(
            _ratio(hi["alt"]["frac_ge_4"], lo["alt"]["frac_ge_4"]), 2
        ),
        "alt_median_pct": _f(100.0 * (hi["alt"]["median_abs_t_nw"] / lo["alt"]["median_abs_t_nw"] - 1.0), 3),
        "dc_median_pct": _f(100.0 * (hi["dc"]["median_abs_t_nw"] / lo["dc"]["median_abs_t_nw"] - 1.0), 3),
        "note": "the DC channel moves down, the ALT channel (P1's channel) moves up and further",
        "phi_hi": PHI_GRID[-1],
        "phi_lo": PHI_GRID[0],
    }
    return out


# ------------------------------------------------- phi_hat from published ESS


def _ess_over_n(phi: float, n: int, reps: int, seed: int) -> float:
    null = ar1_surrogate_null(
        phi=phi, n=n, reps=reps, seed=seed, burn_in=BURN_IN,
        max_lag=NW_MAX_LAG, channels=("dc",),
    )
    return float(null["dc"]["ess_over_n"]["median"])


def _solve_phi(target: float, reps: int, seed: int, iters: int = 24) -> float:
    """Invert median ESS/n through THIS estimator by bisection (NumPy only).

    ESS/n is monotone decreasing in phi over the bracket, so plain bisection is
    enough and avoids a SciPy dependency the module contract does not carry.
    """
    lo, hi = -0.75, -0.02
    f_lo = _ess_over_n(lo, N_OBS_PUBLISHED, reps, seed) - target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _ess_over_n(mid, N_OBS_PUBLISHED, reps, seed) - target
        if (f_mid > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def published_aggregate(frozen_probes: Path) -> Dict[str, Any]:
    """The peeked published frozen tier, and the tau it already implies.

    ``ess = n c_0 / sigma_LR^2`` by the estimator's own definition, so
    ``tau(L = 4) = sigma_LR^2 / c_0 = n / ess``: the published ESS block IS a
    tau measurement at the bandwidth the accumulator used (prereg section 0,
    repair R3).
    """
    pub = json.loads(frozen_probes.read_text())
    ess = pub["pooled"]["ess"]
    t_nw = pub["pooled"]["t_nw"]
    n = N_OBS_PUBLISHED
    return {
        "dc_final_abs_t": {
            "frac_ge_2": _f(t_nw["frac_crossing"]["2"]),
            "frac_ge_3": _f(t_nw["frac_crossing"]["3"]),
            "frac_ge_4": _f(t_nw["frac_crossing"]["4"]),
            "median": _f(t_nw["final_abs_t"]["median"]),
            "q25": _f(t_nw["final_abs_t"]["q25"]),
            "q75": _f(t_nw["final_abs_t"]["q75"]),
        },
        "ess_over_n": {
            "max": _f(ess["max"] / n),
            "median": _f(ess["median"] / n),
            "min": _f(ess["min"] / n),
        },
        "implied_tau_at_L4": {
            "iqr": [_f(n / ess["q75"]), _f(n / ess["q25"])],
            "max": _f(n / ess["min"]),
            "median": _f(n / ess["median"]),
            "min": _f(n / ess["max"]),
            "note": "tau = n/ESS; max < 1 means every published probe has tau < 1",
        },
        "n_nw_floored": int(pub["pooled"]["n_nw_floored"]),
        "n_probes": int(pub["n_frozen_probes"]),
        "n_runs": int(pub["n_runs"]),
        "per_matrix_ess_over_n": {
            m: _f(d["ess"]["median"] / n) for m, d in sorted(pub["per_matrix"].items())
        },
        "per_matrix_n_nw_floored": {
            m: int(d["n_nw_floored"]) for m, d in sorted(pub["per_matrix"].items())
        },
    }


def instrument_null_contrast(
    pub: Dict[str, Any], reps: int, solve_reps: int
) -> Dict[str, Any]:
    """Repair R1: the published DC channel against the AR(1) null at its own phi.

    This is the finding that moves P1 off ``ar1_surrogate_null``: the
    instrument's own |t| distribution is inflated ~1.38x at every quantile and
    ~132x in the >= 4 tail relative to the AR(1) model of it, so an exceedance
    criterion calibrated against that model passes on the null.
    """
    phi_pooled = _solve_phi(pub["ess_over_n"]["median"], solve_reps, BASE_SEED)
    per_matrix = {
        m: _f(_solve_phi(v, solve_reps, BASE_SEED), 4)
        for m, v in sorted(pub["per_matrix_ess_over_n"].items())
    }
    null = _null(phi_pooled, N_OBS_PUBLISHED, reps, BASE_SEED, BURN_IN)
    dc, alt = _channel_summary(null, "dc"), _channel_summary(null, "alt")
    obs = pub["dc_final_abs_t"]

    # Equal-weight 6-matrix mixture: does phi heterogeneity explain the excess?
    mix: Dict[str, List[np.ndarray]] = {"alt": [], "dc": []}
    for i, phi in enumerate(sorted(per_matrix.values())):
        m_null = _null(phi, N_OBS_PUBLISHED, max(reps // 3, 1000), BASE_SEED + i, BURN_IN)
        for ch in ("alt", "dc"):
            mix[ch].append(_abs_t(m_null, ch))
    mixture = {}
    for ch in ("alt", "dc"):
        s = np.concatenate(mix[ch])
        q25, med, q75 = np.quantile(s, [0.25, 0.50, 0.75])
        mixture[ch] = {
            "frac_ge_4": _f(float(np.mean(s >= 4.0))),
            "median_abs_t_nw": _f(med),
            "q25_abs_t_nw": _f(q25),
            "q75_abs_t_nw": _f(q75),
        }

    kappa = obs["median"] / dc["median_abs_t_nw"]
    alt_samples = _abs_t(null, "alt")
    alt_med = float(np.median(alt_samples))
    return {
        "inflation_kappa_median_matched": _f(kappa),
        "mixture_null_6_matrices": mixture,
        "null_dc": dc,
        "observed_over_null_dc": {
            "frac_ge_2": _f(_ratio(obs["frac_ge_2"], dc["frac_ge_2"]), 2),
            "frac_ge_3": _f(_ratio(obs["frac_ge_3"], dc["frac_ge_3"]), 2),
            "frac_ge_4": _f(_ratio(obs["frac_ge_4"], dc["frac_ge_4"]), 2),
            "median": _f(_ratio(obs["median"], dc["median_abs_t_nw"]), 3),
            "q25": _f(_ratio(obs["q25"], dc["q25_abs_t_nw"]), 3),
            "q75": _f(_ratio(obs["q75"], dc["q75_abs_t_nw"]), 3),
        },
        "per_matrix_phi_hat": per_matrix,
        "per_matrix_phi_spread": _f(max(per_matrix.values()) - min(per_matrix.values()), 4),
        "phi_hat_pooled": _f(phi_pooled, 4),
        "previous_draft_p1_on_a_pure_null": {
            "frac_alt_ge_4_scaled": _f(float(np.mean(kappa * alt_samples >= 4.0))),
            "note": (
                "the previous DRAFT's P1a bar was 1.3 and its P1b floor 0.010; "
                "both pass on a stream with zero planted alternating signal"
            ),
            "null_frac_alt_ge_4": _f(alt["frac_ge_4"]),
            "ratio_alt_scaled": _f(float(np.median(kappa * alt_samples)) / alt_med),
        },
    }


# ----------------------------------------------------------- P1 band contrast


def band_contrast_power(
    phi: float, kappa: float, n: int, reps: int, seed: int, null_reps: int
) -> Dict[str, Any]:
    """Prereg section 4's P1 amplitude table (repair R1 / R5).

    Model: a channel-common multiplicative inflation ``kappa`` on |t| in BOTH
    channels -- the shape the published DC channel actually shows -- plus a
    homogeneous planted alternating amplitude ``A`` in units of the per-step
    noise sd.  A = 0 is the pure inflated null and must land in P1's FAIL row.
    """
    null = _null(phi, n, null_reps, BASE_SEED, 0)
    med = {ch: float(np.median(_abs_t(null, ch))) for ch in ("alt", "dc")}
    theta = 4.0 / med["dc"]

    rng = np.random.default_rng(seed)
    streams = _ar1_streams(rng, phi, n, reps)
    parity = (-1.0) ** np.arange(n)

    t_dc = np.abs(
        np.array([channel_t(streams[i], "dc", 0, max_lag=NW_MAX_LAG)["t_nw"] for i in range(reps)])
    ) * kappa
    T_dc = t_dc / med["dc"]
    ratio_dc = float(np.median(T_dc))
    frac_dc = float(np.mean(T_dc >= theta))

    rows: Dict[str, Any] = {}
    for amp in PLANTED_AMPLITUDES:
        t_alt = np.abs(
            np.array([
                channel_t(streams[i] + amp * parity, "alt", 0, max_lag=NW_MAX_LAG)["t_nw"]
                for i in range(reps)
            ])
        ) * kappa
        T_alt = t_alt / med["alt"]
        ratio_alt = float(np.median(T_alt))
        frac_alt = float(np.mean(T_alt >= theta))
        rows["%.2f" % amp] = {
            "band_contrast": _f(ratio_alt / ratio_dc, 4),
            "events_per_core_pool": _f(frac_alt * CORE_PROBES, 1),
            "frac_alt": _f(frac_alt),
            "ratio_alt_raw": _f(ratio_alt, 4),
            "tail_contrast": _f(_ratio(frac_alt, frac_dc), 2),
        }
    return {
        "alt_raw_threshold_at_theta": _f(theta * med["alt"], 3),
        "by_amplitude": rows,
        "frac_dc": _f(frac_dc),
        "kappa": _f(kappa),
        "n": n,
        "null_median_alt": _f(med["alt"]),
        "null_median_dc": _f(med["dc"]),
        "phi": phi,
        "ratio_dc_raw": _f(ratio_dc, 4),
        "reps": reps,
        "theta": _f(theta, 4),
    }


def channel_shape_profile(phi: float, reps: int) -> Dict[str, Any]:
    """K6's diagnostic: the two channels' calibrated quantile profiles.

    Under a channel-common inflation these coincide; under the AR(1) null they
    already do, to within ~1% through q90.  K6 fires when they do not.
    """
    null = _null(phi, N_OBS_PUBLISHED, reps, BASE_SEED, BURN_IN)
    out: Dict[str, Any] = {"phi": phi, "reps": reps}
    for ch in ("alt", "dc"):
        s = _abs_t(null, ch)
        q = np.quantile(s, [0.25, 0.50, 0.75, 0.90]) / np.median(s)
        out[ch] = {"q25": _f(q[0], 4), "q50": _f(q[1], 4), "q75": _f(q[2], 4), "q90": _f(q[3], 4)}
    out["max_abs_pct_divergence"] = _f(
        100.0 * max(
            abs(_ratio(out["alt"][k], out["dc"][k]) - 1.0) for k in ("q25", "q75", "q90")
        ),
        3,
    )
    return out


# ------------------------------------------------------------------- tau (P3)


def _tau_sample(phi: float, n: int, probes: int, seed: int, k: int) -> np.ndarray:
    """Per-probe ``1 + 2 sum_{j<=k} rho_j`` on the bias-corrected ladder."""
    rng = np.random.default_rng(seed)
    streams = _ar1_streams(rng, phi, n + BURN_IN, probes)
    out = np.empty(probes, dtype=np.float64)
    for i in range(probes):
        rho = lag_ladder(
            streams[i], burn_in=BURN_IN, max_lag=LADDER_MAX_LAG, bias_correct=True
        )["rho"]
        out[i] = 1.0 + 2.0 * float(np.sum(rho[:k]))
    return out


def tau_references(n: int, probes: int) -> Dict[str, Any]:
    """Repair R2: the estimator's own white-noise and AR(1) references.

    Both poolings are reported because the difference is the defect: a 32-lag
    sum is right-skewed, so its MEDIAN is biased low by ~12% and the previous
    DRAFT's decisive clause ("bootstrap upper end of tau < 1") fired on white
    noise.  The mean is unbiased for the sum and is the registered pooling.
    """
    out: Dict[str, Any] = {"n": n, "probes": probes}
    for k in TAU_LAGS:
        white = _tau_sample(0.0, n, probes, BASE_SEED, k)
        ci_mean = block_bootstrap_ci(
            white, block=BOOTSTRAP_BLOCK, reps=BOOTSTRAP_REPS, seed=BASE_SEED, statistic=np.mean
        )
        ci_med = block_bootstrap_ci(
            white, block=BOOTSTRAP_BLOCK, reps=BOOTSTRAP_REPS, seed=BASE_SEED, statistic=np.median
        )
        out["K%d" % k] = {
            "tau_white_mean_pooled": _f(ci_mean["point"], 4),
            "tau_white_mean_pooled_ci95": [_f(ci_mean["ci_lo"], 4), _f(ci_mean["ci_hi"], 4)],
            "tau_white_median_pooled": _f(ci_med["point"], 4),
            "tau_white_median_pooled_ci95": [_f(ci_med["ci_lo"], 4), _f(ci_med["ci_hi"], 4)],
        }
    return out


def tau_branch_table(n: int, probes: int) -> Dict[str, Any]:
    """Every registered P3 branch, and which stream produces it.

    The reference is drawn at ``TAU_REF_SEED`` rather than ``BASE_SEED``: with a
    shared seed the white-noise control returns tau_cal == 1.000 exactly and
    tests nothing (the bbp-prereg A2 failure mode).
    """
    ref = {k: float(np.mean(_tau_sample(0.0, n, probes, TAU_REF_SEED, k))) for k in TAU_LAGS}
    out: Dict[str, Any] = {
        "n": n,
        "probes": probes,
        "tau_white_reference": {"K%d" % k: _f(v, 4) for k, v in sorted(ref.items())},
        "tau_white_reference_seed": TAU_REF_SEED,
    }
    for label, phi, tau_true in TAU_STREAMS:
        by_k: Dict[str, Any] = {}
        verdicts = []
        for k in TAU_LAGS:
            ci = block_bootstrap_ci(
                _tau_sample(phi, n, probes, BASE_SEED, k) / ref[k],
                block=BOOTSTRAP_BLOCK,
                reps=BOOTSTRAP_REPS,
                seed=BASE_SEED,
                statistic=np.mean,
            )
            verdict = (
                "DECISIVE" if ci["ci_hi"] < 1.0
                else "FAIL" if ci["ci_lo"] > 1.0
                else "UNDECIDED"
            )
            verdicts.append(verdict)
            by_k["K%d" % k] = {
                "tau_cal": _f(ci["point"], 4),
                "tau_cal_ci95": [_f(ci["ci_lo"], 4), _f(ci["ci_hi"], 4)],
                "verdict": verdict,
            }
        out[label] = {
            "by_K": by_k,
            "k_stable": len(set(verdicts)) == 1,
            "phi": phi,
            "registered_verdict": verdicts[TAU_LAGS.index(TAU_PRIMARY_K)]
            if len(set(verdicts)) == 1 else "UNRESOLVED",
            "tau_true": _f(tau_true, 4),
        }
    return out


def tau_ar1_references(n: int, probes: int) -> Dict[str, Any]:
    """``tau_AR1(phi, n, K)`` -- P3's consistency-clause denominator."""
    out: Dict[str, Any] = {"n": n, "probes": probes}
    for phi in (PHASE_A_PHI, -0.385):
        out["%.3f" % phi] = {
            "K%d" % k: _f(float(np.mean(_tau_sample(phi, n, probes, BASE_SEED, k))), 4)
            for k in TAU_LAGS
        }
    return out


# ------------------------------------------------------------------- reporting


def build_anchors(args: argparse.Namespace) -> Dict[str, Any]:
    pub = published_aggregate(args.frozen_probes)
    contrast = instrument_null_contrast(pub, args.null_reps, args.solve_reps)
    kappa = contrast["inflation_kappa_median_matched"]
    phi_hat = contrast["phi_hat_pooled"]
    return {
        "config": {
            "base_seed": BASE_SEED,
            "bootstrap_block": BOOTSTRAP_BLOCK,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "burn_in": BURN_IN,
            "core_probes": CORE_PROBES,
            "ladder_max_lag": LADDER_MAX_LAG,
            "null_reps": args.null_reps,
            "nw_max_lag": NW_MAX_LAG,
            "power_reps": args.power_reps,
            "solve_reps": args.solve_reps,
            "tau_lags": list(TAU_LAGS),
            "tau_probes": args.tau_probes,
            "tau_reference_seed": TAU_REF_SEED,
        },
        "instrument_null_contrast": contrast,
        "k6_channel_shape": channel_shape_profile(phi_hat, args.null_reps),
        "null_anchors": null_anchor_table(args.null_reps),
        "p1_band_contrast_power": band_contrast_power(
            phi_hat, kappa, 195, args.power_reps, args.power_seed, args.null_reps
        ),
        "phi_sensitivity": phi_sensitivity_table(args.sensitivity_reps),
        "published_aggregate": pub,
        "tau_ar1_references": tau_ar1_references(195, args.tau_probes),
        "tau_branches": tau_branch_table(195, args.tau_probes),
        "tau_references_n187": tau_references(187, args.tau_probes),
        "tau_references_n195": tau_references(195, args.tau_probes),
    }


def to_markdown(a: Dict[str, Any]) -> str:
    c = a["instrument_null_contrast"]
    pub = a["published_aggregate"]
    lines: List[str] = [
        "# Channel audit -- pre-launch anchors (zero-GPU)",
        "",
        "Reproduction of every committed number in prereg sections 2 and 4.",
        "Descriptive only: no criterion is evaluated and no verdict is emitted.",
        "",
        "## Instrument null contrast (repair R1)",
        "",
        "| DC channel, n = 200 | q25 | median | q75 | >= 2 | >= 3 | >= 4 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| AR(1) null at phi_hat = %.3f | %.4f | %.4f | %.4f | %.5f | %.5f | %.5f |"
        % (
            c["phi_hat_pooled"], c["null_dc"]["q25_abs_t_nw"], c["null_dc"]["median_abs_t_nw"],
            c["null_dc"]["q75_abs_t_nw"], c["null_dc"]["frac_ge_2"], c["null_dc"]["frac_ge_3"],
            c["null_dc"]["frac_ge_4"],
        ),
        "| observed, published (%d) | %.4f | %.4f | %.4f | %.5f | %.5f | %.5f |"
        % (
            pub["n_probes"], pub["dc_final_abs_t"]["q25"], pub["dc_final_abs_t"]["median"],
            pub["dc_final_abs_t"]["q75"], pub["dc_final_abs_t"]["frac_ge_2"],
            pub["dc_final_abs_t"]["frac_ge_3"], pub["dc_final_abs_t"]["frac_ge_4"],
        ),
        "| ratio | %.2fx | %.2fx | %.2fx | %.1fx | %.1fx | %.1fx |"
        % tuple(
            c["observed_over_null_dc"][k]
            for k in ("q25", "median", "q75", "frac_ge_2", "frac_ge_3", "frac_ge_4")
        ),
        "",
        "6-matrix mixture null frac(|t_dc| >= 4): %.5f (phi heterogeneity does not explain it)"
        % c["mixture_null_6_matrices"]["dc"]["frac_ge_4"],
        "",
        "Previous DRAFT's P1 on a pure inflated null (kappa = %.4f, zero planted signal): "
        "ratio_alt %.4f, frac(|t_alt| >= 4) %.5f -- both clauses pass."
        % (
            c["inflation_kappa_median_matched"],
            c["previous_draft_p1_on_a_pure_null"]["ratio_alt_scaled"],
            c["previous_draft_p1_on_a_pure_null"]["frac_alt_ge_4_scaled"],
        ),
        "",
        "Per-matrix phi_hat: "
        + ", ".join("%.4f" % v for _, v in sorted(c["per_matrix_phi_hat"].items()))
        + " (spread %.4f)" % c["per_matrix_phi_spread"],
        "",
        "## Published frozen tier: the tau already on disk (repair R3)",
        "",
        "tau(L = 4) = n/ESS: median %.4f, IQR [%.4f, %.4f], max %.4f, min %.4f."
        % (
            pub["implied_tau_at_L4"]["median"], pub["implied_tau_at_L4"]["iqr"][0],
            pub["implied_tau_at_L4"]["iqr"][1], pub["implied_tau_at_L4"]["max"],
            pub["implied_tau_at_L4"]["min"],
        ),
        "Newey-West floorings: %d / %d." % (pub["n_nw_floored"], pub["n_probes"]),
        "",
        "## P1 band contrast, power (repairs R1 / R5)",
        "",
        "| planted A | band_contrast | tail_contrast | frac_alt | events / %d |" % CORE_PROBES,
        "| --- | --- | --- | --- | --- |",
    ]
    for amp, row in sorted(a["p1_band_contrast_power"]["by_amplitude"].items()):
        lines.append(
            "| %s | %.3f | %.2f | %.5f | %.1f |"
            % (amp, row["band_contrast"], row["tail_contrast"], row["frac_alt"],
               row["events_per_core_pool"])
        )
    lines += [
        "",
        "## tau (repair R2)",
        "",
        "| K | tau_white, MEAN pooled (registered) | tau_white, MEDIAN pooled (previous DRAFT) |",
        "| --- | --- | --- |",
    ]
    for k in TAU_LAGS:
        e = a["tau_references_n195"]["K%d" % k]
        lines.append(
            "| %d | %.4f %s | %.4f %s |"
            % (k, e["tau_white_mean_pooled"], e["tau_white_mean_pooled_ci95"],
               e["tau_white_median_pooled"], e["tau_white_median_pooled_ci95"])
        )
    lines += [
        "",
        "| stream | tau_true | tau_cal(K=%d) | CI95 | branch | K-stable |" % TAU_PRIMARY_K,
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, _, _ in TAU_STREAMS:
        e = a["tau_branches"][label]
        k = e["by_K"]["K%d" % TAU_PRIMARY_K]
        lines.append(
            "| %s | %.4f | %.4f | %s | %s | %s |"
            % (label, e["tau_true"], k["tau_cal"], k["tau_cal_ci95"], k["verdict"], e["k_stable"])
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--frozen-probes",
        type=Path,
        default=REPO_ROOT / "reports" / "frozen-probes.json",
        help="the published (already peeked) frozen-tier aggregate",
    )
    ap.add_argument("--out-json", type=Path, default=REPO_ROOT / "reports" / "channel-audit-anchors.json")
    ap.add_argument("--out-md", type=Path, default=None, help="optional markdown mirror")
    ap.add_argument("--null-reps", type=int, default=200_000, help="prereg section 5.6 registers 200000")
    ap.add_argument("--sensitivity-reps", type=int, default=60_000)
    ap.add_argument("--solve-reps", type=int, default=20_000, help="reps per bisection step for phi_hat")
    ap.add_argument("--power-reps", type=int, default=40_000)
    ap.add_argument("--power-seed", type=int, default=99)
    ap.add_argument("--tau-probes", type=int, default=CORE_PROBES)
    args = ap.parse_args(argv)

    anchors = build_anchors(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(anchors, indent=1, sort_keys=True) + "\n")
    md = to_markdown(anchors)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md + "\n")
    print(md)
    print("\nwrote %s" % args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
