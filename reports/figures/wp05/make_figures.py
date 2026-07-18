"""Generate WP0.5 validation figures + recovered-vs-true tables.

Deterministic: fixed dev seeds (>= 1000), same scenario parameters as the
test suite (tests/test_stats_*.py). Writes PNGs next to this file and
prints markdown tables to stdout for embedding in the report.

Run from anywhere:  uv run python reports/figures/wp05/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.stats import (
    DirectionStats,
    Regime,
    RegimeClassifier,
    ar1,
    concat_segments,
    drifting_mean,
    gaussian_noise,
    oscillation,
)

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito subset, validated colorblind-safe (fixed assignment, never cycled)
C_BETA = {0.9: "#0072B2", 0.99: "#E69F00"}  # series identity: EMA timescale
C_GREEN = "#009E73"
C_VERM = "#D55E00"
INK = "#333333"
GRID = dict(color="#dddddd", linewidth=0.6)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)

BETAS = [0.9, 0.99]

# ---------------------------------------------------------------- 1. AR(1) rho
RHOS = [-0.8, -0.4, 0.0, 0.4, 0.8]
NOISE_SCALES = [0.5, 2.0]
SEEDS = [1234, 5678]
N, BURN = 8000, 3000


def recovered_rho(stream, beta):
    st = DirectionStats(beta)
    vals = []
    for t, s in enumerate(stream):
        st.update(s)
        if t >= BURN:
            vals.append(float(st.rho_corrected))
    return float(np.mean(vals))


rho_rows = []  # (beta, rho_true, noise_scale, seed, recovered)
for beta in BETAS:
    for rho in RHOS:
        for ns in NOISE_SCALES:
            for seed in SEEDS:
                rec = recovered_rho(ar1(N, rho, noise_scale=ns, seed=seed), beta)
                rho_rows.append((beta, rho, ns, seed, rec))

fig, ax = plt.subplots(figsize=(5.2, 4.6))
lim = (-1.0, 1.0)
xs = np.linspace(*lim, 2)
ax.fill_between(xs, xs - 0.05, xs + 0.05, color="#000000", alpha=0.06, lw=0,
                label="±0.05 tolerance")
ax.plot(xs, xs, color="#999999", lw=1.0, ls="--", label="identity (perfect)")
jit = {0.9: -0.012, 0.99: 0.012}
for beta in BETAS:
    pts = [(r[1] + jit[beta], r[4]) for r in rho_rows if r[0] == beta]
    ax.scatter(*zip(*pts), s=22, color=C_BETA[beta], alpha=0.85,
               edgecolors="white", linewidths=0.4, label=f"β = {beta}")
ax.set_xlabel("true AR(1) ρ")
ax.set_ylabel("recovered ρ  (time-avg bias-corrected, post burn-in)")
ax.set_title("AR(1) lag-1 autocorrelation recovery\n"
             "(2 noise scales × 2 dev seeds per point column)")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xticks(RHOS)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "rho_recovery.png")
plt.close(fig)

# ------------------------------------------------------ 2. oscillation eta*lambda
CASES = [(0.8, 100, True), (1.0, 300, False), (1.1, 200, False)]
DECAY_MARGIN = 0.05
N_MIN_OSC = 30

osc_rows = []  # (beta, r, true_el, implied_el, regime, decaying)
for beta in BETAS:
    for r, n, expected_decay in CASES:
        clf = RegimeClassifier(beta=beta, tau_sig=4.0, tau_noise=2.0,
                               rho_osc=0.5, n_min=N_MIN_OSC)
        hist = [clf.update(s) for s in oscillation(n, r, amplitude=1.0)]
        osc_rows.append(
            (beta, r, 1.0 + r, float(clf.stats.implied_eta_lambda),
             hist[-1].value, bool(clf.stats.is_decaying(DECAY_MARGIN)),
             expected_decay)
        )

fig, ax = plt.subplots(figsize=(5.2, 4.6))
true_els = [1.0 + r for r, _, _ in CASES]
xs = np.linspace(1.7, 2.2, 2)
ax.fill_between(xs, xs - 0.1, xs + 0.1, color="#000000", alpha=0.06, lw=0,
                label="±0.1 tolerance")
ax.plot(xs, xs, color="#999999", lw=1.0, ls="--", label="identity (perfect)")
jit = {0.9: -0.006, 0.99: 0.006}
for beta in BETAS:
    pts = [(row[2] + jit[beta], row[3]) for row in osc_rows if row[0] == beta]
    ax.scatter(*zip(*pts), s=34, color=C_BETA[beta], edgecolors="white",
               linewidths=0.5, label=f"β = {beta}")
for r, _, _ in CASES:
    ax.annotate(f"r = {r}", (1.0 + r, 1.0 + r - 0.115), ha="center", fontsize=8,
                color="#666666")
ax.set_xlabel("true η·λ  =  1 + r")
ax.set_ylabel("implied η·λ from amplitude ratio")
ax.set_title("Implied η·λ recovery on pure oscillation s(t) = A·(−r)^t")
ax.set_xlim(1.7, 2.2)
ax.set_ylim(1.7, 2.2)
ax.set_xticks(true_els)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "eta_lambda_recovery.png")
plt.close(fig)

# ------------------------------------------------- 3. regime-switch timeline
N_MIN_SW = 15
segments = [
    gaussian_noise(400, seed=3111),
    gaussian_noise(300, mean=5.0, seed=3112),
    oscillation(300, 1.0, amplitude=6.0),
    gaussian_noise(400, seed=3113),
]
expected_seq = [Regime.NOISE, Regime.SIGNAL, Regime.OSCILLATING, Regime.NOISE]
stream, switches = concat_segments(*segments)
boundaries = [0] + switches

switch_runs = {}
for beta in BETAS:
    clf = RegimeClassifier(beta=beta, tau_sig=4.0, tau_noise=2.5, rho_osc=0.5,
                           n_min=N_MIN_SW, z_reset=3.0, innov_needed=2,
                           innov_window=4, z_quiet=0.4, quiet_window=6)
    hist = [clf.update(s) for s in stream]
    switch_runs[beta] = (hist, list(clf.reset_steps))

REGIME_Y = {Regime.NOISE: 0, Regime.SIGNAL: 1, Regime.OSCILLATING: 2}
REGIME_C = {Regime.NOISE: "#999999", Regime.SIGNAL: C_GREEN,
            Regime.OSCILLATING: C_VERM}

fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.4), sharex=True,
                         height_ratios=[1.6, 1, 1])
ax0 = axes[0]
ax0.plot(stream, lw=0.5, color="#555555")
ax0.set_ylabel("s(t)")
ax0.set_title("Regime-switch scenario: noise → signal(μ=5) → oscillation(±6) → noise")
for b in boundaries[1:]:
    for ax in axes:
        ax.axvline(b, color="#000000", lw=0.8, ls=":", alpha=0.55)
seg_names = ["noise", "signal", "oscillation", "noise"]
ends = boundaries[1:] + [len(stream)]
for name, b, e in zip(seg_names, boundaries, ends):
    ax0.annotate(name, ((b + e) / 2, ax0.get_ylim()[1] * 0.9), ha="center",
                 fontsize=8, color="#666666")

for ax, beta in zip(axes[1:], BETAS):
    hist, resets = switch_runs[beta]
    ys = np.array([REGIME_Y[h] for h in hist])
    t = np.arange(len(ys))
    for reg, y in REGIME_Y.items():
        mask = ys == y
        ax.scatter(t[mask], ys[mask], s=3, color=REGIME_C[reg], marker="s")
    for i, r in enumerate(resets):
        ax.axvline(r, color=C_BETA[0.9], lw=0.9, alpha=0.8,
                   label="confidence reset" if i == 0 else None)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["noise", "signal", "oscillating"])
    ax.set_ylim(-0.5, 2.5)
    ax.set_ylabel(f"β = {beta}")
    ax.legend(loc="center right", fontsize=7)
axes[-1].set_xlabel("step")
fig.tight_layout()
fig.savefig(OUT / "switch_timeline.png")
plt.close(fig)

# ------------------------------------------------------------ 4. tables (stdout)


def md(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


print("### TABLE rho\n")
print(md(
    ["β", "true ρ", "noise scale", "seed", "recovered ρ", "|err|", "within ±0.05"],
    [(b, f"{r:+.1f}", ns, sd, f"{rec:+.4f}", f"{abs(rec - r):.4f}",
      "yes" if abs(rec - r) <= 0.05 else "NO")
     for b, r, ns, sd, rec in rho_rows],
))

print("\n### TABLE osc\n")
print(md(
    ["β", "r", "true η·λ", "implied η·λ", "|err|", "final regime",
     "decay flag", "flag correct"],
    [(b, r, f"{te:.1f}", f"{ie:.4f}", f"{abs(ie - te):.4f}", reg, dec,
      "yes" if dec == exp else "NO")
     for b, r, te, ie, reg, dec, exp in osc_rows],
))

# drift t-stat measured vs analytic
print("\n### TABLE drift\n")
DRIFT_A, PERIOD, SEED_D, ND, BURND = 0.2, 4000, 2024, 8000, 2000
drift_rows = []
for beta, tau in [(0.99, 4.0), (0.9, 1.5)]:
    for snr in [0.1, 1.0, 10.0]:
        st = DirectionStats(beta)
        s_stream = drifting_mean(ND, snr, drift_amplitude=DRIFT_A,
                                 period=PERIOD, seed=SEED_D)
        ts = np.empty(ND)
        for t, s in enumerate(s_stream):
            st.update(s)
            ts[t] = float(st.t_stat)
        ess_inf = (1 + beta) / (1 - beta)
        t_idx = np.arange(BURND, ND)
        snr_t = snr * (1 + DRIFT_A * np.sin(2 * np.pi * t_idx / PERIOD))
        analytic = float(np.mean(snr_t)) * np.sqrt(ess_inf)
        frac = float(np.mean(np.abs(ts[BURND:]) > tau))
        pred = ("sustained" if snr * (1 - DRIFT_A) * np.sqrt(ess_inf) > 2 * tau
                else "absent" if snr * (1 + DRIFT_A) * np.sqrt(ess_inf) < tau
                else "ambiguous")
        drift_rows.append((beta, tau, snr, f"{analytic:.2f}",
                           f"{float(np.mean(ts[BURND:])):.2f}",
                           f"{frac:.3f}", pred))
print(md(["β", "τ", "SNR", "analytic E[t]", "measured mean t",
          "frac |t|>τ (post-burn)", "analytic prediction"], drift_rows))

# switch reclassification delays
print("\n### TABLE switch\n")
sw_rows = []
for beta in BETAS:
    hist, resets = switch_runs[beta]
    for name, b, e, want in zip(seg_names, boundaries, ends, expected_seq):
        first = next(i for i in range(b, len(hist)) if hist[i] == want)
        sw_rows.append((beta, f"{name} @ t={b}", want.value, first - b,
                        f"<= {N_MIN_SW}",
                        "yes" if first - b <= N_MIN_SW else "NO"))
    sw_rows.append((beta, "resets fired at", str(resets), "", "", ""))
print(md(["β", "segment (start)", "target regime", "delay (steps)",
          "bound (n_min)", "within bound"], sw_rows))

print("\nFigures written to", OUT)
