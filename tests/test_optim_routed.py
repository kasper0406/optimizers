"""WP2.1 unit tests: Routed Muon v0 (src/optim/routed.py).

Synthetic momentum matrices with PLANTED signal / noise / oscillating
directions: gradient streams are constructed through fixed orthonormal
direction pairs with known temporal statistics (all seeds >= 1000, dev-seed
policy). Verified here:

* after burn-in the classifier assigns the planted labels and the applied
  per-direction gains match the plan section 2.1 gain rules;
* the actual O_t correction equals
  O_ns + sum_i (g_i - 1) (u_i^T O_ns v_i) u_i v_i^T within tolerance,
  and deviates from stock NS only in a rank-<=(#gated) subspace (NS
  property preserved for the bulk);
* confidence gating: bit-for-bit stock Muon before n_min observations;
* subspace refresh resets confidence for rotated directions (start-in-
  SIGNAL, gains revert to 1, re-classification after n_min fresh samples);
* rho_ignored (ablation 4c) and random_gating (ablation 4d) behave as
  specified;
* with both channels disabled, routed == stock Muon bit-for-bit on a fixed
  trajectory.

The streams are seed-deterministic; every stochastic quantity asserted on
is fixed by the dev seeds below.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim import MatrixOptimizer, OPTIMIZER_REGISTRY, build_optimizer
from src.optim.muon import Muon
from src.optim.newton_schulz import zeropower_via_newtonschulz5
from src.optim.routed import RoutedMuon
from src.stats import Regime

SEED = 1234  # dev seed (>= 1000)

# Shared optimizer settings for the synthetic tests: float32 NS for tight
# tolerances (both Muon and RoutedMuon get the identical setting), fast
# statistics (beta 0.9), small confidence gate.
BASE_KW = dict(
    lr=1e-3,
    momentum=0.6,
    nesterov=True,
    ns_steps=3,
    weight_decay=0.0,
    ns_dtype=torch.float32,
)
ROUTE_KW = dict(
    k=3,
    t_refresh=1000,  # single refresh at step 1: subspace locked to G_1
    beta=0.9,
    tau_sig=4.0,
    tau_noise=2.0,
    rho_osc=0.5,
    g_noise=0.25,
    n_min=30,
    align_min=0.9,
    seed=SEED,
)


# ----------------------------------------------------------------- streams


def orthonormal_pairs(m, n, count, seed):
    gen = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, count, generator=gen))
    V, _ = torch.linalg.qr(torch.randn(n, count, generator=gen))
    return U, V


class PlantedStream:
    """G_t = sum_j s_j(t) u_j v_j^T + eps * background noise."""

    def __init__(self, m, n, U, V, coeff_fns, bg=1e-3, seed=SEED):
        self.m, self.n = m, n
        self.U, self.V = U, V
        self.coeff_fns = coeff_fns  # list of t -> float (t starts at 1)
        self.bg = bg
        self.rng = np.random.default_rng(seed)

    def grad(self, t):
        s = torch.tensor(
            [fn(t) for fn in self.coeff_fns], dtype=torch.float32
        )
        G = (self.U * s.unsqueeze(0)) @ self.V.T
        G = G + self.bg * torch.from_numpy(
            self.rng.standard_normal((self.m, self.n))
        ).to(torch.float32)
        return G


def three_regime_stream(m=14, n=12, r_osc=1.05, seed=SEED):
    """Planted signal / noise / oscillating coefficient streams.

    The signal jitter (0.3 around mean 1.2) is deliberately not tiny: for a
    near-constant stream the lag-1 autocovariance a - mu^2 is an
    ill-conditioned difference and the rho estimator swings wildly -- a
    property of the (canonical, WP0.5-validated) statistics, not of the
    optimizer. First samples are pinned so the step-1 subspace refresh sees
    three well-separated singular values (1.2 > 0.42 > 0.35 >> bg).
    """
    U, V = orthonormal_pairs(m, n, 3, seed)
    rng = np.random.default_rng(seed + 1)

    def s_signal(t):
        return 1.2 if t == 1 else 1.2 + 0.3 * rng.standard_normal()

    def s_noise(t):
        return 0.35 if t == 1 else 0.3 * rng.standard_normal()

    def s_osc(t):
        return 0.4 * (-r_osc) ** t

    return PlantedStream(m, n, U, V, [s_signal, s_noise, s_osc], seed=seed + 2)


def match_planted(tier, u, v, min_align=0.8):
    """Index of the tracked pair aligned with planted (u, v); -1 if none."""
    align = torch.abs((tier.U.T @ u) * (tier.V.T @ v))
    i = int(torch.argmax(align))
    return i if float(align[i]) >= min_align else -1


def drive(opt, param, stream, steps, start=1):
    for t in range(start, start + steps):
        param.grad = stream.grad(t)
        opt.step()


# ------------------------------------------------- registry & construction


def test_routed_registered_and_constructible():
    assert OPTIMIZER_REGISTRY["routed"] is RoutedMuon
    params = [torch.nn.Parameter(torch.zeros(6, 5))]
    opt = build_optimizer("routed", params, {})
    assert isinstance(opt, MatrixOptimizer)
    assert isinstance(opt, RoutedMuon)


def test_plan_defaults():
    opt = RoutedMuon([torch.nn.Parameter(torch.zeros(4, 3))])
    assert opt.k == 16
    assert opt.t_refresh == 50
    assert opt.beta == 0.99
    assert opt.g_noise == 0.25
    assert opt.n_min == 50
    assert opt.g_osc_min == 0.1
    assert opt.enable_noise_channel and opt.enable_oscillation_channel
    assert not opt.rho_ignored and not opt.random_gating
    assert opt.state_damping is False  # output-side by default
    assert opt.defaults["momentum"] == 0.95  # Muon-side defaults intact
    assert opt.defaults["nesterov"] is True


def test_invalid_hyperparameters_rejected():
    p = [torch.nn.Parameter(torch.zeros(4, 3))]
    with pytest.raises(ValueError, match="k must"):
        RoutedMuon(p, k=0)
    with pytest.raises(ValueError, match="g_noise"):
        RoutedMuon(p, g_noise=1.5)
    with pytest.raises(ValueError, match="beta"):
        RoutedMuon(p, beta=1.0)
    with pytest.raises(ValueError, match="tau_noise"):
        RoutedMuon(p, tau_sig=1.0, tau_noise=2.0)  # classifier validation


def test_steps_on_conv_shaped_param():
    param = torch.nn.Parameter(torch.full((8, 4, 3, 3), 0.5))
    opt = build_optimizer("routed", [param], {"k": 4, "n_min": 2})
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.isfinite(param.detach()).all()
    tier = opt.state[param]["routing"]
    assert tier.k == 4  # flattened 8 x 36 supports k = 4


def test_k_clamped_to_matrix_rank_and_small_matrices_skipped():
    small = torch.nn.Parameter(torch.randn(3, 2))
    vec = torch.nn.Parameter(torch.randn(5, 1))
    opt = RoutedMuon([small, vec], **{**BASE_KW, **ROUTE_KW, "k": 16})
    small.grad = torch.randn(3, 2)
    vec.grad = torch.randn(5, 1)
    opt.step()
    assert opt.state[small]["routing"].k == 2  # min(m, n)
    assert "routing" not in opt.state[vec]  # min dim < 2: stock path


# ------------------------------------------- planted regimes -> labels/gains


PLANTED_STEPS = 119  # driven via opt.step(); step 120 done manually below
PLANTED_R = 1.05


@pytest.fixture(scope="module")
def planted_run():
    stream = three_regime_stream(r_osc=PLANTED_R)
    param = torch.nn.Parameter(
        0.1 * torch.randn(14, 12, generator=torch.Generator().manual_seed(SEED))
    )
    opt = RoutedMuon([param], **{**BASE_KW, **ROUTE_KW})
    drive(opt, param, stream, PLANTED_STEPS)

    # Final step through the public hooks so the returned update direction
    # O_t can be captured exactly (no lr division noise).
    group = opt.param_groups[0]
    state = opt.state[param]
    G = stream.grad(PLANTED_STEPS + 1)
    with torch.no_grad():
        state["step"] += 1
        shaped = opt.pre_step(G, state, group)
        M = shaped.clone()
        update = opt.shape_spectrum(shaped, state, group)
    tier = state["routing"]
    return stream, tier, M, update


def _planted_indices(stream, tier):
    idx = [match_planted(tier, stream.U[:, j], stream.V[:, j]) for j in range(3)]
    assert -1 not in idx, f"tracked pairs lost a planted direction: {idx}"
    assert len(set(idx)) == 3
    return idx  # [signal, noise, osc]


def test_planted_labels_recovered(planted_run):
    stream, tier, _, _ = planted_run
    i_sig, i_noise, i_osc = _planted_indices(stream, tier)
    regimes = tier.last_regimes
    assert regimes[i_sig] is Regime.SIGNAL
    assert regimes[i_noise] is Regime.NOISE
    assert regimes[i_osc] is Regime.OSCILLATING


def test_gains_follow_plan_rules(planted_run):
    stream, tier, _, _ = planted_run
    i_sig, i_noise, i_osc = _planted_indices(stream, tier)
    gains = tier.last_gains
    assert gains[i_sig] == 1.0  # signal: stock Muon
    assert gains[i_noise] == 0.25  # noise: g_noise floor
    # Oscillation: g = clip(1 / amplitude_ratio, 0.1, 1) from the
    # classifier's own estimate, and close to ground truth 1/r.
    amp = tier.classifier.stats.amplitude_ratio[i_osc]
    assert gains[i_osc] == pytest.approx(
        float(np.clip(1.0 / amp, 0.1, 1.0)), abs=1e-12
    )
    assert gains[i_osc] == pytest.approx(1.0 / PLANTED_R, abs=0.03)
    assert gains[i_osc] < 1.0


def test_actual_correction_matches_gains(planted_run):
    """O_t == NS(M_t) + sum_i (g_i - 1) (u_i^T O v_i) u_i v_i^T."""
    _, tier, M, update = planted_run
    O_ns = zeropower_via_newtonschulz5(
        M, steps=BASE_KW["ns_steps"], eps=1e-7, dtype=torch.float32
    )
    U, V = tier.U, tier.V
    g = torch.from_numpy(tier.last_gains).to(torch.float32)
    proj = ((U.T @ O_ns) * V.T).sum(dim=1)
    expected = O_ns + (U * ((g - 1.0) * proj).unsqueeze(0)) @ V.T
    assert torch.allclose(update, expected, atol=1e-5)
    # And the correction is genuinely applied (noise + osc gated).
    assert not torch.allclose(update, O_ns, atol=1e-4)


def test_ns_property_preserved_outside_gated_subspace(planted_run):
    """The deviation from stock NS is rank <= #gated directions."""
    _, tier, M, update = planted_run
    O_ns = zeropower_via_newtonschulz5(
        M, steps=BASE_KW["ns_steps"], eps=1e-7, dtype=torch.float32
    )
    n_gated = int(np.sum(tier.last_gains != 1.0))
    assert n_gated >= 2  # noise + oscillation both active
    sv = torch.linalg.svdvals((update - O_ns).to(torch.float64))
    assert float(sv[n_gated]) < 1e-6 * max(1.0, float(sv[0]))


# ----------------------------------------------------- confidence gating


def test_no_gain_deviation_before_n_min():
    stream = three_regime_stream()
    gen = torch.Generator().manual_seed(SEED)
    w0 = 0.1 * torch.randn(14, 12, generator=gen)
    p_routed = torch.nn.Parameter(w0.clone())
    p_muon = torch.nn.Parameter(w0.clone())
    routed = RoutedMuon([p_routed], **{**BASE_KW, **ROUTE_KW})
    muon = Muon([p_muon], **BASE_KW)

    n_min = ROUTE_KW["n_min"]
    for t in range(1, n_min):  # steps 1 .. n_min - 1
        G = stream.grad(t)
        p_routed.grad = G.clone()
        p_muon.grad = G.clone()
        routed.step()
        muon.step()
        assert torch.equal(p_routed.detach(), p_muon.detach()), (
            f"deviation before the n_min confidence gate at step {t}"
        )

    for t in range(n_min, n_min + 15):
        G = stream.grad(t)
        p_routed.grad = G.clone()
        p_muon.grad = G.clone()
        routed.step()
        muon.step()
    assert not torch.equal(p_routed.detach(), p_muon.detach()), (
        "classifier never left SIGNAL after the gate opened"
    )


# ------------------------------------------------------ refresh reset


def test_refresh_resets_rotated_directions():
    m, n, r = 10, 8, 1.05
    U, V = orthonormal_pairs(m, n, 2, SEED + 10)
    switch_at = 30  # stream rotates from pair 0 to pair 1 after this step

    def s0(t):
        return 0.4 * (-r) ** t if t <= switch_at else 0.0

    def s1(t):
        return 0.0 if t <= switch_at else 0.4 * (-r) ** (t - switch_at)

    stream = PlantedStream(m, n, U, V, [s0, s1], seed=SEED + 11)
    param = torch.nn.Parameter(
        0.1 * torch.randn(m, n, generator=torch.Generator().manual_seed(SEED))
    )
    kw = {**BASE_KW, **ROUTE_KW, "k": 2, "t_refresh": 20, "n_min": 10}
    opt = RoutedMuon([param], **kw)

    # Phase 1 (steps 1-30): oscillation through pair 0. t_refresh = 20 is
    # even, so the period-2 sign pattern is identical at refreshes 1 and 21
    # -> no spurious rotation reset for the planted pair.
    drive(opt, param, stream, switch_at)
    tier = opt.state[param]["routing"]
    i0 = match_planted(tier, U[:, 0], V[:, 0])
    assert i0 == 0  # dominant pair
    assert tier.last_regimes[i0] is Regime.OSCILLATING
    assert tier.last_gains[i0] < 1.0

    # Steps 31-40: stream moved to pair 1; no refresh yet.
    drive(opt, param, stream, 10, start=switch_at + 1)

    # Step 41 is a refresh step: the top tracked pair rotates onto pair 1
    # -> innovation (alignment < align_min) -> confidence reset -> SIGNAL,
    # all gains revert to 1 (stock Muon) immediately.
    drive(opt, param, stream, 1, start=41)
    assert float(tier.last_alignment[0]) < kw["align_min"]
    i1 = match_planted(tier, U[:, 1], V[:, 1])
    assert i1 == 0
    assert tier.last_regimes[i1] is Regime.SIGNAL
    assert np.all(tier.last_gains == 1.0)
    assert int(tier.classifier.n_since_reset[i1]) <= 2  # n_min clock restarted

    # After n_min fresh observations the rotated direction re-earns its
    # oscillation label and gain.
    drive(opt, param, stream, 19, start=42)  # through step 60
    assert tier.last_regimes[i1] is Regime.OSCILLATING
    assert tier.last_gains[i1] < 1.0


# ------------------------------------------------------- ablation modes


def _osc_only_stream(m=8, n=6, r=1.05, seed=SEED + 20):
    U, V = orthonormal_pairs(m, n, 1, seed)
    return PlantedStream(
        m, n, U, V, [lambda t: 0.4 * (-r) ** t], seed=seed + 1
    ), U, V


@pytest.mark.parametrize("rho_ignored", [False, True])
def test_rho_ignored_is_magnitude_only_gate(rho_ignored):
    stream, U, V = _osc_only_stream()
    param = torch.nn.Parameter(
        0.1 * torch.randn(8, 6, generator=torch.Generator().manual_seed(SEED))
    )
    kw = {**BASE_KW, **ROUTE_KW, "k": 2, "rho_ignored": rho_ignored}
    opt = RoutedMuon([param], **kw)
    drive(opt, param, stream, 100)
    tier = opt.state[param]["routing"]
    i = match_planted(tier, U[:, 0], V[:, 0])
    assert i == 0
    if rho_ignored:
        # Autocorrelation invisible: the alternating stream has |t| below
        # tau_noise, so the magnitude-only gate calls it NOISE.
        assert tier.last_regimes[i] is Regime.NOISE
        assert tier.last_gains[i] == kw["g_noise"]
        assert not any(r is Regime.OSCILLATING for r in tier.last_regimes)
    else:
        assert tier.last_regimes[i] is Regime.OSCILLATING
        assert tier.last_gains[i] == pytest.approx(1.0 / 1.05, abs=0.03)


def test_random_gating_permutes_the_same_gain_multiset():
    def build(random_gating, seed=SEED):
        stream = three_regime_stream()
        param = torch.nn.Parameter(
            0.1
            * torch.randn(14, 12, generator=torch.Generator().manual_seed(SEED))
        )
        kw = {**BASE_KW, **ROUTE_KW, "random_gating": random_gating, "seed": seed}
        opt = RoutedMuon([param], **kw)
        gains_hist = []
        for t in range(1, 81):
            param.grad = stream.grad(t)
            opt.step()
            tier = opt.state[param]["routing"]
            gains_hist.append(tier.last_gains.copy())
        return param, gains_hist

    p_true, hist_true = build(False)
    p_rand, hist_rand = build(True)
    p_rand2, hist_rand2 = build(True)

    # Placebo contract: at every step, the same fraction of directions is
    # gated with the same gain values -- only the assignment is random.
    # (The classifier state is gradient-driven, identical across runs.)
    for g_t, g_r in zip(hist_true, hist_rand):
        assert np.allclose(np.sort(g_t), np.sort(g_r))
    # The assignment actually differs somewhere once gating is active.
    assert any(
        not np.array_equal(g_t, g_r) for g_t, g_r in zip(hist_true, hist_rand)
    )
    assert not torch.equal(p_true.detach(), p_rand.detach())
    # Dedicated RNG seeded from the optimizer seed: fully deterministic.
    assert all(
        np.array_equal(a, b) for a, b in zip(hist_rand, hist_rand2)
    )
    assert torch.equal(p_rand.detach(), p_rand2.detach())


# ----------------------------- constant oscillation gain (Gate-1 A2)


class _StubStats:
    def __init__(self, amp):
        self.amplitude_ratio = np.asarray(amp, dtype=np.float64)

    def is_decaying(self, margin):
        return self.amplitude_ratio < 1.0 - float(margin)


class _StubTier:
    """Minimal tier facade for _compute_gains: labels + amplitude ratios."""

    def __init__(self, regimes, amp, seed=SEED):
        self.k = len(regimes)
        self.gating_rng = np.random.default_rng(seed)
        self.classifier = type(
            "Clf", (), {"regimes": list(regimes), "stats": _StubStats(amp)}
        )()


def _gains(opt_kwargs, regimes, amp):
    opt = RoutedMuon(
        [torch.nn.Parameter(torch.zeros(6, 5))], **{**BASE_KW, **ROUTE_KW, **opt_kwargs}
    )
    return opt._compute_gains(_StubTier(regimes, amp))


def test_g_osc_const_validation_and_default():
    p = [torch.nn.Parameter(torch.zeros(4, 3))]
    assert RoutedMuon(p).g_osc_const is None  # adaptive is the default
    assert RoutedMuon(p, g_osc_const=0.5).g_osc_const == 0.5
    for bad in (0.0, -0.25, 1.5):
        with pytest.raises(ValueError, match="g_osc_const"):
            RoutedMuon(p, g_osc_const=bad)


def test_constant_mode_applies_exactly_g_osc_const():
    regimes = [Regime.OSCILLATING, Regime.NOISE, Regime.SIGNAL]
    amp = [1.6, 1.0, 1.0]  # osc non-decaying; adaptive would give 1/1.6
    for g_const in (0.25, 0.5, 0.75):
        gains = _gains({"g_osc_const": g_const}, regimes, amp)
        assert gains[0] == g_const  # exact, no clip/formula involvement
        assert gains[1] == ROUTE_KW["g_noise"]
        assert gains[2] == 1.0
    # Adaptive mode on the identical state: the clip formula, not a constant.
    adaptive = _gains({}, regimes, amp)
    assert adaptive[0] == pytest.approx(1.0 / 1.6)


def test_decay_escape_identical_in_both_modes():
    """amp_decay_margin behavior is unchanged by g_osc_const: a decaying
    oscillation (amplitude_ratio < 1 - margin) is left alone (g = 1)."""
    regimes = [Regime.OSCILLATING, Regime.OSCILLATING]
    amp = [0.9, 1.2]  # first decaying (margin 0.05), second not
    for kwargs in ({}, {"g_osc_const": 0.5}):
        gains = _gains(kwargs, regimes, amp)
        assert gains[0] == 1.0, f"decaying direction gated in mode {kwargs}"
        assert gains[1] < 1.0
    assert _gains({"g_osc_const": 0.5}, regimes, amp)[1] == 0.5
    assert _gains({}, regimes, amp)[1] == pytest.approx(1.0 / 1.2)


def test_constant_mode_end_to_end_on_planted_oscillation():
    """Driven through real steps: the planted oscillating direction gets
    exactly g_osc_const while noise/signal gains are untouched."""
    stream = three_regime_stream(r_osc=PLANTED_R)
    param = torch.nn.Parameter(
        0.1 * torch.randn(14, 12, generator=torch.Generator().manual_seed(SEED))
    )
    opt = RoutedMuon([param], **{**BASE_KW, **ROUTE_KW, "g_osc_const": 0.5})
    drive(opt, param, stream, PLANTED_STEPS + 1)
    tier = opt.state[param]["routing"]
    i_sig, i_noise, i_osc = _planted_indices(stream, tier)
    assert tier.last_regimes[i_osc] is Regime.OSCILLATING
    assert tier.last_gains[i_osc] == 0.5  # exactly the constant
    assert tier.last_gains[i_noise] == ROUTE_KW["g_noise"]
    assert tier.last_gains[i_sig] == 1.0


# --------------------------------- state-side damping (compounding hypothesis)


def test_state_damping_compounds_geometrically_in_the_buffer():
    """Brainstorm program #1 (compounding). A persistently-oscillating
    direction is classified, then the forcing goes silent. In state-damping
    mode the buffer component along that direction relaxes at rate
    g * momentum (undamped control: momentum), so the ratio damped/control
    multiplies by exactly g each treated step -- geometric compounding of a
    single per-step edit. An OUTPUT-mode twin leaves the buffer bit-for-bit
    identical to stock Muon (the edit is re-derived away and cannot compound):
    that asymmetry is the whole hypothesis.
    """
    m, n, r = 8, 6, 1.05
    U, V = orthonormal_pairs(m, n, 1, SEED + 20)
    u0, v0 = U[:, 0], V[:, 0]
    stop = 40  # oscillate to classify through here, then go silent

    def s(t):
        return 0.4 * (-r) ** t if t <= stop else 0.0

    # bg ~ 0 so the free relaxation after `stop` is not swamped by noise.
    stream = PlantedStream(m, n, U, V, [s], bg=1e-9, seed=SEED + 21)
    kw = {
        **BASE_KW,
        **ROUTE_KW,
        "k": 2,
        "t_refresh": 1000,  # lock subspace at step 1
        "n_min": 20,
        "g_osc_const": 0.5,  # fixed gain -> clean geometric factor
        "enable_noise_channel": False,  # osc-only, as in the probe configs
    }
    gen = torch.Generator().manual_seed(SEED)
    w0 = 0.1 * torch.randn(m, n, generator=gen)
    p_state = torch.nn.Parameter(w0.clone())
    p_out = torch.nn.Parameter(w0.clone())
    p_ctrl = torch.nn.Parameter(w0.clone())
    state_opt = RoutedMuon([p_state], **{**kw, "state_damping": True})
    out_opt = RoutedMuon([p_out], **{**kw, "state_damping": False})
    ctrl = Muon([p_ctrl], **BASE_KW)

    def comp(opt, p):
        buf = opt.state[p]["momentum_buffer"]
        return float(u0 @ buf @ v0)

    activated = False
    # (step, ratio, gain) records inside the SILENT relaxation window, where
    # the buffer is driven only by momentum decay + the per-step edit.
    silent = []
    for t in range(1, stop + 10):
        G = stream.grad(t)
        for p, o in ((p_state, state_opt), (p_out, out_opt), (p_ctrl, ctrl)):
            p.grad = G.clone()
            o.step()
        # OUTPUT mode never touches the buffer: identical to stock every step.
        assert torch.equal(
            out_opt.state[p_out]["momentum_buffer"],
            ctrl.state[p_ctrl]["momentum_buffer"],
        ), f"output-mode buffer diverged from stock at step {t}"
        g = float(state_opt.state[p_state]["routing"].last_gains[0])
        if g < 1.0:
            activated = True
        cc = comp(ctrl, p_ctrl)
        cd = comp(state_opt, p_state)
        if t >= stop and abs(cc) > 1e-12:
            silent.append((t, abs(cd) / abs(cc), g))

    assert activated, "oscillating direction was never treated"
    # STATE mode DID move the buffer (unlike output mode).
    assert not torch.equal(
        state_opt.state[p_state]["momentum_buffer"],
        ctrl.state[p_ctrl]["momentum_buffer"],
    )
    # In the silent window the forcing is gone: control relaxes at `momentum`,
    # the state-damped buffer at g * momentum, so the ratio multiplies by
    # exactly g each consecutive step (holding flat on any one-step classifier
    # flicker back to g = 1) -- a geometric law, the signature of compounding.
    for (tp, prev, _), (tc, cur, g) in zip(silent, silent[1:]):
        assert tc == tp + 1
        assert cur == pytest.approx(prev * g, abs=2e-3), (tp, prev, cur, g)
    # Net effect: the damped component is suppressed by orders of magnitude
    # relative to the undamped control across the window.
    assert silent[-1][1] < 0.05 * silent[0][1]


def test_state_damping_channels_off_is_bit_for_bit_stock_muon():
    """state_damping only ever edits the buffer for TREATED directions; with
    both channels disabled nothing is treated, so params AND the momentum
    buffer stay bit-for-bit identical to stock Muon (default-false regression
    guarantee: turning the flag on changes nothing until a direction is
    actually gated)."""
    gen = torch.Generator().manual_seed(1302)
    w0 = 0.1 * torch.randn(10, 8, generator=gen)
    p_routed = torch.nn.Parameter(w0.clone())
    p_muon = torch.nn.Parameter(w0.clone())
    kw = {
        **BASE_KW,
        **ROUTE_KW,
        "k": 4,
        "t_refresh": 25,
        "n_min": 5,
        "state_damping": True,
        "enable_noise_channel": False,
        "enable_oscillation_channel": False,
    }
    routed = RoutedMuon([p_routed], **kw)
    muon = Muon([p_muon], **BASE_KW)
    for _ in range(70):
        G = torch.randn(10, 8, generator=gen)
        p_routed.grad = G.clone()
        p_muon.grad = G.clone()
        routed.step()
        muon.step()
        assert torch.equal(p_routed.detach(), p_muon.detach())
        assert torch.equal(
            routed.state[p_routed]["momentum_buffer"],
            muon.state[p_muon]["momentum_buffer"],
        )
    tier = routed.state[p_routed]["routing"]
    assert tier.last_gains is not None and np.all(tier.last_gains == 1.0)


def test_state_damping_records_applied_gains_in_telemetry():
    """Telemetry must record the applied state gains identically to output
    mode: same classifier, same gain vector, only the application point
    differs."""
    stream = three_regime_stream(r_osc=PLANTED_R)
    param = torch.nn.Parameter(
        0.1 * torch.randn(14, 12, generator=torch.Generator().manual_seed(SEED))
    )
    opt = RoutedMuon(
        [param], **{**BASE_KW, **ROUTE_KW, "g_osc_const": 0.5, "state_damping": True}
    )
    drive(opt, param, stream, PLANTED_STEPS + 1)
    tier = opt.state[param]["routing"]
    i_sig, i_noise, i_osc = _planted_indices(stream, tier)
    # The oscillating direction is treated with exactly the constant gain...
    assert tier.last_regimes[i_osc] is Regime.OSCILLATING
    assert tier.last_gains[i_osc] == 0.5
    # ... and the telemetry snapshot reflects the applied gain vector.
    snap = opt.routing_stats()["per_matrix"]["matrix0_14x12"]["last"]
    g = tier.last_gains
    assert snap["n_treated"] == int(np.sum(g != 1.0))
    assert snap["gain_min"] == pytest.approx(g.min())
    assert snap["gain_sum"] == pytest.approx(g.sum())


# ------------------------------------- routing telemetry (Gate-1 A5)


def test_routing_stats_empty_before_any_step():
    opt = RoutedMuon([torch.nn.Parameter(torch.zeros(6, 5))], **BASE_KW, **ROUTE_KW)
    stats = opt.routing_stats()
    assert stats["per_matrix"] == {}
    assert stats["aggregate"] == {"last": None, "cumulative": None}


def test_routing_stats_counts_and_json_roundtrip():
    import json

    stream = three_regime_stream()
    param = torch.nn.Parameter(
        0.1 * torch.randn(14, 12, generator=torch.Generator().manual_seed(SEED))
    )
    other = torch.nn.Parameter(
        0.1 * torch.randn(6, 5, generator=torch.Generator().manual_seed(SEED + 1))
    )
    opt = RoutedMuon([param, other], **{**BASE_KW, **ROUTE_KW, "k": 3})
    n_steps = 80
    for t in range(1, n_steps + 1):
        param.grad = stream.grad(t)
        other.grad = torch.randn(6, 5, generator=torch.Generator().manual_seed(2000 + t))
        opt.step()

    stats = opt.routing_stats()
    json.loads(json.dumps(stats))  # fully JSON-able
    assert set(stats["per_matrix"]) == {"matrix0_14x12", "matrix1_6x5"}

    tier = opt.state[param]["routing"]
    m = stats["per_matrix"]["matrix0_14x12"]
    last = m["last"]
    # Last-step channel counts match the classifier's current labels...
    assert last["n_signal"] == sum(r is Regime.SIGNAL for r in tier.last_regimes)
    assert last["n_noise"] == sum(r is Regime.NOISE for r in tier.last_regimes)
    assert last["n_oscillating"] == sum(
        r is Regime.OSCILLATING for r in tier.last_regimes
    )
    assert last["n_signal"] + last["n_noise"] + last["n_oscillating"] == tier.k
    # ... and the treated/gain figures match the applied gains exactly.
    g = tier.last_gains
    assert last["n_treated"] == int(np.sum(g != 1.0))
    assert last["treated_fraction"] == pytest.approx(np.mean(g != 1.0))
    assert last["gain_sum"] == pytest.approx(g.sum())
    assert last["gain_min"] == pytest.approx(g.min())
    assert last["gain_max"] == pytest.approx(g.max())
    assert last["step"] == n_steps
    # In-confidence-window count from the classifier's n_min clock.
    n_min = ROUTE_KW["n_min"]
    assert last["n_in_confidence_window"] == int(
        np.sum(tier.classifier.n_since_reset < n_min)
    )
    # Cumulative bookkeeping.
    cum = m["cumulative"]
    assert cum["n_steps"] == n_steps
    assert cum["direction_steps"] == n_steps * tier.k
    assert (
        cum["signal_direction_steps"]
        + cum["noise_direction_steps"]
        + cum["oscillating_direction_steps"]
        == cum["direction_steps"]
    )
    assert cum["treated_direction_steps"] <= cum["direction_steps"]
    assert 0.0 < cum["gain_min"] <= cum["gain_max"] == 1.0
    # Confidence window: the first n_min - 1 steps of every direction at
    # minimum (no resets in this stream for matrix 0).
    assert cum["in_confidence_window_direction_steps"] >= (n_min - 1) * tier.k
    # Aggregate = sum over matrices.
    agg = stats["aggregate"]
    assert agg["last"]["k_total"] == tier.k + opt.state[other]["routing"].k
    assert agg["last"]["n_treated"] == sum(
        stats["per_matrix"][k]["last"]["n_treated"] for k in stats["per_matrix"]
    )
    assert agg["cumulative"]["direction_steps"] == sum(
        stats["per_matrix"][k]["cumulative"]["direction_steps"]
        for k in stats["per_matrix"]
    )
    assert agg["cumulative"]["n_refreshes"] == sum(
        stats["per_matrix"][k]["n_refreshes"] for k in stats["per_matrix"]
    )


def test_routing_stats_counts_confidence_resets():
    """The rotation-reset counter mirrors the refresh-innovation reset of
    test_refresh_resets_rotated_directions."""
    m, n, r = 10, 8, 1.05
    U, V = orthonormal_pairs(m, n, 2, SEED + 10)
    switch_at = 30

    def s0(t):
        return 0.4 * (-r) ** t if t <= switch_at else 0.0

    def s1(t):
        return 0.0 if t <= switch_at else 0.4 * (-r) ** (t - switch_at)

    stream = PlantedStream(m, n, U, V, [s0, s1], seed=SEED + 11)
    param = torch.nn.Parameter(
        0.1 * torch.randn(m, n, generator=torch.Generator().manual_seed(SEED))
    )
    kw = {**BASE_KW, **ROUTE_KW, "k": 2, "t_refresh": 20, "n_min": 10}
    opt = RoutedMuon([param], **kw)
    drive(opt, param, stream, 41)  # through the rotating refresh at step 41
    snap = opt.routing_stats()["per_matrix"]["matrix0_10x8"]
    assert snap["n_rotation_resets"] >= 1
    assert snap["n_confidence_resets"] >= snap["n_rotation_resets"]
    assert snap["last"]["n_in_confidence_window"] >= 1  # freshly reset


def test_smoke_experiment_records_routing_stats_on_cpu():
    """The scripts/run.py 'smoke' experiment (CPU MLP) is the CPU-verifiable
    harness path for the A5 telemetry wiring (airbench itself needs CUDA)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_module_routed_smoke", REPO_ROOT / "scripts" / "run.py"
    )
    run_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_mod)

    config = {
        "experiment": "smoke",
        "seed": 1000,
        "model": {"in_dim": 16, "hidden_dim": 24, "out_dim": 4, "bias": False},
        "train": {"steps": 40},
        "optimizer": {
            "name": "routed",
            "lr": 1e-3,
            "k": 4,
            "n_min": 10,
            "t_refresh": 20,
            "beta": 0.9,
            "seed": 1500,
        },
    }
    metrics = run_mod.run_smoke(config, torch.device("cpu"))
    stats = metrics["routing_stats"]
    assert len(stats["per_matrix"]) == 2  # both weight matrices routed
    agg = stats["aggregate"]
    assert agg["last"]["step"] == 40
    assert agg["cumulative"]["direction_steps"] == 40 * agg["last"]["k_total"]
    for key in (
        "n_signal",
        "n_noise",
        "n_oscillating",
        "n_treated",
        "treated_fraction",
        "n_in_confidence_window",
        "gain_sum",
        "gain_min",
        "gain_max",
    ):
        assert key in agg["last"], key
    import json

    json.loads(json.dumps(metrics))  # results-JSON safe


# ------------------------------------------- stock-Muon equivalence


def test_channels_disabled_is_bit_for_bit_stock_muon():
    gen = torch.Generator().manual_seed(1300)
    w0 = 0.1 * torch.randn(10, 8, generator=gen)
    p_routed = torch.nn.Parameter(w0.clone())
    p_muon = torch.nn.Parameter(w0.clone())
    kw = {
        **BASE_KW,
        **ROUTE_KW,
        "k": 4,
        "t_refresh": 25,  # several refreshes inside the trajectory
        "n_min": 5,  # classifier active -- gains must still all be 1
        "enable_noise_channel": False,
        "enable_oscillation_channel": False,
    }
    routed = RoutedMuon([p_routed], **kw)
    muon = Muon([p_muon], **BASE_KW)
    for _ in range(70):
        G = torch.randn(10, 8, generator=gen)
        p_routed.grad = G.clone()
        p_muon.grad = G.clone()
        routed.step()
        muon.step()
        assert torch.equal(p_routed.detach(), p_muon.detach())
    tier = routed.state[p_routed]["routing"]
    assert tier.last_gains is not None and np.all(tier.last_gains == 1.0)


def test_weight_decay_and_adjust_lr_match_muon_when_disabled():
    gen = torch.Generator().manual_seed(1301)
    w0 = torch.randn(6, 9, generator=gen)
    p_r = torch.nn.Parameter(w0.clone())
    p_m = torch.nn.Parameter(w0.clone())
    kw = dict(BASE_KW, weight_decay=0.01, adjust_lr="spectral_norm")
    routed = RoutedMuon(
        [p_r],
        **kw,
        **{**ROUTE_KW, "enable_noise_channel": False,
           "enable_oscillation_channel": False},
    )
    muon = Muon([p_m], **kw)
    for _ in range(10):
        G = torch.randn(6, 9, generator=gen)
        p_r.grad = G.clone()
        p_m.grad = G.clone()
        routed.step()
        muon.step()
        assert torch.equal(p_r.detach(), p_m.detach())


# ------------------------------------------------------------ smoke config


CONFIG_PATH = REPO_ROOT / "configs" / "dev" / "airbench_smoke_routed.yaml"


def test_smoke_config_parses_and_is_constructible():
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    assert config["experiment"] == "airbench_smoke"
    assert isinstance(config["seed"], int) and config["seed"] >= 1000
    opt_cfg = dict(config["optimizer"])
    assert opt_cfg.pop("name") == "routed"
    # Plan defaults present in the config.
    assert opt_cfg["k"] == 16
    assert opt_cfg["t_refresh"] == 50
    assert opt_cfg["beta"] == 0.99
    assert opt_cfg["g_noise"] == 0.25
    assert opt_cfg["n_min"] == 50
    # Constructible on an airbench-shaped conv filter, as the harness does.
    filter_like = [torch.nn.Parameter(torch.randn(8, 4, 3, 3))]
    opt = build_optimizer("routed", filter_like, opt_cfg)
    assert isinstance(opt, RoutedMuon)
