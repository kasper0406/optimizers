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
