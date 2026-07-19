<!-- Lit-review agent report, 2026-07-19. See README.md for verdict summary. -->

# Literature Map: Stability Theory of Normalized / Orthogonalized / Sign-Based Updates

## (1) Map of the most relevant works

**A. The direct hit — Islamov, Crawshaw, Cohen & Gower, "Non-Euclidean Gradient Descent Operates at the Edge of Stability" (arXiv:2603.05002, ICML 2026 oral)**
Extends edge-of-stability (EoS) theory from Euclidean GD to steepest descent under arbitrary norms — explicitly including signGD (ℓ∞), spectral GD (polar-factor / Muon-style updates), and their *normalized* variants. Defines **directional smoothness** D(w_t, w_{t+1}) = [L(y) − L(w) − ⟨∇L, y−w⟩]/(½‖y−w‖²) in the chosen norm and a **generalized sharpness** per norm, and proves loss decreases iff directional smoothness ≤ 2/η *in that geometry*. Experiments (MLP/CNN/Transformer, incl. full CIFAR-10) show the *generalized* sharpness equilibrates at 2/η while the **Euclidean ℓ₂ sharpness is decoupled from stability entirely** — EoS "only occurs with respect to generalized sharpness." For normalized variants the threshold becomes 2‖∇L‖_*/η (gradient-dual-norm-rescaled). They also report a **pre-EoS oscillatory regime unique to ℓ∞/spectral geometry**: iterates oscillate in parameter space (broadband) while still stable on the quadratic approximation, *before* generalized sharpness reaches threshold. Implication for you: Muon's max stable LR is set by spectral-geometry directional smoothness hitting 2/η, not by η·λ_max(H) < 2. Limitations they state: full-batch, momentum-free, deterministic; divergence proven only from special initializations.

**B. Cohen et al., "Adaptive Gradient Methods at the Edge of Stability" (arXiv:2207.14484)**
First systematic evidence that the raw η·λ < 2 bound does not govern preconditioned updates: for full-batch Adam the *preconditioned* sharpness equilibrates at ≈ 38/η (β₁ = 0.9), and adaptive methods "can and do enter high-curvature regions" (raw sharpness ≫ 2/η) by adapting the preconditioner. Implies: expect a momentum-corrected constant (not 2) for momentum-Muon too.

**C. Cohen, Damian et al., "Understanding Optimization in Deep Learning with Central Flows" (arXiv:2410.24206, ICLR 2025)**
The machinery for the oscillatory regime: derives an ODE for the *time-averaged* trajectory, treating EoS oscillations via their covariance Σ. Shows adaptive optimizers implicitly adapt effective step size to the maximum stable value, and that oscillations feed back to reduce sharpness ("acceleration via regularization"). Covers GD, RMSProp and Scalar-RMSProp-type (normalized-GD-like) methods — **not Muon, sign-momentum, or heavy-ball momentum** (left open). This is the template a "central flow for Muon" would follow.

**D. Arora, Li & Panigrahi, "Understanding Gradient Descent on the Edge of Stability" (arXiv:2205.09745)**
Foundational: proves **normalized GD** provably operates at EoS and drifts along the minimum manifold reducing sharpness — the first rigorous statement that a magnitude-decoupled update self-organizes into a bounded oscillation + sharpness-reduction flow rather than diverging.

**E. Nguyen, Truong et al., "Spectral Flattening Is All Muon Needs" (arXiv:2605.13079)**
Derives explicit descent thresholds: SGD's η_max = 2/λ_max, but **Muon's η_max = (2/λ_max)·(Σσᵢ/m)** — governed by the *average* singular value of the update spectrum under a K-FAC curvature model, an explicit published replacement for η·λ < 2. Empirically Muon stays stable at LRs that diverge SGD immediately.

**F. Beneventano, Abdelmoneum & Poggio, "The Spectral Dynamics and Noise Geometry of Muon" (arXiv:2606.08388)**
Exact singular-value dynamics for continuous-time Muon; empirically documents Muon's **broad learning-rate plateau** in NanoGPT pretraining — the published cousin of your "graceful degradation at 2× record LR" observation, but without an HVP-level stability mechanism.

**G. "Why Muon Outperforms Adam: A Curvature Perspective" (arXiv:2606.04662)**
Measures Hessian interaction directly: Muon's advantage comes from lower **normalized directional sharpness (NDS)** of its update direction, not smaller update norm — spectral flattening spreads update energy across curvature modes. Closest published *measurement* program to your HVP-along-tracked-directions instrumentation (but per-update-direction, not per-tracked-eigendirection over time, and no η·λ threshold claim).

**H. Bernstein & Newhouse ("Old Optimizer, New Norm" arXiv:2409.20325; "Modular Duality" arXiv:2410.21265) + Yang, Simon & Bernstein, "A Spectral Condition for Feature Learning" (arXiv:2310.17813)**
The geometric a priori theory: Muon = momentum steepest descent under the spectral norm; sign descent = steepest descent under ℓ∞. Max stable LR should be set by **smoothness measured in the dual/spectral geometry**, which is roughly width-independent — explaining LR transfer and predicting O(1) spectral-norm step sizes regardless of Euclidean sharpness. Predicts *what quantity* the threshold should involve, but proves no EoS-style threshold.

**I. Balles, Pedregosa & Le Roux, "The Geometry of Sign Gradient Descent" (arXiv:2002.08056)**
Foundational for sign updates: signGD max LR governed by the **ℓ∞ smoothness constant** (≈ Hessian's density/diagonal concentration), not λ_max — the earliest clear statement that curvature-in-the-right-norm sets sign-descent stability.

**J. Chen et al. / Liu et al., "Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts" (arXiv:2310.05898)**
Lyapunov analysis showing Lion's sign-momentum dynamics enforce a bound constraint ‖w‖∞ ≤ 1/λ: sign-based updates yield **structurally bounded iterates** — divergence in the GD sense is impossible; the failure mode is a bounded oscillation band of width ~η, matching your "self-limiting amplitude" hypothesis at the level of mechanism (though for weights, not per-direction loss oscillation).

**K. Kosson et al., "Rotational Equilibrium" (arXiv:2305.17212)**
For magnitude-decoupled updates + weight decay, angular update per step converges to a **self-limiting steady state** independent of gradient magnitude — another published "the step size, not the curvature, bounds the dynamics" mechanism (equilibrium/effective-LR view rather than a stability threshold).

**L. Shen et al., "Power and Limits of the Muon Optimizer: A River-Valley Perspective" (arXiv:2606.21514)**
Muon's flat spectrum discards residual scale information, making it "prone to overshooting and oscillation near the target solution" — i.e., persistent bounded oscillation is intrinsic to orthogonalized updates near minima, framed as a *convergence* liability rather than a stability asset.

(Also relevant but secondary: "On the Convergence Analysis of Muon" arXiv:2505.23737 and follow-ups — step-size bounds via non-Euclidean smoothness assumptions; "Muon Optimizes Under Spectral Norm Constraints" arXiv:2506.15054.)

## (2) Verdict on your observation

**Partially reported — the qualitative claim is now published; your quantitative regime is not.** Confidence: high.

- The core claim — *Euclidean η·λ < 2 demonstrably does not govern sign/normalized/spectral steepest descent; what governs is directional smoothness / generalized sharpness in the update's own norm hitting 2/η* — is exactly the thesis of **Islamov, Crawshaw, Cohen & Gower (arXiv:2603.05002, ICML 2026 oral)**, with CIFAR-10 experiments showing ℓ₂ sharpness decoupled from stability under spectral GD. Your "η·λ up to ~65" is a quantitative instance of their qualitative decoupling. Their **pre-EoS broadband parameter-space oscillation** for ℓ∞/spectral norms is the closest published match to your broadband negative lag-1 autocorrelation (period-2 bouncing shows up as negative lag-1 autocorrelation in per-direction gradient projections).
- Not published anywhere I found: (a) the **momentum + minibatch practical-Muon** version — 2603.05002 is full-batch and momentum-free, and explicitly lists momentum/stochastic/adaptive extension as open; (b) **HVP-measured per-eigendirection η·λ time series under Muon** with a concrete number like 65 (the Adam analog, 38/η, exists — 2207.14484 — but no Muon analog); (c) the **momentum-independence of the oscillation** finding; (d) a derived *amplitude* law for the self-limited oscillation (Lyapunov/Lion and the river-valley paper give boundedness mechanisms, central flows gives the averaging machinery, but nobody has combined them into "the oscillation amplitude under a fixed-magnitude update is ~η·(update scale), hence no exponential blow-up, hence graceful degradation").

## (3) Sharpest open question your instrumentation could address

**What is the momentum-corrected stability threshold for practical (stochastic, momentum) Muon, and what law sets the bounded oscillation amplitude?** Concretely: does directional smoothness / generalized *spectral* sharpness along the actual trajectory equilibrate at a constant c/η (the Muon analog of GD's 2/η and Adam's 38/η), with c depending on momentum β — and is the per-direction oscillation amplitude pinned at ~η per step (fixed-magnitude update) rather than growing with Euclidean λ? Your tracked-direction HVP + per-direction gradient-projection autocorrelation machinery is precisely the instrument: measure both Euclidean η·λ (you have: ~65, unbounded-looking) *and* trajectory directional smoothness in the spectral norm simultaneously, across β and LR, in the minibatch regime. Nobody has published that measurement; 2603.05002's authors name it as their main open problem, and central flows lacks the momentum/Muon extension it would calibrate.

## (4) Queries run

- "edge of stability sign gradient descent normalized gradient descent stability learning rate"
- "Muon optimizer convergence analysis theory maximum stable learning rate 2025"
- "Lion optimizer theory Lyapunov convergence sign momentum edge of stability"
- "'central flow' Cohen optimizer dynamics edge of stability 2024 2025 adaptive optimizers"
- "Muon edge of stability sharpness dynamics orthogonalized updates 2025 2026"
- "signSGD stability analysis oscillation bounded quadratic maximum learning rate"
- "'adaptive gradient methods at the edge of stability' Cohen preconditioned sharpness Adam"
- "Bernstein Newhouse steepest descent spectral norm Muon modular norms learning rate transfer"
- "arxiv 2606.08388 spectral dynamics noise geometry Muon sharpness"
- "'edge of stability' Muon spectral norm steepest descent stability threshold non-Euclidean"
- "Yang Simon Bernstein 'spectral condition' feature learning maximum learning rate width transfer"
- "central flow Muon orthogonalized optimizer 2026 time-averaged trajectory"
- "'edge of stability' 'steepest descent' arbitrary norm generalization stability threshold 2025 2026 sharpness dual norm"
- "Kosson rotational equilibrium weight decay effective learning rate normalized updates"
- "'Why Muon Outperforms Adam' curvature perspective Hessian sharpness measurement"
- "'sign gradient descent' quadratic dynamics bounded oscillation never diverges amplitude step size"
- "'Muon' 'edge of stability' experiment sharpness 2/eta LLM training 2026"
- Full-text fetches: arXiv 2603.05002 (abs + html), 2605.13079, 2606.21514, 2410.24206, 2606.08388 (via search snippets)

Sources: [Non-Euclidean Gradient Descent Operates at the Edge of Stability](https://arxiv.org/abs/2603.05002), [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/abs/2207.14484), [Understanding Optimization in Deep Learning with Central Flows](https://arxiv.org/abs/2410.24206), [Understanding Gradient Descent on Edge of Stability in Deep Learning](https://arxiv.org/abs/2205.09745), [Spectral Flattening Is All Muon Needs](https://arxiv.org/html/2605.13079), [The Spectral Dynamics and Noise Geometry of Muon](https://arxiv.org/abs/2606.08388), [Why Muon Outperforms Adam: A Curvature Perspective](https://arxiv.org/abs/2606.04662), [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813), [The Geometry of Sign Gradient Descent](https://arxiv.org/pdf/2002.08056), [Lion Secretly Solves Constrained Optimization](https://arxiv.org/abs/2310.05898), [Rotational Equilibrium](https://arxiv.org/abs/2305.17212), [Power and Limits of Muon: A River-Valley Perspective](https://arxiv.org/pdf/2606.21514), [On the Convergence Analysis of Muon](https://arxiv.org/pdf/2505.23737), [Muon Optimizes Under Spectral Norm Constraints](https://arxiv.org/html/2506.15054v2)
