<!-- Lit-review agent report, 2026-07-20. Direction: frozen-probe long-integration signal detection. Verdict: unreported with partial anticipation. -->

All searches complete. Here is the structured literature report.

---

## (1) Closest works (citation, contribution, diff from our frozen-probe detector)

**1. Yao, Wang, Lin, Liu, Zhong (2026), "Gradient-Flow Optimization as Dynamic Random-Effects Inference: Testing and Early Stopping" — [arXiv:2605.27991](https://arxiv.org/html/2605.27991).** The closest hit found. Casts training as random-effects inference: a variance-component **score test** for whether training extracts signal beyond initialization, computed along **fixed directions** (eigenvectors of a fixed Gram/NTK operator), integrated **cumulatively over the whole trajectory** (their Empirical Spectral Covariance Ψₙ(t)), used for REML-based early stopping. **Diff:** directions are operator eigenvectors, not random probes; the test lives in function/residual space in a kernel-regime model; the deliverable is one global stopping time, not per-direction signal/noise classification; no t ≤ SNR·√ESS instrument-limit framing, no comparison against refreshed-subspace tracking.

**2. Litman & Guo (2026), "A Theory of Generalization in Deep Learning" — [arXiv:2605.01172](https://arxiv.org/abs/2605.01172).** Theory: decomposes SGD per direction (partitioned by the empirical NTK) into drift + martingale diffusion; on signal directions drift accumulates **linearly in T** while zero-mean noise sums like **√T**, so signal directions dominate as √T/T → 0. This is precisely the asymptotic separation our slow tier exploits. **Diff:** theoretical; signal directions are data/kernel-defined, not frozen random probes; no measurement instrument, no accumulated t-statistics during real runs.

**3. Lang, Zhang & Xiao (2019) SASA / Zhang et al. (2020) SASA+ — [arXiv:2002.10597](https://arxiv.org/abs/2002.10597), plus Pessia & Chee/Toulis-style diagnostics ([arXiv:1710.06382](https://arxiv.org/pdf/1710.06382)), SplitSGD ([arXiv:1910.08597](https://arxiv.org/pdf/1910.08597)), coupling-based diagnostics ([arXiv:2412.11341](https://arxiv.org/pdf/2412.11341)).** Online statistical tests (with growing/expanding windows in SASA) on functionals of SGD iterates to detect stationarity and drop the LR. **Diff:** scalar full-vector statistics (e.g., ⟨g, x−x̄⟩), detecting *stationarity of the whole chain*, not per-direction persistent-signal detection; no fixed probe directions; the lineage has no 2024–2026 per-direction successor that I could find.

**4. Liu et al. (2020) GSNR lineage; VRGD "Accelerating Large Batch Training via GSNR" — [arXiv:2309.13681](https://arxiv.org/abs/2309.13681); GSNR-guided domain generalization [arXiv:2310.07361](https://arxiv.org/abs/2310.07361).** Per-**parameter** gradient mean²/variance over the data distribution; used for generalization bounds and optimizer scaling. **Diff:** coordinate basis, not projections; estimated per step or short EMA — **no growing/unbounded window**, no test-statistic calibration, no notion of instrument-limited integration.

**5. Gur-Ari, Roberts & Dyer (2018) "tiny subspace" [arXiv:1812.04754](https://arxiv.org/abs/1812.04754) + Song, Ahn & Yun (ICLR 2025) "Does SGD really happen in tiny subspaces?" — [arXiv:2405.16002](https://arxiv.org/abs/2405.16002).** The follow-up shows gradient alignment to the top-Hessian subspace is **spurious**: projecting the update onto the dominant subspace stops loss decrease, while projecting it out (Bulk-SGD) works fine — dominant-subspace components are largely oscillatory back-and-forth. **Diff:** evolving (top-k) directions and per-step alignment measures, no long-window accumulation; but this is the key published evidence that per-window statistics on tracked/top directions can be dominated by oscillation rather than persistent signal.

**6. Antil & Verma (2025), "Randomized Matrix Sketching for Neural Network Training and Gradient Monitoring" — [arXiv:2510.00442](https://arxiv.org/html/2510.00442v1).** I verified via fetch: random sketch matrices are **initialized once and kept fixed** through training; sketched gradients are EMA-smoothed and used for monitoring (vanishing/exploding gradients, stable rank, convergence patterns). **Diff:** explicitly **no hypothesis tests, t-statistics, or SNR detection**; EMA window (β ≤ 0.99), not unbounded integration. This is the nearest "fixed random probes as measurement device" paper, and it stops short of signal detection.

**7. Frozen-direction curvature tracking: Alain, Roux & Manzagol (2019) "Negative eigenvalues of the Hessian" — [arXiv:1902.02366](https://arxiv.org/pdf/1902.02366); newer: "Characterizing Optimizer-Dependent Training Dynamics Through Hessian Eigenvector Displacement" [arXiv:2606.30226](https://arxiv.org/pdf/2606.30226).** Track curvature along a frozen eigenvector v₁(t₀) over training. **Diff:** frozen direction, but the tracked quantity is curvature, not gradient projections; no statistical testing.

**8. Li et al. (2018) intrinsic dimension / Li et al. "Low Dimensional Trajectory Hypothesis" ([ResearchGate](https://www.researchgate.net/publication/360883049_Low_Dimensional_Trajectory_Hypothesis_is_True_DNNs_can_be_Trained_in_Tiny_Subspaces)), GaLore/SubTrack++ ([OpenReview](https://openreview.net/pdf?id=6geRIdlFWJ)), "Randomized Gradient Subspaces for Efficient LLM Training" — [arXiv:2510.01878](https://arxiv.org/html/2510.01878v1).** Fixed or tracked random/low-rank subspaces used for **training/memory efficiency**, not measurement; the last one does analyze how much gradient energy stays in the residual bulk (relevant descriptive statistic). **Diff:** subspaces are the update mechanism; no per-direction inference over time.

**9. Cockpit, Schneider et al. (2021) — [arXiv:2102.06604](https://arxiv.org/pdf/2102.06604).** The canonical "measurement infrastructure" paper: per-iteration gradient SNR-type instruments, α-quantity, etc. **Diff:** all per-step or short-window; coordinate/scalar level; no fixed probes, no cumulative tests.

**10. "Spectral Edge Dynamics of Training Trajectories: Signal–Noise Geometry Across Scales" (2026) — [arXiv:2603.15678](https://arxiv.org/pdf/2603.15678).** Rolling-window SVD of parameter deltas, separating signal directions from a noise bulk via the spectral edge. **Diff:** rolling (bounded) windows, evolving directions, matrix-spectrum statistic rather than per-direction hypothesis test.

Also checked and further away: Chatterjee & Zielinski gradient coherence ([arXiv:2008.01217](https://ar5iv.labs.arxiv.org/html/2008.01217) — per-example alignment within a batch, not over time), PCA-of-trajectories ([arXiv:1806.08805](https://arxiv.org/pdf/1806.08805)), NN-CUSUM change-point work ([arXiv:2210.17312](https://arxiv.org/abs/2210.17312) — data streams, not gradients-in-training), loss-landscape slicing along Hessian vs random directions ([arXiv:2208.13219](https://arxiv.org/abs/2208.13219) — random directions found *uninformative* for visualization, a different use), Muon momentum spectral-filtering analysis ([arXiv:2606.03899](https://arxiv.org/pdf/2606.03899)).

## (2) Novelty verdict

**Unreported, with partial anticipation. Confidence: moderate-high.** No paper found that (a) freezes random probe directions for an entire run, (b) accumulates a per-direction t/SNR statistic with an **unbounded** integration window, and (c) uses the |t| ≤ SNR·√ESS instrument limit to distinguish "no persistent signal" from "insufficient integration." The two nearest neighbors each have half: Yao et al. 2026 has fixed directions + full-run cumulative testing but in a kernel-regime function space for early stopping; Antil & Verma 2025 has fixed random gradient probes but EMA-windowed monitoring with no inference. Litman & Guo 2026 supply the theory (drift ~T vs martingale ~√T per direction) but no instrument. Caveat on confidence: this literature is scattered across optimization, statistics, and empirical-DL venues, and a workshop paper doing exactly this could evade keyword search; but ten-plus distinct query formulations produced no direct hit.

## (3) What the strongest published results imply about persistent per-direction signal

Three mutually consistent signals:
- **Litman & Guo (2605.01172):** on directions carrying population signal, SGD drift accumulates linearly in T while zero-mean noise accumulates as √T — so a frozen-direction t-statistic *should* grow ~√t wherever persistent signal exists, and wash out (√T/T → 0) on pure-noise directions. This is the direct theoretical warrant for the slow tier: the separation is real but only visible at long integration times.
- **Song, Ahn & Yun (ICLR 2025):** the large per-step gradient components along top-Hessian (dominant) directions are oscillatory and do not drive loss decrease; the loss-reducing motion lives in the small bulk components. Implication: per-window statistics on tracked/top directions will be dominated by oscillation, and the persistent signal per direction is small — precisely the regime where |t| is SNR-capped in any bounded window and only unbounded integration can resolve it. This is arguably the strongest published empirical evidence that our "per-window statistics can't see it" premise is correct.
- **GSNR lineage (Liu et al. 2020; 2309.13681):** per-parameter gradient mean²/variance is measurable and nonzero but decays over training — persistent signal exists but gets weak, again implying long windows are needed late in training. No GSNR successor found that combines fixed projections with growing windows.

## (4) Queries run

1. gradient projection fixed random direction training dynamics measurement
2. GSNR per-direction long window statistic
3. streaming hypothesis test gradient stationarity SGD per-coordinate
4. random/Hessian fixed-direction loss-landscape probing + curvature tracking
5. Gur-Ari tiny subspace 2024–2025 follow-ups
6. cumulative t-statistic gradient drift detection sequential test
7. Song/Ahn/Yun tiny-subspaces paper (targeted)
8. random probe vectors, fixed random vector gradient inner products in LMs
9. gradient coherence / gradient mean vs noise per-direction anisotropy
10. frozen Hessian eigenvector curvature tracking
11. CUSUM on stochastic gradients 2024–2025
12. random-projection SGD trajectory diagnostics / gradient sketching monitors
13. drift-diffusion √T theory paper (targeted, + fetch)
14. per-direction momentum signal/noise classification in optimizers
15. SASA/growing-window stationarity tests + successors
16. spectral analysis of gradient time series, edge-of-stability per-eigenvector
17. Fetches: arXiv 2605.27991, 2510.00442, 2605.01172

Sources: [arXiv:2605.27991](https://arxiv.org/html/2605.27991), [arXiv:2605.01172](https://arxiv.org/abs/2605.01172), [arXiv:2405.16002](https://arxiv.org/abs/2405.16002), [arXiv:2510.00442](https://arxiv.org/html/2510.00442v1), [arXiv:2309.13681](https://arxiv.org/abs/2309.13681), [arXiv:2310.07361](https://arxiv.org/abs/2310.07361), [arXiv:2002.10597](https://arxiv.org/abs/2002.10597), [arXiv:1910.08597](https://arxiv.org/pdf/1910.08597), [arXiv:1710.06382](https://arxiv.org/pdf/1710.06382), [arXiv:2412.11341](https://arxiv.org/pdf/2412.11341), [arXiv:1812.04754](https://arxiv.org/abs/1812.04754), [arXiv:1902.02366](https://arxiv.org/pdf/1902.02366), [arXiv:2606.30226](https://arxiv.org/pdf/2606.30226), [arXiv:2102.06604](https://arxiv.org/pdf/2102.06604), [arXiv:2603.15678](https://arxiv.org/pdf/2603.15678), [arXiv:2510.01878](https://arxiv.org/html/2510.01878v1), [SubTrack++ OpenReview](https://openreview.net/pdf?id=6geRIdlFWJ), [arXiv:2008.01217](https://ar5iv.labs.arxiv.org/html/2008.01217), [arXiv:2208.13219](https://arxiv.org/abs/2208.13219), [arXiv:1806.08805](https://arxiv.org/pdf/1806.08805), [arXiv:2210.17312](https://arxiv.org/abs/2210.17312), [arXiv:2606.03899](https://arxiv.org/pdf/2606.03899), [Low Dimensional Trajectory Hypothesis](https://www.researchgate.net/publication/360883049_Low_Dimensional_Trajectory_Hypothesis_is_True_DNNs_can_be_Trained_in_Tiny_Subspaces)
