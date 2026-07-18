# DynMuon — Verbatim Quote Extraction (for the human-written differentiation memo)

**Purpose.** Raw material for the differentiation memo per plan "Risks & honest priors" item 1. Per CLAUDE.md, interpretation is human-only: this file contains verbatim quotes and source locations ONLY. No paraphrase inside quote blocks, no interpretation, no conclusions.

**Sources used.**

1. **Vendored repo:** `vendor/DynMuon` at pinned SHA `89baa66693819ef09e26915a5b46bccfc77913eb` (files: `README.md`, `NOTICE.md`, `dynmuon/dynmuon.py`, `train_gpt.py`, and greps across all `*.py`/`*.md`/`*.yaml`).
2. **arXiv paper:** the repo README cites arXiv:2605.17109 ("DynMuon: A Dynamic Spectral Shaping View of Muon", Wu, Shah, Silwal, Zhang). LaTeX source of **v3** (arXiv e-print, archive last modified 2026-06-02; abs page lists v1 2026-05-16, v2 2026-05-22, v3 2026-06-01) downloaded 2026-07-18 from `https://arxiv.org/e-print/2605.17109`. File/line references below are into that source archive (top-level `neurips_2026.tex`, sections under `src/`, tables under `tables/`). No HTML render exists on arXiv ("No HTML for '2605.17109v3'"); quotes are taken directly from the `.tex` files, LaTeX markup preserved. Note: the macro `\sys` is defined as the system name — `src/preamble.tex:109`: `\newcommand{\sys}{\tsf{DynMuon}\xspace}`.

Line numbers are exact for the files as retrieved. LaTeX commands, math, and citations are reproduced as-is.

---

## (a) Their p-selection theory — curvature, noise, and training-stage arguments

### Abstract-level statement of the theory's inputs

`src/abs.tex`, lines 3–5 (identical text in `vendor/DynMuon/README.md`, lines 8–10):

> In this work, we consider a class of Muon-like updates, where we replace the update $M$ with $U\Sigma^p V^\top$ for some parameter $p$.
> We call this a ``spectral-shaping'' operation, and develop a theory of how to pick $p$ which depends on (a) local curvature of the loss function, (b) noise stemming from stochastic gradients and label noise, and (c) training stage.
> Our theory and experimentation reveal a previously overlooked behavior: positive $p$ helps early by emphasizing high-curvature directions and accelerating signal contraction, while mildly negative $p$ helps later by reallocating update strength toward low-curvature directions that still contain useful training signals.

### Scope of the model

`src/prelim.tex`, lines 5–8:

> Before proposing our method, we first develop a simple noise-aware local model, using a standard local quadratic and gradient-noise approximation, to isolate how $p$ controls the trade-off between useful training signal and gradient noise~(\Cref{sec:modeling}). We note that our modelling and analysis make several simplifying approximations (which we empirically validate).
> Our goal is not to provide a full end-to-end convergence theory, but rather to introduce a minimal set of reasonable approximations that yield mechanistic insights into the role of the spectral exponent $p$ in training dynamics.
> In particular, our modelling predicts two stage-dependent regimes: positive $p$ can benefit early training, whereas mildly negative $p$ can improve late-stage training.
> We \emph{empirically validate these predictions} in~\Cref{sec:positive,sec:negative,sec:validation}, showing that our simplified modelling can serve as a useful and \emph{predictive} guide for designing dynamic spectral shaping.

### The local model: curvature setup

`src/prelim.tex`, lines 26–37:

> Since the update is driven by the gradient, we relate the gradient to the residual signal $E_t$ by performing a locally linear approximation of the population gradient around $W^\star$~\citep{nocedal2006numerical}.
> We further use a one-sided Kronecker-factored approximation to the local Hessian action, in the spirit of K-FAC~\citep{pmlr-v37-martens15}.
> We apply this approximation over a short training window in which the effective local curvature is empirically stable (see Figure~\ref{fig:grad2-curv-alignment} for empirical support):
> \begin{equation}
> \nabla L(W_t) \approx \nabla L(W^\star)+\nabla^2 L(W^\star)[E_t] \approx \kappa_t H E_t,
> \label{eq:local_approx}
> \end{equation}
> \vskip -0.1in
> where $H=Q\Lambda Q^\top$ is a normalized effective local curvature matrix with $\Lambda=\mathrm{diag}(h_1,\dots,h_m)$, $h_i\in(0,1]$, and $\max_i h_i =1 $.
> The scalar $\kappa_t>0$ captures the overall curvature scale.
> The eigenvectors of $H$ define the \emph{modes} of the local loss landscape, where each mode corresponds to one curvature direction and $h_i$ measures the curvature along that direction.
> Modes with larger $h_i$ are called \emph{\textbf{strong modes}} and modes with smaller $h_i$ are called \emph{\textbf{flat modes}}.

### Noise assumption and curvature-aligned surrogate

`src/prelim.tex`, lines 38 and 46–51:

> To account for stochasticity in the actual training update, we decompose the stochastic gradient $G_t$ into the population gradient and a zero-mean noise $\Xi_t$, following standard unbiased SGD assumptions~\citep{doi:10.1137/16M1080173}.

> Since the shaped update depends on $(G_tG_t^\top)^{\frac{p-1}{2}}$, we relate the spectral structure of $G_tG_t^\top$ to the effective local curvature.
> Motivated by the common use of gradient second moments as Fisher-type proxies for local curvature~\citep{NEURIPS2019_46a558d9,JMLR:v21:17-678}, we use a curvature-aligned surrogate:
> $(G_t G_t^\top)^{\frac{p-1}{2}} \approx \alpha_t^{\frac{p-1}{2}} H^{\frac{p-1}{2}}$,
> where $\alpha_t> 0$ is a scalar factor.
> This approximation models spectral shaping as a curvature-dependent reweighting.
> The underlying gradient-curvature alignment is empirically supported in Appendix~\Cref{fig:grad2-curv-alignment}.

### Mode-wise dynamics and the signal–noise trade-off

`src/prelim.tex`, lines 63–72 (mode-wise projection and boxed dynamics):

> Since $p$ acts through powers of the curvature matrix $H=Q\Lambda Q^\top$ in~\Cref{eq:matrix_error_dynamics_model}, its effect can vary across curvature directions.
> This motivates a mode-wise analysis, where we project the residual signal $E_t$ and noise $\Xi_t$ onto the eigenbasis of $H$.
> Define $\widetilde{E}_t := Q^\top E_t$ and $Z_t := Q^\top \Xi_t$.
> Let $\delta_{i,t}$ and $\xi_{i,t}$ denote the $i$-th coordinates of $\widetilde{E}_t$ and $Z_t$.
> Then mode $i$ evolves as
> % \vspace{-0.5em}
> \begin{equation}
> \boxed{\delta_{i,t+1} = \left(1-\eta_t \kappa_t h_{i}^{\frac{p+1}{2}}\right)\delta_{i,t} - \eta_t h_{i}^{\frac{p-1}{2}}\xi_{i,t}.}
> \label{eq:scalar_dynamics_model}
> \end{equation}

`src/prelim.tex`, lines 87–92:

> Thus, $p$ induces a \emph{mode-wise signal--noise trade-off}.
> The deterministic multiplier $1-\eta_t\kappa_t h_i^{(p+1)/2}$ controls residual-signal contraction: within the stable range $(0,1)$, larger $h_i^{(p+1)/2}$ makes this multiplier smaller and contracts $\delta_{i,t}$ faster.
> Increasing $p$ therefore favors contraction in strong, high-curvature modes, whereas decreasing $p$ increases the relative contraction strength in flat modes.
> However, the stochastic term is scaled by $h_i^{p-1}$, so decreasing $p$ also amplifies noise most strongly in flat modes.
> Without noise, $p=-1$ would maximize contraction in flat modes; with noise, the choice of $p$ must balance residual-signal contraction against noise amplification.
> This decomposition also provides the basis for the mode-wise predictions tested in~\Cref{sec:validation}, where we empirically estimate the residual-signal ``energy'' $\delta_{i,t}^2$ and noise level $c_{i,t}$ during training.

`src/prelim.tex`, lines 95–98 (boxed takeaway):

> \textbf{Takeaway:}
> The spectral exponent $p$ controls a mode-wise signal--noise tradeoff.
> Larger $p$ accelerates residual signal contraction in strong modes, while smaller $p$ shifts more emphasis toward flat modes but also amplifies noise along them.

### Training-stage argument: late stage (negative p)

`src/prelim.tex`, lines 107–113:

> \noindent\textbf{Why Residual Signal Concentrates in Flat Modes Late in Training.}
> For $h_i\in(0,1]$, the contraction strength $h_i^{\frac{p+1}{2}}$ increases with $h_i$.
> Thus, the residual signal in strong modes tends to decay earlier, whereas the signal in flat modes decays more slowly and can remain substantial later in training.
> This suggests that the residual signal becomes relatively more concentrated in flat modes as training progresses.
> This is also consistent with the local gradient approximation in~\Cref{eq:local_approx}.
> Projecting the population gradient onto mode $i$ gives $g_{i,t} \approx \kappa_t h_i \delta_{i,t}$, hence $\delta_{i,t} \approx \frac{g_{i,t}}{\kappa_t h_i}$.
> Thus, for comparable projected gradient magnitudes, a smaller curvature $h_i$ corresponds to a larger residual signal.

`src/prelim.tex`, lines 116–122:

> \noindent\textbf{When a Mildly Negative Exponent Helps in the Late Stage.}
> Once the residual signal becomes more concentrated in flat modes in the late training stage, it can be beneficial to place relatively more emphasis on those modes rather than on strong modes whose residual signal has already decayed.
> % Decreasing $p$ achieves this effect.
> Decreasing $p$ achieves this effect by allocating relatively more contraction to flat modes.
> However, from~\Cref{eq:one_step_second_moment}, lowering $p$ also increases the noise level, especially in flat modes.
> Thus, a negative $p$ can improve over $p=0$ only when the residual signal in flat modes is large enough relative to the noise.
> This also explains why the exponent should be only mildly negative: otherwise, noise amplification can outweigh the benefit from reducing the residual signal and degrade optimization.

### Training-stage argument: early stage (positive p)

`src/prelim.tex`, lines 209–211:

> Our analysis suggests otherwise: early residual signal can remain concentrated in high-curvature modes, and increasing $p$ accelerates contraction of these strong-mode residual signal while reducing the noise-amplification factor $h_i^{\frac{p-1}{2}}$.
> Thus, before the residual distribution shifts toward flat modes, a positive exponent can accelerate early optimization by prioritizing the strong-mode residual signal with limited noise amplification.

`src/prelim.tex`, lines 226–227 (boxed takeaway):

> \textbf{Takeaway:}
> A positive early-stage exponent helps reduce strong-mode residual signal before the remaining residual signal shifts toward flat modes, yielding lower validation loss than fixed Muon.

### How the theory's mode-wise quantities are measured (validation instrumentation)

`src/prelim.tex`, lines 159–163:

> \emph{\underline{Empirical Modes and Proxies.}}
> In our analysis, modes are curvature directions, which are expensive to track directly in practice.
> We therefore use the singular directions of the Muon update as empirical modes, since spectral shaping reweights singular values along these directions.
> For each empirical mode $i$ at step $t$, we estimate local curvature $\hat{h}_{i,t}$, mode-wise noise level $\hat{c}_{i,t}$, and residual-signal energy $\hat{\delta}_{i,t}^{2}$.
> Specifically, $\hat{h}_{i,t}$ is estimated via Hessian-vector products, $\hat{c}_{i,t}$ from the variance of independent mini-batch gradient projections, and $\hat{\delta}_{i,t}^{2}$ from fixed-probe gradient projections using the local gradient approximation in~\Cref{eq:local_approx}.

`src/appendix.tex`, lines 124–126 and 133–141:

> For efficiency, at each step $t$, we retain the top-$k$ directions with $k=256$ and represent each as
> $B_{i,t}:=u_{i,t}v_{i,t}^\top$.

> We estimate the mode-wise noise level by measuring how much the gradient projection varies across independent mini-batches.
> Specifically, for $n_b=32$ mini-batches, we compute
> \begin{equation*}
> g_{i,t}^{(b)} := \langle G_t^{(b)}, B_{i,t}\rangle,
> \end{equation*}
> and define the noise level proxy as the sample variance
> \begin{equation*}
> \hat{c}_{i,t} := \mathrm{Var}_{b\in[n_b]} [g_{i,t}^{(b)}].
> \end{equation*}

### Noise–curvature scaling and batch-size dependence

`src/appendix.tex`, lines 176–180:

> \noindent\textbf{Noise-Curvature Scaling.}
> Motivated by prior work on the geometry of gradient noise, we examine how noise level varies with curvature across modes by fitting a power-law relation $\hat{c}_{i,t} \asymp N_t \hat{h}_{i,t}^{\beta_t}$.
> The exponent $\beta_t$ describes the curvature dependence of noise, where a positive $\beta_t$ means that the raw noise level is larger on high-curvature modes and smaller on flat modes.
> Figure~\ref{fig:beta} (left) shows that $\beta_t$ remains stable around $1.4$, while Figure~\ref{fig:beta} (right) shows consistently high $R^2$ values, indicating that this power-law relation provides a reliable description of the mode-wise noise structure.
> Thus, although decreasing $p$ amplifies noise more strongly on flat modes, their raw noise level remains comparatively smaller, leaving room for slightly negative spectral shaping to exploit the flat-mode residual signal.

`src/appendix.tex`, lines 186–189:

> As shown in Figure~\ref{fig:noise_impact}, smaller batch sizes, which induce higher gradient noise, favor mildly negative exponents closer to $0$: the best exponent is $p=-0.1$ when the batch size is $2$, and shifts to $p=-0.25$ when the batch size is $16$.
> As the batch size further increases to $128$, the preferred exponent becomes more negative, with $p=-0.5$ achieving the best validation loss.
> This trend is consistent with our analysis: negative spectral shaping can improve late-stage optimization by emphasizing flat modes, but overly negative exponents also amplify noise and can degrade performance, especially when the gradient-noise level is high.
> These results also suggest that reducing gradient noise, for example, through larger batch sizes, may make more negative spectral exponents beneficial.

---

## (b) What is global / scheduled in their method

### The method is a schedule of a single exponent p over training

`src/introduction.tex`, lines 39–41:

> Motivated by these observations, we propose \sys, a dynamic spectral shaping algorithm that adapts the spectral exponent $p$ over training~(\Cref{sec:method}).
> It leverages a simple decreasing logistic schedule for $p$, interpolating from positive values early in training to mildly negative values later.
> To realize our scheduled spectral shaping efficiently, \sys extends  Newton--Schulz approximation to approximate $U\Sigma^pV^\top$ for varying values of $p$, avoiding full SVD and retaining the per-step cost as Muon.

`src/algo.tex`, lines 3 and 6–18:

> The above analysis and observations motivate \sys~(\Cref{alg:dyn-full} in~\Cref{app:algo}), which dynamically adapts spectral shaping by {monotonically} decreasing the spectral exponent from a positive early-stage value to a mildly negative late-stage value, while maintaining computational efficiency.

> \noindent\textbf{Logistic Scheduling of the Spectral Exponent.}
> Although the residual-signal distribution across modes can guide the choice of $p$, estimating it online would require additional forward and backward passes.
> We therefore use a simple logistic schedule to approximate a smooth decreasing transition of $p_t$ over training without this extra cost.
> Given the current training step $t$ and total steps $T$, we set
> \begin{equation*}
> u_t=\frac{t/T-\tau}{w},
> \qquad
> a_t=\frac{1}{1+\exp(u_t)},
> \qquad
> p_t=p_{\min}+a_t(p_{\max}-p_{\min}).
> \end{equation*}
> Here, $\tau$ controls the transition point and $w$ controls the transition width, with a smaller $w$ producing a sharper switch.
> We set $p_{\max}=1$ and $p_{\min}=-0.25$, where $p_{\min}=-0.25$ is the best-performing negative exponent based on our observations in~\Cref{sec:negative} (we also ablate $p_{\min}$; see Section \ref{sec:evaluation}).

### The schedule's input is only (step t, total steps T) — algorithm listing

`tables/algo.tex`, lines 10–23 (Algorithm 1, `\label{alg:dyn-full}`):

> \STATE \textbf{Input:} Update matrix $M$, step $t$, total steps $T$, schedule parameters $(p_{\max}, p_{\min}, \tau, w)$
>
> \STATE {\ttfamily \textcolor{red}{\(\triangleright\) /* Logistic Scheduling */}}
> \STATE $u \gets \left({t}/{T}-\tau\right)/w $
> \STATE $ a \gets 1/(1+\exp(u))$
> \STATE $p_t \gets p_{\min}+a(p_{\max}-p_{\min})$
>
> \STATE {\ttfamily \textcolor{red}{\(\triangleright\) /* Positive Anchoring */}}
> \IF{$p_t \geq 1/4$}
>     \RETURN $M$
> \ELSIF{$p_t \geq 0$}
>     \RETURN \texttt{Newton--Schulz}$(M)$
> \ELSE \RETURN \tsf{Fast--Spectral}$(M, p_t)$
> \ENDIF

### Stage-wise anchoring of the positive regime

`src/algo.tex`, lines 49–55:

> \noindent\textbf{Stable Anchoring for Positive Exponents.}
> \sys implements the spectral shaping scheduled exponent $p_t$ through a simple stage-wise scheme.
> The main reason is stability: when $p_t$ is positive and sufficiently large, the correction factor $\textcolor{red}{A^{\frac p2}}$ is no longer a mild adjustment to the Muon update, so the Taylor approximation around $A\approx I$ can become unreliable.
> To avoid this instability, \sys anchors the positive regime to two stable operators (lines 7--10 in~\Cref{alg:dyn-full}).
> \sys uses the original update when $p_t \geq 0.25$.
> For $p_t \in [0, 0.25)$, \sys applies standard NS orthogonalization, recovering the Muon-style update.
> For $p_t \in [p_{\min}, 0)$, the exponent is only mildly negative, so \sys uses the efficient spectral approximation described above for a continuous schedule.

### The exponent applies uniformly to the whole singular spectrum of each matrix

`src/introduction.tex`, lines 23–26:

> \begin{equation}
>     D^{(p)} := U \Sigma^p V^\top, \qquad \Sigma^p = \mathrm{diag}(\sigma_1^p,\dots,\sigma_r^p),
> \end{equation}
> for a matrix-valued update $M=U\Sigma V^\top.$ This is demonstratively a very expressive family of operations: $p=-1$ gives an inverse-spectrum update, $p=0$ recovers Muon, and $p=1$ corresponds to the standard SGD-style updates.

### Code: p is a process-global scalar set once per step for all matrices

`vendor/DynMuon/dynmuon/dynmuon.py`, lines 24–39:

```python
# global chosen hyperparams for this step
_GLOBAL_P = 0.0

def set_global_p(p: float):
    global _GLOBAL_P
    _GLOBAL_P = float(p)


def dynmuon_spectral_transform(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    if _GLOBAL_P >= 0.25:
        return G

    if _GLOBAL_P >= 0.0:
        return newton_schulz_triton(G, epsilon=epsilon)

    return fast_spectral(G, p=_GLOBAL_P)
```

`vendor/DynMuon/dynmuon/dynmuon.py`, lines 44–55 (scheduler class):

```python
class Logistic_Scheduler:
    def __init__(self, p_max=1.0, p_min=-0.25, tau_ratio=0.02, width_ratio=0.08):
        self.p_max = p_max
        self.p_min = p_min
        self.tau_ratio = tau_ratio 
        self.width_ratio = width_ratio

    def get_p(self, step, total_steps=10000):
        q_t = step / float(total_steps)
        u = (q_t - self.tau_ratio) / max(self.width_ratio, 1e-8)
        anneal = 1.0 / (1.0 + math.exp(u))
        return self.p_min + (self.p_max - self.p_min) * anneal
```

`vendor/DynMuon/dynmuon/dynmuon.py`, lines 186–192 (inside `step()`):

```python
        # schedule p
        if len(muon_groups) > 0:
            cur_step = max(group["step"] for group in muon_groups)
            p_t = self.log_psf.get_p(cur_step, self._total_steps)
            if self._device_rank == 0:
                print(f"[step {cur_step}], p: {p_t}")
            set_global_p(p_t)
```

### README description of the schedule parameters

`vendor/DynMuon/README.md`, lines 64–65:

> - `dynmuon_pmax=1.0`, `dynmuon_pmin=-0.25`: Spectral schedule range (transitions from positive to mildly negative)
> - `dynmuon_w=0.04`, `dynmuon_tau=0.04`: Logistic schedule width and center ratios

### Why the schedule (not an online estimate) is used

`src/algo.tex`, lines 7–8:

> Although the residual-signal distribution across modes can guide the choice of $p$, estimating it online would require additional forward and backward passes.
> We therefore use a simple logistic schedule to approximate a smooth decreasing transition of $p_t$ over training without this extra cost.

### Distributed/parallel claim

`src/appendix.tex`, lines 264–266:

> \noindent\textbf{Parallel Training.}
> \sys preserves the matrix-wise update structure of Muon and only changes the local spectral shaping rule through the scheduled exponent $p_t$.
> It therefore introduces no additional cross-layer or cross-device coupling beyond Muon, and should be compatible with existing Muon-style parallel training implementations.

---

## (c) Their claimed results and conditions

### Headline claims

`src/abs.tex`, line 7 (same text in `vendor/DynMuon/README.md`, line 12):

> Extensive experiments across model sizes, architectures, and training settings show that \sys consistently achieves lower validation loss than Muon, while requiring \textbf{10.6--26.5\%} fewer steps to reach the same target loss.

`vendor/DynMuon/README.md`, line 16:

> - [2026.05.31] 🚀 Our code is now released! DynMuon achieves a validated [Track 3 result of 3175 steps](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization), placing it among the strongest **standalone Muon-style optimization methods** on the modded-nanoGPT optimization benchmark.

### Definition of the step-saving metric and headline numbers

`src/experiments.tex`, lines 74–81:

> We train GPT-style models at three scales on FineWeb using both 10B and 20B token budgets.
> As shown in Figure~\ref{fig:main}, \sys consistently achieves the lowest validation loss across all three model scales compared with baselines.
> The improvement over Muon is clear in the late training stage, where scheduling the spectral exponent toward mildly negative values provides a stable advantage.
> Table~\ref{tab:efficiency} quantifies the practical significance of these gains in terms of both step efficiency and runtime overhead.
> For each model size and token budget, we define a fixed target as the validation loss reached by Muon at 80\% of training, and record the first step at which \sys reaches it.
> Across model scales, \sys reaches the target \textbf{10.6--26.5\%} earlier than Muon, requiring substantially fewer training steps to reach the same validation loss.
> Meanwhile, \sys has a per-step time ratio of only \textbf{1.003--1.025$\times$} relative to Muon, indicating negligible additional runtime cost.

### Per-configuration numbers (Table `tab:efficiency`)

`tables/overhead.tex`, lines 24–28 (caption) and 41–71 (data rows):

> Performance and efficiency of \sys relative to Muon across GPT-style model scales.
> Steps to Target uses the validation loss reached by Muon at 80\% of training as the target.
> Step Saving reports the relative step reduction, and Per-Step Time is the average ms/step.

Data rows verbatim (10B tokens block, lines 41–55):

> & Muon (127M) 
> & 3.190 & 16000 & 0.0\% & 1142.4  \\
> & \sys (127M)
> & \textbf{3.171} & \textbf{12500} & \textbf{21.9\%} & 1150.3  \\
> \cmidrule(lr){2-6}
> & Muon (601M)
> & 2.872 & 16000 & 0.0\% & 4121.7 \\
> & \sys (601M)
> & \textbf{2.858} & \textbf{13950} & \textbf{12.8\%} & 4200.1  \\
> \cmidrule(lr){2-6}
> & Muon (1.1B)
> & 2.788 & 16000 & 0.0\% & 6883.3  \\
> & \sys (1.1B)
> & \textbf{2.776} & \textbf{14300} & \textbf{10.6\%} & 7055.8 \\

Data rows verbatim (20B tokens block, lines 57–71):

> & Muon (127M)
> & 3.139 & 30400 & 0.0\% & 1137.3 \\
> & \sys (127M)
> & \textbf{3.124} & \textbf{22350} & \textbf{26.5\%} & 1151.8 \\
> \cmidrule(lr){2-6}
> & Muon (601M)
> & 2.808 & 30400 & 0.0\% & 4126.2 \\
> & \sys (601M)
> & \textbf{2.797} & \textbf{25000} & \textbf{17.8\%} & 4184.8 \\
> \cmidrule(lr){2-6}
> & Muon (1.1B)
> & 2.722 & 30400 & 0.0\% & 6889.77 \\
> & \sys (1.1B)
> & \textbf{2.713} & \textbf{26450} & \textbf{13.0\%} & 6910.1 \\

### Experimental conditions: models, data, batch, hardware

`src/experiments.tex`, lines 5–11:

> \noindent\textbf{Models and Datasets.}
> We evaluate \sys on two decoder-only Transformer families: GPT-style models at multiple scales following modded-nanoGPT~\cite{modded_nanogpt_2024} and a Qwen-style model, with detailed configurations summarized in~\Cref{tab:gpt_scales,tab:qwen_scales}.
> The GPT-style models use rotary position embeddings~\cite{su2024roformer}, RMSNorm, and squared ReLU MLPs~\cite{NEURIPS2021_2f3c6a4c}.
> The Qwen-style model uses pre-normalized Transformer blocks with RMSNorm, grouped-query attention, and gated SiLU MLPs.
> All models use sequence length $1024$ and global batch size $512$.
> Our main experiments use 10B tokens from FineWeb, and we additionally evaluate on FineWeb-Edu~\cite{NEURIPS2024_370df50c}.
> To study training-budget scaling, we vary the number of training tokens from 2.5B to 20B.

`src/appendix.tex`, lines 202–203:

> The GPT-style models use $d_{\mathrm{model}}=512, 1280,$ and $1792$, while the Qwen-style model uses $d_{\mathrm{model}}=512$.
> Across all models, we use the same training setup, with a global batch size of $512$, per-device batch size $64$, and sequence length $1024$.

`src/appendix.tex`, lines 222–223:

> \noindent\textbf{Devices.}
> We use NVIDIA H200 GPUs for all experiments.

### Baselines and tuning effort

`src/experiments.tex`, lines 13–19:

> \noindent\textbf{Baselines.}
> We compare \sys against Muon~\cite{jordan2024muon} and AdamW~\cite{loshchilov2018decoupled}.
> Muon is our primary and most directly relevant baseline, while AdamW serves as a standard, widely used optimizer baseline.
> We further include NorMuon~\citep{li2025normuonmakingmuonefficient} as an additional Muon-variant baseline in~\Cref{app:results}.
> Unless otherwise specified, the default learning rates are $0.01$ for Muon and \sys, and $0.002$ for AdamW.
> We also vary learning rates for all methods.
> By default, \sys uses $p_{\max}=1$ and $p_{\min}=-0.25$.

`src/appendix.tex`, lines 210 and 213–214:

> We tune the learning rate over $\{0.003, 0.005, 0.01, 0.02, 0.04\}$ for Muon and \sys, and over $\{10^{-4}, 3\times 10^{-4}, 5\times 10^{-4}, 10^{-3}, 2\times 10^{-3}, 4\times10^{-3}, 8\times10^{-3}\}$ for AdamW.

> For \sys, we use $w=0.01$, $\tau=0.02$, $p_{\max}=1$, and $p_{\min}=-0.25$ by default.
> For all methods, we use a linear warmup for the first $0.01$ of training steps, followed by a cosine decay over the remaining steps, with a final warmdown ratio of $0.2$.

### Robustness claims (LR, tokens, architecture, corpus)

`src/experiments.tex`, lines 99–101:

> \noindent\textbf{Learning Rate.}
> We test learning-rate robustness on the 127M model under the 10B-token budget by sweeping Muon and \sys over learning rates from $0.003$ to $0.04$.
> As shown in Figure~\ref{fig:token_lr} (right), \sys outperforms Muon across all tested learning rates and has a flatter curve near its optimum, indicating lower sensitivity to learning-rate choice.

`src/experiments.tex`, lines 95–96:

> To evaluate robustness across training budgets, we vary the number of training tokens from 2.5B to 20B for the 601M model.
> As shown in Figure~\ref{fig:token_lr} (middle), \sys consistently achieves lower validation loss than Muon across all tested budgets, suggesting that the benefit of dynamic spectral shaping is not tied to a particular training horizon.

`src/experiments.tex`, lines 90–91:

> To evaluate transferability beyond GPT-style models, we train a 171M Qwen-style decoder-only Transformer using the configuration in Table~\ref{tab:qwen_scales}.
> As shown in Figure~\ref{fig:token_lr} (left), \sys consistently achieves lower validation loss than Muon, suggesting that dynamic spectral shaping transfers across decoder-only architectures.

### Ablations: p_min, schedule shape, SVD comparison

`src/experiments.tex`, lines 110–113:

> \noindent\textbf{Ablation on $p_{\min}$.}
> We vary the scheduling endpoint $p_{\min}$ on the 127M model.
> As shown in Figure~\ref{fig:data_pmin} (middle), mildly negative choices outperform Muon, with $p_{\min}=-0.25$ achieving the best validation loss.
> More aggressive negative choices, e.g., $p_{\min}=-0.5$, perform worse, consistent with our analysis that overly negative exponents can degrade training performance.

`src/experiments.tex`, lines 122–127:

> \noindent\textbf{Ablation on Spectral Scheduling, Logistic Parameters $(\tau,w)$.}
> We ablate both the scheduling strategy and the logistic schedule parameters $(\tau,w)$.
> As shown in Figure~\ref{fig:ablation} (left), our default logistic schedule outperforms standard Muon and two ablations: an abrupt switch from $p=1$ to $p=-0.25$ at step 500, and a fixed negative schedule with $p=-0.25$ throughout training.
> The abrupt schedule underperforms the logistic schedule, suggesting that a smooth transition between spectral shaping is more effective than a sharp switch.
> The fixed negative schedule performs substantially worse, showing that negative shaping throughout training is insufficient and supporting our stage-dependent design.
> Figures~\ref{fig:ablation} (middle, right) further show that \sys is reasonably robust to the transition point $\tau$ and transition width $w$, with the best performance observed around $\tau=0.04$ and $w=0.04$.

`src/experiments.tex`, lines 118–120:

> We compare \sys with an exact-SVD implementation of the same dynamic spectral schedule.
> As shown in Figure~\ref{fig:data_pmin} (right), \sys closely matches exact SVD in validation loss, and both outperform Muon.
> Since exact SVD is roughly $3\times$ slower, \sys captures the benefit of dynamic spectral shaping at much lower cost.

### Alternative-explanation controls (LR bump, reversed schedule)

`src/prelim.tex`, lines 218–222:

> We further test two alternative explanations.
> First, to rule out the possibility that the early positive $p$ merely acts as a larger effective learning rate, we test a Muon variant that doubles the learning rate during the first 500 steps while keeping $p=0$.
> This variant performs worse than fixed Muon, indicating that the gain from an early positive $p$ is distinct from simple step-size scaling.
> Second, we evaluate a reverse schedule that switches from $p=-0.1$ to $p=1$ at step 500.
> This schedule also performs worse than fixed Muon, consistent with our analysis that positive $p$ should be applied early, while mildly negative $p$ is beneficial only later.

### Seed robustness and NorMuon comparison

`src/appendix.tex`, lines 228–231:

> \noindent\textbf{Seed Robustness.}
> To assess the robustness of \sys to training randomness, we run Muon and \sys with three different seeds $\{0,1,42\}$.
> Figure~\ref{fig:std} reports the mean validation loss with one-standard-deviation bands across seeds.
> \sys consistently outperforms Muon, and the very small across-seed variance suggests that the improvement is robust to training randomness rather than seed-specific effects.

`src/appendix.tex`, lines 234–237:

> \noindent\textbf{Comparison with NorMuon.}
> We further compare \sys with NorMuon~\cite{li2025normuonmakingmuonefficient}, a recent Muon variant that augments Muon orthogonalization with neuron-wise normalization based on second-moment statistics.
> We implement NorMuon following its original algorithmic design in our controlled setting.
> As shown in Figure~\ref{fig:normuon}, \sys consistently achieves lower validation loss than both Muon and NorMuon.

### Validation-experiment conditions (theory validation, not main results)

`src/appendix.tex`, lines 108–118:

> To validate the predictions of our model, we run experiments on a GPT-style model with hidden dimension 768, 12 layers, and 6 attention heads~\citep{ahn2025diondistributedorthonormalizedupdates}, trained on FineWeb~\citep{NEURIPS2024_370df50c}.
> We consider two settings: last-layer-only training and full-model training.
> For last-layer-only training, we freeze all model parameters and optimize only one selected matrix-valued parameter in the final Transformer block.
> We train for 6000 steps in the last-layer-only setting and 10000 steps in the full-model setting, using up to 3B training tokens.
> For optimization, we use a learning rate of $0.01$ for matrix-valued parameter updates under both Muon and stage-wise spectral shaping.
> For non-matrix parameter groups, we use AdamW with a learning rate of $0.001$.
> We apply a weight decay of $0.01$ to the main matrix parameter groups.
> The learning rate uses a linear warmup for the first $1\%$ of training steps, followed by cosine decay with a final warmdown ratio of $0.2$.
> For these validation experiments, we use exact SVD to compute spectral shaping for different exponents.
> For the experiments in~\Cref{sec:negative}, we switch from $p=0$ to a negative exponent at step 500 and compare $p \in \{-0.1,-0.25,-0.5,-0.75,-1\}$ in both last-layer-only and full-model training.

---

## (d) Statements about per-direction adaptation or temporal statistics

### Present: closest statements found

`src/conclusion.tex`, lines 6–11 (future-work statement — the only place adaptive/online selection of p is discussed):

> Our experiments focus on validating the hypothesis that practical improvements are obtainable by varying $p$ over training.
> We show that this schedule outperforms fixed $p$ choices and the standard $p=0$ Muon update, but it is not the final word on the matter.
> Indeed, the best choice of $p$ may depend on training-time dynamics that cannot be fully anticipated in advance.
> This suggests interesting directions for future work: selecting $p$ \emph{online} based on observed optimization statistics.
> Such methods could use more expensive signals, provided they are applied only occasionally rather than at every training step.
> Since the useful range of $p$ appears relatively small, we conjecture that a future near-optimal adaptive Muon-style scheduler may only need to re-estimate $p$ occasionally during training.

`src/algo.tex`, lines 7–8 (stated reason for not measuring online):

> Although the residual-signal distribution across modes can guide the choice of $p$, estimating it online would require additional forward and backward passes.
> We therefore use a simple logistic schedule to approximate a smooth decreasing transition of $p_t$ over training without this extra cost.

`src/introduction.tex`, line 45 (the "adaptive problem" framing — note the adaptation variable is the single exponent $p$):

> \item We introduce a dynamic spectral-shaping perspective for matrix-valued updates, reframing Muon-style optimization from a fixed spectral operation into the adaptive problem of choosing a suitable spectral exponent $p$ as training dynamics evolve.

`src/prelim.tex`, line 28 (temporal window statement in the modeling assumptions):

> We apply this approximation over a short training window in which the effective local curvature is empirically stable (see Figure~\ref{fig:grad2-curv-alignment} for empirical support):

`src/appendix.tex`, lines 156–160 (temporal tracking used in validation analysis only):

> \noindent\textbf{Stability of Effective Directional Curvatures.}
> To examine the local-stability assumption in~\Cref{eq:local_approx}, we track
> $\operatorname{median}_{i\in[k]}\log \hat{h}_{i,t}$ over the 4k--6k step window,
> where $\hat{h}_{i,t}$ denotes the empirical local curvature estimated along the corresponding direction $B_{i,t}$, as defined in~\Cref{sec:validation}.
> As shown in Figure~\ref{fig:grad2-curv-alignment} (left), this median changes only mildly over this window, supporting our use of an approximately fixed effective local curvature over short training windows.

Code (unused-looking state in the optimizer; recorded here because it is per-parameter EMA-named state) — `vendor/DynMuon/dynmuon/dynmuon.py`, lines 215–216, inside `_get_or_initialize_state`:

```python
        if "g2_ema" not in state:
            state["g2_ema"] = torch.zeros((), device=param.device, dtype=torch.float32)
```

Grep note: `g2_ema` appears only at these two lines in the entire repo (`grep -rn "g2_ema" --include="*.py"` over `vendor/DynMuon`); no code reads or updates it.

### Absent: no statement found

After searching, **no statement found** in the paper source (all files: `neurips_2026.tex`, `src/abs.tex`, `src/introduction.tex`, `src/problem.tex`, `src/prelim.tex`, `src/algo.tex`, `src/experiments.tex`, `src/discussion.tex`, `src/conclusion.tex`, `src/appendix.tex`, `tables/*.tex`) or the code repo (`vendor/DynMuon`, all `*.py`, `*.md`, `*.yaml`, `*.txt`) for any of the following:

- **Autocorrelation / lag statistics:** search terms `autocorrel`, `lag-1`, `lag 1` — zero hits in paper source and repo.
- **Temporal / time-series statistics of per-direction projections:** search terms `temporal`, `time series`, `time-series` — zero hits.
- **Oscillation / edge-of-stability:** search terms `oscillat`, `edge of stability` — zero hits.
- **EMA / moving-average statistics as part of the method:** search terms `EMA`, `moving average`, `exponential moving` — zero hits in the paper source; in code only the vestigial `g2_ema` state quoted above (never read or written after initialization).
- **Per-direction / per-mode choice of the exponent in the method:** search terms `per-direction`, `per direction`, `direction-wise`, `mode-wise p`, `p_i`, `different p per`, `each mode ... exponent` — zero hits describing the method. The word "mode-wise" appears only in the *analysis/validation* sections (e.g., "mode-wise signal--noise trade-off", "mode-wise noise level"), quoted under (a) above; the algorithm input (Algorithm 1, `tables/algo.tex` line 10) is a single matrix $M$ plus $(t, T, p_{\max}, p_{\min}, \tau, w)$ with a single scalar $p_t$ returned per step.
- **Routing / regime classification:** search terms `routing`, `route`, `classif`, `regime` — `classifiers` appears once as a Python packaging field (`vendor/DynMuon/setup.py`, line 61); `regime` appears in the paper only in the phrases "stage-dependent regimes" (`src/prelim.tex` line 7), "positive regime" (`src/algo.tex` lines 51–52), "mild-negative spectral regime" / "preferred mild-negative spectral regime" (`src/appendix.tex` lines 243, 251), and "a surprising stage-dependent regime" (`src/introduction.tex` line 47) — all quoted in context under (a)/(b) above. No hit describes classifying or routing individual directions.

Search-scope note on the bibliography file: the terms `routing`/`router` and `classif` do match entries inside `src/references.bib` (e.g., bib keys `lu2023routing`, `hu2024routerbench`, `song2025irt`, and abstracts of LLM-routing papers), but none of those 13 bib keys (`agrawal2025llmrank`, `hari2023tryage`, `jin2025radialrouter`, `jitkrittum2025universal`, `somerstep2025carrot`, `ding2024hybrid`, `feng2024graphrouter`, `lu2023routing`, `shnitzer2023large`, `song2025irt`, `stripelis2024tensoropera`, `wang2025mixllm`, `hu2024routerbench`) is cited anywhere in `neurips_2026.tex`, `src/*.tex`, or `tables/*.tex` (verified by grep per key). They are uncited entries in a shared bibliography file, not statements of the paper. Case-insensitive substring matches for `ema` likewise occur only inside words such as "remaining"/"scheme" and in `references.bib` abstracts; word-boundary `\bEMA\b`/`\bEMAs\b` has zero hits across `neurips_2026.tex`, `src/*.tex`, `tables/*.tex`, and `src/references.bib`.
