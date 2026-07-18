# WP0.3 — DynMuon reproduction plan (smallest config exhibiting the claimed gap)

**Status:** plan + source extraction only. No runs performed, no reproduction verdicts, no gate evaluation. Per CLAUDE.md, interpretation of DynMuon's theory is human-only; this report extracts quotes and structures a protocol.

**Sources:**
- Code: `vendor/DynMuon/` (pinned submodule; README, `train_gpt.py`, `dynmuon/dynmuon.py`, `configs/`).
- Paper: DynMuon, Wu et al., arXiv:2605.17109v3 (PDF fetched 2026-07-18; not in-repo — page/table citations below refer to the v3 PDF).
- Cross-reference: `vendor/modded-nanogpt/records/track_3_optimization/README.md`.

---

## 1. The claim, verbatim

- `vendor/DynMuon/README.md:12`: "Extensive experiments across model sizes, architectures, and training settings show that DynMuon consistently achieves lower validation loss than Muon, while requiring **10.6-26.5%** fewer steps to reach the same target loss."
- Paper abstract (p.1): "…10.6–26.5% fewer steps to reach the same target loss."
- Paper §4 Main Results (p.8): "Across model scales, DynMuon reaches the target 10.6–26.5% earlier than Muon, requiring substantially fewer training steps to reach the same validation loss." And: "DynMuon has a per-step time ratio of only 1.003–1.025× relative to Muon".

Paper Table 1 (p.8), transcribed verbatim (caption: "Performance and efficiency of DynMuon relative to Muon across GPT-style model scales. Steps to Target uses the validation loss reached by Muon at 80% of training as the target. Step Saving reports the relative step reduction, and Per-Step Time is the average ms/step."):

| Tokens | Method (Size) | Best Val. Loss | Steps to Target | Step Saving | Per-Step Time (ms) |
|---|---|---|---|---|---|
| 10B | Muon (127M) | 3.190 | 16000 | 0.0% | 1142.4 |
| 10B | DynMuon (127M) | 3.171 | 12500 | 21.9% | 1150.3 |
| 10B | Muon (601M) | 2.872 | 16000 | 0.0% | 4121.7 |
| 10B | DynMuon (601M) | 2.858 | 13950 | 12.8% | 4200.1 |
| 10B | Muon (1.1B) | 2.788 | 16000 | 0.0% | 6883.3 |
| 10B | DynMuon (1.1B) | 2.776 | 14300 | 10.6% | 7055.8 |
| 20B | Muon (127M) | 3.139 | 30400 | 0.0% | 1137.3 |
| 20B | DynMuon (127M) | 3.124 | 22350 | 26.5% | 1151.8 |
| 20B | Muon (601M) | 2.808 | 30400 | 0.0% | 4126.2 |
| 20B | DynMuon (601M) | 2.797 | 25000 | 17.8% | 4184.8 |
| 20B | Muon (1.1B) | 2.722 | 30400 | 0.0% | 6889.77 |
| 20B | DynMuon (1.1B) | 2.713 | 26450 | 13.0% | 6910.1 |

Range endpoints: 26.5% = 127M at 20B tokens; 10.6% = 1.1B at 10B tokens.

Model scales, paper Table 2 (p.14): "127M: 512 / 24 / 8 (d_model / Layers / Heads), 0.524M Tokens/Step, 20K Total Steps, 10B Total Tokens; 601M: 1280 / 24 / 20; 1.11B: 1792 / 24 / 28" — all at 0.524M tokens/step. Appendix D (p.18): "global batch size of 512, per-device batch size 64, and sequence length 1024"; token budgets "2.5B to 20B tokens, corresponding to 5K, 10K, 20K, and 38K training steps, respectively"; dataset FineWeb (FineWeb-Edu as robustness check). Devices (App. D, p.18): "We use NVIDIA H200 GPUs for all experiments." Seed robustness (App. E, p.19): "we run Muon and DynMuon with three different seeds {0,1,42}".

## 2. Smallest config exhibiting the claimed gap

**The 127M GPT-style model (d_model 512, 24 layers, 8 heads), sequence length 1024, global batch 512, 20K steps / 10B FineWeb tokens — reported Step Saving 21.9% (Muon 16000 → DynMuon 12500 steps to target), which lies inside the claimed 10.6–26.5% range** (paper Table 1, p.8; Table 2, p.14). The 26.5% endpoint requires the 20B-token budget (38K steps) on the same model — roughly double the compute; it is the fallback if the human wants the range endpoint itself.

**Exact launch config:** there is no YAML for this model in the repo. The four YAMLs in `vendor/DynMuon/configs/` (`muon_160m.yaml`, `normuon_160m.yaml`, `dion_160m.yaml`, `dion2_160m.yaml`) describe a different setup (768/12/6, batch 1024, 3000 iterations, `scalar_opt: lion`, `lr: 0.02`) that does not match any row of paper Table 2 — and there is no `dynmuon_*.yaml` at all. The paper's 127M config is reachable only via the CLI command published in `vendor/DynMuon/README.md:40–57`:

```bash
torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --optimizer dynmuon --scalar_opt adamw --lr 0.01 \
    --batch_size 512 --device_batch_size 64 --sequence_length 1024 \
    --num_iterations 20000 --model_dim 512 --n_layer 24 --n_head 8 \
    --dynmuon_pmax 1.0 --dynmuon_pmin -0.25 --dynmuon_w 0.04 --dynmuon_tau 0.04
```

Muon side: identical command with `--optimizer muon` and without the `--dynmuon_*` flags (optimizer selection in `train_gpt.py:133–134`; Muon default hyperparameters shared via the same `Hyperparameters` dataclass, `train_gpt.py:46–95`: `mu=0.95`, `weight_decay=0.01`, `scalar_lr=0.001`, `warmup_ratio=0.01`, `warmdown_ratio=0.2`, `adjust_lr="spectral_norm"`). Gradient accumulation is native: `grad_accum_steps = hp.batch_size // sequences_in_global_batch` (`train_gpt.py:680–689`), so single-GPU runs use 8 micro-batches of 64×1024 tokens. Data: `python data/cached_fineweb100B.py 100` (`README.md:29–35`); 20K steps × 0.524M tokens/step ≈ 10.5B tokens (derived).

**Parameter discrepancies to flag for the human (do not resolve silently):**
1. **Schedule width/center:** paper App. D (p.18) states "For DynMuon, we use w = 0.01, τ = 0.02 … by default"; the repo README command and `train_gpt.py:92–93` default to `dynmuon_w = 0.04`, `dynmuon_tau = 0.04`; the `Logistic_Scheduler` class defaults are `tau_ratio=0.02, width_ratio=0.08` (`dynmuon/dynmuon.py:44–49`, overridden by `train_gpt.py:499–502`). Proposed default for the repro: the as-released README/train_gpt values (0.04/0.04) — human to confirm.
2. **Seed:** `train_gpt.py:40–44` hardcodes `seed = 0` for random/numpy/torch with no CLI argument. Reproduction under our seed discipline (dev seeds ≥ 1000; vendor code unmodifiable) requires a thin out-of-tree driver in `src/`/`scripts/` that sets seeds and launches, or an audited copy of the entrypoint — implementation decision deferred to the WP0.3 execution step; listed under Risks.
3. The 160m YAMLs' `lr: 0.02` and `scalar_opt: lion` do not match the paper protocol (lr 0.01, AdamW scalar, App. D); the repro follows the paper/README values.
4. **LR-schedule shape:** paper App. D (p.18) states "a linear warmup for the first 0.01 of training steps, followed by a cosine decay over the remaining steps, with a final warmdown ratio of 0.2", but the released `get_lr` (`train_gpt.py:795–803`) implements linear warmup → constant → *linear* warmdown over the last `warmdown_ratio` of steps (no cosine anywhere in the schedule). The repro runs the code as released; the human may want to note this when comparing against Table 1.

## 3. What "steps to target loss" means in their harness

- The harness itself never computes it. `train_gpt.py` evaluates validation loss every `val_loss_every` steps (default 50, `train_gpt.py:66`; evaluation and logging at `train_gpt.py:904–936`) over a fixed `val_tokens = 10485760` (`train_gpt.py:67`).
- The paper defines the metric post-hoc: "we define a fixed target as the validation loss reached by Muon at 80% of training, and record the first step at which DynMuon reaches it" (§4 Main Results, p.8; Table 1 caption). For the 10B/20K-step budget, 80% of training = step 16000; for 20B/38K steps = step 30400 — matching the Muon "Steps to Target" column exactly.
- Consequently steps-to-target is quantized to the validation cadence; every Table 1 entry is a multiple of 50, consistent with `val_loss_every = 50` (inference from defaults + table granularity; the paper does not state the cadence explicitly).
- Our aggregation script must therefore: (a) fix target = Muon-arm val loss at the 80% checkpoint, (b) scan the val-loss trace of each arm for the first step ≤ target, (c) report saving % = 1 − steps_DynMuon/steps_Muon.

## 4. Estimated runtime and GPU (estimates flagged; no measurements)

- Paper hardware: NVIDIA H200 (App. D, p.18). GPU count per run is not stated in the paper; the released command is single-GPU (`--nproc_per_node=1`, `README.md:41`) with 8-way grad accumulation.
- Derived from Table 1 per-step times (127M: 1142.4–1151.8 ms/step): 20,000 steps × ~1.15 s ≈ **~6.4 h per run** in the paper's setup. 5K-step (2.5B-token) pilot runs ≈ ~1.6 h each. These scale H200→H100 by an unknown factor; treat as order-of-magnitude.
- Suitable GPU per plan hardware guidance ("instrumented nanogpt runs want a single A100/H100 or a 2× GPU VM with DDP", `routed-muon-research-plan.md` Hardware section): **a single H100 80 GB Hyperstack VM** for all comparison runs. Unlike the modded-nanogpt records, this code has no FP8 requirement, so A100-80GB is also technically feasible — but the plan's same-GPU-type rule means every run in the WP0.3 comparison uses one chosen type. DynMuon's Triton Newton-Schulz kernel is disabled in the DynMuon instantiation path (`use_triton=False`, `train_gpt.py:497`).
- Local 2×RTX 5090 box: dev/debug smoke runs only (short `--num_iterations`, reduced `--device_batch_size` if 32 GB VRAM requires it, dev seeds ≥ 1000); not comparable numerics, never in the comparison table.
- Wandb: the harness imports wandb; use `--no_wandb` (`train_gpt.py:191`) in scripted runs.

## 5. Equal-tuning-effort protocol (per CLAUDE.md WP0.3: "equal tuning effort documented")

Principle: both arms receive the identical tuning procedure, budget, and seed set; DynMuon-only knobs stay at released defaults (they have no Muon counterpart, so sweeping them would create unequal effort — this asymmetry is documented rather than "compensated").

1. **Swept for BOTH arms, identically:** learning rate over the paper's own grid {0.003, 0.005, 0.01, 0.02, 0.04} (App. D, p.18: "We tune the learning rate over {0.003, 0.005, 0.01, 0.02, 0.04} for Muon and DynMuon"). One pilot run per (arm, LR) at the 2.5B-token/5K-step budget (paper's smallest tabled budget, App. D), one dev seed (≥ 1000), best-val-loss selects each arm's LR. 10 pilot runs total.
2. **Frozen identically for both arms:** `mu=0.95`, `weight_decay=0.01`, scalar AdamW at `scalar_lr=0.001`, `warmup_ratio=0.01`, `warmdown_ratio=0.2`, `adjust_lr="spectral_norm"`, batch 512, seq 1024, `val_loss_every=50`, `val_tokens=10485760` (all from `train_gpt.py:46–95` defaults, matching App. D).
3. **Frozen for DynMuon only (released defaults, human to confirm w/τ per §2):** `pmax=1.0`, `pmin=−0.25`, `w`, `tau` (`train_gpt.py:90–93`).
4. **Final comparison:** 127M / 10B tokens / 20K steps, both arms at their pilot-selected LR, ≥3 seeds per arm (paper used 3 seeds {0,1,42} for its seed-robustness check, App. E; our seed values resolved by launch tooling per repo seed policy). Report: steps-to-target saving per seed pair, mean ± CI, plus best-val-loss deltas — alongside the paper's 21.9%/16000→12500 reference values. Per ground rules, the report states the observed gap and tuning ledger; it draws no pass/fail conclusion.
5. **Tuning ledger (deliverable):** a table of every run on each side (config hash, seed, budget, result) demonstrating run-count and budget parity.
6. **Cost estimate (derived from §4 rates; flag as estimate):** pilots 10 × ~1.6 h ≈ 16 GPU-h; finals 6 × ~6.4 h ≈ 38 GPU-h; total ≈ **54 H100/H200-GPU-hours** plus margin. Cloud-launched by the human per the compute boundary; agent supplies configs + entrypoints; results consumed from `results/`.

Minimal fallback if budget is constrained: skip the sweep, run both arms at the paper-default lr 0.01, 3 seeds (≈ 38 GPU-h) — weaker claim ("as-released settings"), equal effort trivially satisfied; the plan's "how much tuning" question is then answered as "none beyond released defaults".

## 6. Cross-reference (informational)

DynMuon has a validated entry on the modded-nanogpt optimization track: "#28 | 3175 | 3.2790 (n=25)✓ | DynMuon (p: 0.25 -> -0.25, tau=0.04, w=0.04, lr=0.02, wd=0.025) | 2026/05/19" (`vendor/modded-nanogpt/records/track_3_optimization/README.md:85`; also `vendor/DynMuon/README.md:16`). That is a different harness (modded-nanogpt architecture, fixed batch/data) and different hyperparameters from the paper's Table 1 runs; it is not the WP0.3 target but is a second, independently validated config pair the human may use for sanity cross-checks.

## 7. Risks / ambiguities (findings, not gaps to fill)

1. Hardcoded `seed = 0` with no CLI override (`train_gpt.py:40–44`) — needs an out-of-tree seeding driver; without it all runs share one seed and violate our seed discipline.
2. w/τ triple discrepancy (paper 0.01/0.02 vs released defaults 0.04/0.04 vs class defaults 0.08/0.02) — human decision required before launch.
3. GPU count behind Table 1 per-step times is unstated — wall-time estimates are order-of-magnitude only.
4. Validation cadence behind Table 1 is inferred (multiples of 50), not stated.
5. The 26.5% endpoint lives at the 20B-token budget (≈2× cost); the proposed smallest config reproduces the 21.9% cell, not the range endpoint.
6. FineWeb snapshot drift between the paper's runs and `data/cached_fineweb100B.py` downloads today cannot be excluded from here.
7. No training logs, loss traces, or results files are present in `vendor/DynMuon` — there is nothing in-repo to overlay against; the reproduction target is the Table 1 numbers alone.

## Numbers audit

Every number above is read from a cited file/line in `vendor/DynMuon/`, a cited table/page of arXiv:2605.17109v3, or explicitly labeled as derived/estimate with arithmetic shown. No reproduction runs were performed and no verdicts are stated.
