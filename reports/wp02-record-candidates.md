# WP0.2 — Candidate historical modded-nanogpt records for the pinned baseline

**Status:** proposal for human decision. Per CLAUDE.md, pinned-record selection is human-only. This report proposes three candidates with evidence and a ranked suggestion; it makes no decision and launches no runs.

**Task definition (plan §0.2):** pin an exact historical record with its released config/log, port to the provisioned GPU count via gradient accumulation, and verify the loss-vs-tokens curve overlays the record's log. Plan guidance: "Prefer a mid-2025 record — recent records are hyper-co-adapted engineering artifacts; older ones are cleaner optimizer testbeds" (`routed-muon-research-plan.md`, §0.2).

**Benchmark target:** ≤ 3.28 validation loss on FineWeb, timed on 8×NVIDIA H100 (`vendor/modded-nanogpt/README.md:3`, record-history table under the heading at `README.md:101`).

---

## Shared context for all three candidates

All three candidates belong to the "record #21 family": record #21 ("Reduced batch size", 01/26/25, `vendor/modded-nanogpt/README.md:131`) fixed the ML content, and the README describes the subsequent entries #22–#25 as systems/timing changes only (faster all-reduce, comm overlap, reduce_scatter, PyTorch upgrade; `README.md:134–137`). This makes the family the natural "mid-2025, cleaner testbed" point: it predates the Aug–Dec 2025 wave that entangled the optimizer itself with the speedrun (sparse attention gates, NorMuon, cautious weight decay w/ LR-tied schedule, batch-size schedules, multi-token prediction — see the techniques list at `README.md:13–40` and records `2025-08-23_SparseAttnGate` through `2025-12-31_*`).

Shared ML configuration (citations into the 2025-07-13 log file, which embeds the full training script — all record `.txt` files in this repo are self-contained: complete source code followed by the training log):

- Model: 12 layers, 6 heads, model_dim 768, vocab padded to 50304 (`records/track_1_short/2025-07-13_UpgradeTorch190/692f80e0-....txt:596`); one attention layer skipped (`...txt:384`).
- Optimizer, as actually instantiated (not the class-signature defaults at `...txt:151`): distributed Muon for hidden matrix params — `Muon(hidden_matrix_params, lr=0.05, momentum=0.95, ..., weight_decay=0.0)` (`...txt:614`) — plus `DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, ...)` (`...txt:613`, class at `...txt:206`) with per-parameter `lr_mul` multipliers set in the model (75 / 27.5 / 5.0; `...txt:405–427`). Candidate C differs on the Adam side: plain `torch.optim.Adam` with per-group lrs 0.22 / 0.6 / 0.04 (`2025-05-24_StableTorch/89d9f224-....txt:515–518`) and `Muon(hidden_matrix_params, lr=0.05, momentum=0.95, ...)` with no weight-decay argument (`...txt:519`; its Muon class has no `weight_decay` parameter, `...txt:158`).
- Tokens/optimizer-step: `world_size * seq_len` with `seq_len = 48*1024` and world_size 8 → 8 × 49,152 = 393,216 tokens/step (derived; `...txt:551`, `...txt:661`).
- Validation: fixed `val_tokens = 10485760` (`...txt:544`).
- FP8 custom matmul ops via `torch._scaled_mm` on `float8_e4m3fn` for the LM head (`...txt:27–80`). **Hardware consequence:** these ops require an FP8-capable GPU (H100-class); a plain A100 cannot execute this code path unmodified. This applies to all three candidates.

Architecture co-adaptations present in all three (inherited from records #14–#21): value embeddings, U-net-pattern skip connections, merged QKV, long-short sliding-window attention with window warmup, FP8 head, logit softcap (README techniques list, `README.md:13–40`). This is the irreducible co-adaptation floor for any 2025 record; the optimizer itself, however, is stock Muon + Adam in all three candidates — no NorMuon, no cautious WD, no batch-size schedule.

### Porting to 1–2 GPUs via gradient accumulation (common to all three)

The training scripts perform exactly one forward/backward per optimizer step and derive tokens/step from `world_size` (`...txt:661`), so at 1–2 GPUs tokens/step would silently shrink 8×/4×. The port must:

1. Add a micro-batch accumulation loop: G micro-steps of 49,152 tokens per rank per optimizer step, with G=8 (1 GPU) or G=4 (2 GPUs), keeping 393,216 tokens/optimizer-step; scale gradients by 1/G before the optimizer step (the record averages across ranks via `ReduceOp.AVG`, `...txt:178`).
2. Preserve data order: `distributed_data_generator` is called with span `world_size * seq_len` — the port should reproduce the record's token-to-step assignment if a tight loss-vs-tokens overlay is wanted; otherwise the overlay is only statistically comparable.
3. Remove/relax the `assert world_size == 8` where present (per-candidate below).
4. Leave all schedules untouched — LR cooldown, momentum warmup, and attention-window warmup are keyed on optimizer-step index, which the port preserves.
5. Accept non-bit-identical numerics (bf16 accumulation order, reduce order, FP8): overlay tolerance is a human-authored criterion in `criteria/nanogpt_tolerance.yaml`.
6. Memory: optimizer state is no longer sharded 8 ways; at this model scale that fits a single 80 GB GPU together with the unchanged 48Ki-token activation footprint (the record already runs 48Ki tokens per GPU).

Estimated single-GPU wall time (naive 8× scaling of recorded train_time, ignoring lost comm overlap; **estimate, not a measurement**): ~8 × 174 s ≈ 23 min/run on 1×H100, plus ~7 min first-run `torch.compile` latency (`README.md:77`). Data prerequisite: `python data/cached_fineweb10B.py 9` (first 900M tokens; `README.md:72`) covers 1770 × 393,216 ≈ 696M train tokens + validation.

---

## Candidate A — Record #25, "Upgrade PyTorch to 2.9.0" (2025-07-13_UpgradeTorch190)

- **Record row:** #25, 2.896 minutes, 07/13/25 (`vendor/modded-nanogpt/README.md:137`).
- **Log + config in-repo:** yes — single self-contained file (full training script + log): `vendor/modded-nanogpt/records/track_1_short/2025-07-13_UpgradeTorch190/692f80e0-5e64-4819-97d4-0dc83b7106b9.txt`. Hyperparameters at lines 540–554 (`num_iterations = 1770`, `cooldown_frac = 0.4`, `seq_len = 48*1024`).
- **Final val loss in log:** 3.2794 at step 1770/1770, train_time 173,789 ms (log tail).
- **Hardware it was set on:** 8×NVIDIA H100 80GB HBM3 (nvidia-smi block in log, line ~765); PyTorch 2.9.0.dev20250713+cu126 (log line 756).
- **Porting:** as shared section; notably this script has **no** `assert world_size == 8` (world_size read from env, `...txt:557–558`), so it is the least intrusive to port.
- **For:** ML-identical to the #21 family but running on the newest pinned torch of the family — most likely to run unmodified on a current CUDA stack; terminal, most-debugged member of the ML-stable family; no optimizer-entangled tricks beyond the family floor.
- **Against:** only one run log in-repo — no in-repo seed distribution to calibrate an overlay tolerance against; single-seed val loss (3.2794) is the only anchor.

## Candidate B — Record #26, "BOS-aligned batching" (2025-07-12_BosAlign)

- **Record row:** #26, 2.863 minutes, 07/13/25 (`vendor/modded-nanogpt/README.md:138`).
- **Log + config in-repo:** yes, unusually rich — 20+ self-contained run logs in `vendor/modded-nanogpt/records/track_1_short/2025-07-12_BosAlign/` (primary cited log `c1fd8a38-bb9f-45c4-8af0-d37f70c993f3.txt`, Hyperparameters at lines 556–569: `num_iterations = 1750`, `cooldown_frac = 0.45`), plus a `README.md` in the record directory documenting a 20-run validation: per-run val losses (mean 3.2791, std 0.0013) and per-run times (mean 173.36 s), with the p-value computation shown.
- **Final val loss in cited log:** 3.2771 at step 1750/1750, train_time 171,743 ms.
- **Hardware it was set on:** 8×NVIDIA H100 80GB HBM3 (nvidia-smi block in log); PyTorch 2.9.0.dev20250713+cu126 (log line 745). The record README notes the official time came from a 07/13/25 retiming "with a refactored version of the code".
- **Porting:** as shared section; has `assert world_size == 8` (`...txt:574`) to relax; additionally the BOS-aligned `distributed_data_generator` selects per-rank starting points jointly across 8 ranks, so preserving the record's exact token order at 1–2 GPUs requires emulating the 8-rank start-point selection — slightly more porting work than A or C.
- **For:** the only mid-2025 record with an in-repo n=20 seed distribution (mean/std for both loss and time) — directly usable by the human to author `criteria/nanogpt_tolerance.yaml` and by WP0.2's 3-seed variance table for context. Otherwise same optimizer cleanliness as A.
- **Against:** adds one ML delta beyond the #21 family (BOS-aligned data loading + 1750 iters + cooldown 0.45 + min-LR factor 0.05, per its README) — one more moving part in the data pipeline, and the exact-token-order port is more fiddly; official timing used refactored code, so the in-repo script and official time are not from the identical artifact.

## Candidate C — Record #21 retimed on stable torch (2025-05-24_StableTorch)

- **Record row:** "21 | 3.014 minutes | 21st record with latest torch | 05/24/25 | ... not a new record, just re-timing #21 with latest torch" (`vendor/modded-nanogpt/README.md:133`; original #21: `README.md:131`).
- **Log + config in-repo:** yes — single self-contained file: `vendor/modded-nanogpt/records/track_1_short/2025-05-24_StableTorch/89d9f224-3b01-4581-966e-358d692335e0.txt`. Hyperparameters at lines 442–457 (`num_iterations = 1770`, `cooldown_frac = 0.4`, `train_seq_len = 48*1024`).
- **Final val loss in log:** 3.2830 at step 1770/1770, train_time 180,832 ms. Note this single retiming run finished above 3.28; the record's statistical validation belongs to the original Jan-2025 #21 runs (`records/track_1_short/2025-01-26_BatchSize/`), and the retime row is explicitly "not a new record". A baseline pinned here inherits that ambiguity about the expected single-run loss.
- **Hardware it was set on:** 8×NVIDIA H100 80GB HBM3 (nvidia-smi block in log); PyTorch 2.8.0.dev20250524+cu126 (log line 643).
- **Porting:** as shared section; has `assert world_size == 8` (`...txt:462`) to relax.
- **For:** the earliest (hence, by the plan's own heuristic, least co-adapted) member of the family that still runs on a modern-ish torch; ML identical to A.
- **Against:** ML-identical to A while being on an older torch nightly and carrying a >3.28 single-run log — it offers no testbed advantage over A beyond a 2-month-earlier timestamp of the systems code.

---

## Comparison table

| | A: 2025-07-13_UpgradeTorch190 | B: 2025-07-12_BosAlign | C: 2025-05-24_StableTorch |
|---|---|---|---|
| Record # / official time | #25 / 2.896 min (`README.md:137`) | #26 / 2.863 min (`README.md:138`) | #21 retime / 3.014 min (`README.md:133`) |
| Date | 07/13/25 | 07/13/25 (dir dated 07/12) | 05/24/25 (ML from 01/26/25) |
| Steps × tokens/step | 1770 × 393,216 (≈696M, derived) | 1750 × 393,216 (≈688M, derived) | 1770 × 393,216 (≈696M, derived) |
| Final val loss in cited log | 3.2794 | 3.2771 | 3.2830 |
| Seed evidence in-repo | 1 log | 20+ logs + n=20 mean/std in record README | 1 log (original #21 logs exist separately) |
| Torch pinned in log | 2.9.0.dev20250713+cu126 | 2.9.0.dev20250713+cu126 | 2.8.0.dev20250524+cu126 |
| `assert world_size == 8` | absent | present (line 574) | present (line 462) |
| ML delta vs #21 family | none | BOS-aligned loader, 1750 iters, cooldown 0.45 | none |
| Port complexity | lowest | highest (data-order emulation) | low |
| Optimizer entanglement | family floor (stock Muon+Adam) | family floor | family floor |
| GPU requirement (FP8 ops) | H100-class | H100-class | H100-class |

## Ranked suggestion (human decides)

1. **A (UpgradeTorch190)** — cleanest combination of testbed simplicity, modern stack, and lowest port friction; its weakness (single log) can be repaired by our own 3-seed variance runs, which WP0.2 requires anyway.
2. **B (BosAlign)** — choose if the in-repo n=20 distribution is judged more valuable than the extra data-loader complexity; it is the strongest option for authoring a defensible tolerance criterion.
3. **C (StableTorch)** — fallback if the torch-2.9 nightly proves hard to reproduce in the container; otherwise dominated by A.

**Adjacent option, explicitly not one of the three:** the optimization track (`vendor/modded-nanogpt/records/track_3_optimization/`, fixed arch/data/batch-size, minimize steps to 3.28, runs on "{1,2,4,8}x-{A100,H100}" per its README Quickstart) is purpose-built as an optimizer benchmark and contains a validated DynMuon entry (#28, 3175 steps, n=25; `records/track_3_optimization/README.md:85`). It is a 2026 artifact and not a "historical main-track record", so it does not satisfy the WP0.2 brief as written — but the human may want it on the radar for WP0.3/Phase-2 cross-checks.

## Numbers audit

Every number above is either read from a cited file/line in `vendor/modded-nanogpt/` or explicitly labeled "derived"/"estimate" with its arithmetic shown. No runs were performed for this report.

---

## DECISION (human, 2026-07-19)

Pinned record: **2025-07-12_BosAlign** (candidate B), selected in-session by
Kasper Nielsen. Rationale accepted: in-repo n=20 seed distribution (mean
3.2791, std 0.0013) is the strongest basis for authoring
`criteria/nanogpt_tolerance.yaml`. WP0.2 port work targets this record;
`world_size == 8` assertions to be relaxed for the 1–2 GPU grad-accumulation
port per the candidates analysis above.
