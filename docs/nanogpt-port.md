# WP0.2 — modded-nanogpt record port: what changed, and what it costs

Port of the **pinned record 2025-07-12_BosAlign** (human decision,
`reports/wp02-record-candidates.md` §DECISION) to a 1–2 GPU testbed.
Code: `src/nanogpt/`. Configs: `configs/wp02_nanogpt_repro*.yaml`.
Analysis: `scripts/analyze_nanogpt.py`. Tests: `tests/test_nanogpt_port.py`.

Why this benchmark exists: our airbench result is a powered *equivalence*
(any effect > 0.042pp excluded at n=100), which disqualifies airbench for
method claims. The literature sweep (`docs/litreview/i-benchmark-headroom.md`)
ranks modded-nanogpt steps-to-3.28 first among cheap testbeds with real
headroom: optimizer-only effects there are 1–3% of steps and are resolvable,
because the record publishes run distributions (σ ≈ 0.0013 loss at target).
**Everything below exists to make sure the 1–3% we might measure is the
method's and not the port's.**

---

## 0. Which record log is authoritative (a correction to the candidates report)

The record directory holds **21** logs. Twenty embed a byte-identical script
(md5 `5ffba04f…`), and their twenty final val losses are an exact multiset
match for the `accs` list in the record's own README — *those twenty runs are
the n=20 distribution (mean 3.2791, std 0.0013) we power against*.

The 21st, `c1fd8a38-…txt`, is the log `reports/wp02-record-candidates.md`
cites as "primary". It is the **07/13 retiming run**, done — in the record
README's own words — "with a refactored version of the code", and it is not
one of the twenty. Its script differs in **ML**, not only plumbing:

| | 20 validation logs (used here) | `c1fd8a38` retiming log |
|---|---|---|
| LR schedule floor | `w*1.0 + (1-w)***0.05**` | `w*1.0 + (1-w)***0.1**` |
| cooldown branch | `w = min((1-x)/cooldown_frac, 1.0)` | `if x < 1-cooldown_frac: 1.0 else …` |
| `assert world_size == 8` | absent | present |
| final val loss | 3.2770–3.2819 (n=20) | 3.2771 (n=1) |

The record README lists "decreased minimum lr schedule factor from 0.1 to
0.05" as one of the three changes that *constitute* this record, so **0.05 is
the record's ML** and the retiming script is the odd one out. The port
therefore reproduces the validation script:

```
vendor/modded-nanogpt/records/track_1_short/2025-07-12_BosAlign/
    0c5449cc-0b01-4ecc-bec3-f46a09741d60.txt      (lines 1–783 = the script)
```

Every `RECORD:<n>` comment in `src/nanogpt/` is a line number in that file.
`tests/test_nanogpt_port.py::test_retiming_script_has_the_other_lr_floor`
pins this finding so it cannot quietly regress.

---

## 1. The accumulation math

The record derives its batch from `world_size` alone (RECORD:692
`distributed_data_generator(..., world_size * args.seq_len, ...)`,
RECORD:579 `seq_len = 48*1024`):

```
tokens / optimizer step = record_world_size * train_seq_len
                        = 8 * 49,152 = 393,216
```

On D devices the port runs G micro-batches per device per optimizer step:

```
G = 8 / D                       tokens/step = D * G * 49,152 = 393,216 (invariant)

  D = 1  ->  G = 8       D = 2  ->  G = 4       D = 4  ->  G = 2      D = 8  ->  G = 1 (the record)
```

D must divide 8; anything else raises (`ConfigError`) rather than rounding.

**Gradient scaling — why 1/G is exact.** Training loss uses
`reduction="sum"` (RECORD:503), so each chunk's backward yields the *sum* over
its 49,152 tokens, and the record averages across the 8 ranks with
`ReduceOp.AVG` (RECORD:178, RECORD:233):

```
g_record = (1/8) · Σ_{c=1..8} g_c
```

Per rank the port accumulates its G chunks and the AVG over D ranks gives
`(1/D) · Σ_{c=1..8} g_c`. Scaling each micro-loss by `1/G` before `.backward()`
yields `(1/(D·G)) · Σ g_c = (1/8) · Σ g_c = g_record`, exactly, because
gradients are linear in the loss and `D·G ≡ 8`. G ∈ {1,2,4,8} is a power of
two, so the rescale is exact in binary floating point — which also keeps the
FP8 head backward (`grad_s = 1/448`, e5m2 quantization) at *identical relative
precision*: dividing by a power of two only shifts exponents.

Verified in `tests/test_nanogpt_port.py`:
`test_micro_loss_scaling_reproduces_the_record_gradient` (the identity on
numbers) and `test_accumulation_factor_reproduces_record_token_batch`
(D ∈ {1,2,4,8}).

**Data order is preserved, not approximated.** The record's BOS-aligned loader
picks *8 chunk starts jointly* per step and advances the file position by the
span they cover — the chunking is a function of the rank count, so running it
at `world_size=1` would produce a *different token stream*. `src/nanogpt/data.py`
therefore always computes the record's 8 chunks and hands each device its G of
them (`chunk = micro·D + rank`). Since the step consumes the sum over all 8,
the assignment cannot affect the gradient.
`test_ranks_and_micro_steps_cover_the_records_chunks_exactly_once` checks, for
D ∈ {1,2,4,8}, that the port's chunks are exactly the D=8 reference chunks and
that the file position advances identically.

Validation is the record's fixed 10,485,760 tokens in fixed 262,144-token
chunks (40 of them), redistributed over D devices — same tokens, same order,
same pieces, same mean.

---

## 2. Complete deviation list

Anything that changes numerics is marked **NOT RECORD-FAITHFUL**; every one of
them flips `metrics.record_faithful` to `false` and is enumerated in
`metrics.deviations` in the results JSON, so no run can hide one.

> **`device_count != 8` is one of them: D != 8 is NEVER record-faithful.**
> The token batch is preserved exactly (the 1/G proof below), but the gradient
> *reduction order* is not — and for the bf16 embedding grads that is a
> genuine precision difference, not a rounding-order nicety (see "Deviations
> that exist whether we like them or not"). Numerics-changing means not
> record-faithful, so only `device_count: 8` can be a reproduction of the
> record; every smaller D is an accumulation *arm* of it.
>
> This was a code defect until 2026-07-20: `NanoGPTConfig.record_faithful` did
> not inspect `device_count`, so a D=1 run reported `record_faithful: true`
> while its own `deviations` dict listed `grad_accumulation` — the flag and the
> deviation list contradicted each other. Fixed in `src/nanogpt/config.py` and
> pinned in both directions by
> `tests/test_nanogpt_port.py::test_record_faithful_requires_the_records_device_count`.
> `results/nanogpt_seed17*.json` predate the fix and carry the old `true`;
> `results/` is append-only, so the correction is stated in
> `reports/wp02-nanogpt-repro.md`, which `scripts/analyze_nanogpt.py`
> regenerates with the recomputed flag.

### Structural (the point of WP0.2; ML-neutral by the argument above)

| # | Change | Where |
|---|---|---|
| T1 | Gradient accumulation, micro-loss × 1/G | `train.py` |
| T2 | `assert world_size == 8` relaxed → replaced by a *check* that D divides 8 and tokens/step is unchanged | `train.py` |
| T3 | Total steps / target loss / seed / device count are config-driven (record: literals) | `config.py` |
| P2 | `scalars` padding uses the **record's** world size (8) so the parameter tensor is the record's 64-entry one at any D (record: process world size) | `model.py` |
| P3 | Block-mask index tensor takes its device from the input (record hardcodes `"cuda"`) | `model.py` |
| P5/T7 | The record's `torch.empty(1, device="cuda").backward()` import-time hack moved into startup so the package imports on a CPU box | `train.py` |
| T6 | The validation script's second, profiler-wrapped 10-step warmup is dropped; one kernel warmup remains, state restored exactly as the record does | `train.py` |
| T5 | Metrics JSON + optional VM-local checkpoint/resume (spot tier) | `train.py` |
| T8 | Process-group bootstrap: rank/world/local-rank default to (0, 1, 0) and a 1-rank run rendezvouses through an in-process `dist.HashStore` — **no `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT` needed for a single-device run.** Under `torchrun` the env is present and behaviour is the record's | `train.py` |
| O1 | `DistAdam.step` writes the parameter through `p.detach()` instead of through the `Parameter` (same storage, same writes) | `optim.py` |
| — | `pin_memory` only requested when CUDA is present | `data.py` |

**O1 in detail — the bug that killed the first record-faithful GPU run.**
RECORD:249 slices a `Parameter` inside `@torch.no_grad()`:

```python
p_slice = p[rank * rank_size:(rank + 1) * rank_size]   # RECORD:249
...
p_slice.mul_(...); p_slice.add_(...)                    # RECORD:246-259
dist.all_gather_into_tensor(p, p_slice, async_op=True)  # RECORD:262
```

`p_slice` is an autograd *differentiable view* with creation-meta
`NO_GRAD_MODE`, and the in-place writes bump the base's version counter.
`DistAdam.step` is `@torch.compile`d (RECORD:223), so the collective at
RECORD:262 is a dynamo graph break; resuming the trace re-wraps `p_slice` as a
graph input, reads `.is_leaf`, hits the view-rebase check and raises

```
torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: A view was created in
no_grad mode and its base or another view of its base has been modified inplace
with grad mode enabled.
```

before step 1 completes. (Handing the collective a base/view pair with mixed
`._base` states additionally trips AOTAutograd's `merge_view_inputs`.) The port
takes `p_view = p.detach()` once and uses it for both the slice and the
collective's output. `p.detach()` shares `p`'s storage, sizes, strides and
version counter, so every write lands in the same memory with the same
arithmetic and `p` observes the same values — **the ML is unchanged**; only the
autograd metadata of two temporary handles changes, and nothing here is
differentiated (`@torch.no_grad()`, `p` is a leaf). It is the record's own idiom
for writing into a parameter's storage (RECORD:632 `dist.broadcast(param.detach(), 0)`,
RECORD:346/372/417 `weight.detach().zero_()`). Regression-tested by
`tests/test_nanogpt_port.py::test_dist_adam_param_slice_is_not_a_no_grad_view`
and `::test_training_loop_end_to_end_on_cpu_with_compile`, both of which fail
with the exact message above if O1 is reverted.

### Behaviour-changing, opt-in, all OFF in the shipped configs

| Flag | Effect | Status |
|---|---|---|
| `precision_mode: bf16` | LM head leaves FP8 (`torch._scaled_mm` / `float8_e4m3fn`) for `F.linear` in bf16. **Required on A100/A6000/L40, which cannot execute the FP8 path at all.** | **NOT RECORD-FAITHFUL** — changes head numerics; a run using it is not a reproduction of the record and must be reported as a separate arm |
| `attention_impl: sdpa` | Dense-mask SDPA instead of FlexAttention block masks (FlexAttention has no CPU backward) | **NOT RECORD-FAITHFUL** — CPU test path only; refuses sequences > 8192 |
| `max_steps` | Truncates the run | **NOT RECORD-FAITHFUL** — smoke runs only |
| `compile: false` | Slower, ML-neutral | reported |
| `fp32_embed_grad_accum` | Embedding grads accumulate across micro-batches in an fp32 master buffer, cast back to bf16 once per step (one rounding instead of G). The §6.1 diagnostic probe; `configs/wp02_nanogpt_fp32embed.yaml` | **NOT RECORD-FAITHFUL** — diagnostic only, never a reproduction arm |

### Deviations that exist whether we like them or not

- **Seeding (T4).** The record seeds *nothing*; its 20 runs differ by
  nondeterministic init/order. We seed from `config.seed` (dev seeds ≥ 1000,
  CLAUDE.md rule 2) so runs are reproducible and arms can be seed-paired. This
  makes our runs a *different* random ensemble from the record's — comparable
  in distribution, never run-for-run.
- **Reduction order.** At D<8 the 8 chunk gradients are summed sequentially in
  `p.grad` instead of being tree-reduced across 8 ranks. In fp32 this is a
  rounding-order difference; for the **bf16 embedding grads** (RECORD:628-630
  casts embeddings to bf16) it is a genuine precision difference — 8 sequential
  bf16 accumulations vs an 8-way AVG. This is the least-controlled numeric
  deviation in the port and is the first suspect if the overlay is off.
- **Wall-clock is not comparable.** The record's 173 s is 8×H100-SXM5 with
  comm/compute overlap. Our timings measure our hardware. **The metric of
  record here is steps-to-val-loss-3.28, never seconds.**
- **Torch version.** The record ran `2.9.0.dev20250524/0713+cu126`; the
  container pins whatever `uv.lock`/`Dockerfile` provide. Kernel-level numeric
  differences are possible and unmeasured.

---

## 3. GPU requirements, commands, runtime and cost

### Hardware

| GPU | FP8 head? | Verdict |
|---|---|---|
| **H100-80G (PCIe or SXM)** | yes | **record-faithful**; the only tier that reproduces the record's numerics |
| A100-80G, L40, RTX-A6000 | no | runnable **only** with `precision_mode: bf16` → not a reproduction |
| 2× of the above | yes/no as above | `device_count: 2`, G=4, same token batch |

Memory: at D=1 the optimizer state is unsharded, but the model is 124M params
with Adam state on embeddings/head only, and activations are one 48Ki-token
micro-batch — the same activation footprint the record has per GPU. 80 GB is
comfortable; the record itself reports its peak in-log.

Hyperstack stock at time of writing (`check_stocks`, CANADA-1):
`H100-80G-PCIe` 1×/2× plentiful (72 / 30), `A100-80G-PCIe-spot` 1× available,
`RTX-A6000-spot` available. **CANADA-1 H100-80G-PCIe 1× is the recommended
flavor** — it is the only listed available option that keeps the FP8 path.

### Commands

```bash
# 1. push + build (human, per the compute boundary)
export RM_VM=ubuntu@<ip>
bash scripts/launch_cloud.sh push
bash scripts/launch_cloud.sh build

# 0/2. data (~2.0 GB, resumable, verified) — ON THE VM, into the data/ mount
#      that launch_cloud.sh excludes from push and maps to /workspace/data
ssh $RM_VM 'cd ~/routed-muon && uv run python scripts/fetch_fineweb.py \
    --config configs/wp02_nanogpt_repro.yaml --data-dir ~/routed-muon/data/fineweb10B'
# (or locally with the same command, without the ssh prefix, for a local GPU box)

# 2a. single 1-GPU reproduction run
bash scripts/launch_cloud.sh run configs/wp02_nanogpt_repro.yaml

# 2b. the 3-dev-seed variance set
bash scripts/launch_cloud.sh sweep configs/wp02_nanogpt_repro_3seed.yaml

# 2b'. NOTE: a 1-GPU run needs NO distributed environment variables (PORT
#      CHANGE T8). Do not export RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/
#      MASTER_PORT and do not wrap it in torchrun — `device_count: 1` builds
#      its own single-rank group through an in-process store.

# 2c. two GPUs: set device_count: 2 in the config, then on the VM
docker run --rm --gpus all -v $PWD/results_out:/workspace/results \
    -v $PWD/data:/workspace/data --entrypoint uv routed-muon \
    run --frozen torchrun --nproc_per_node=2 scripts/run.py \
    configs/wp02_nanogpt_repro.yaml

# 3. sync, cost, ingest, analyse
bash scripts/launch_cloud.sh pull
bash scripts/launch_cloud.sh fill-cost <usd> cloud_staging/nanogpt_*.json
bash scripts/launch_cloud.sh ingest
uv run python scripts/analyze_nanogpt.py --results results/nanogpt_seed10*.json \
    --out-md reports/wp02-nanogpt-repro.md --out-png reports/figures/wp02-overlay.png
```

Data lives in the VM-local `data/` mount that `launch_cloud.sh` already
excludes from `push` and maps into the container, so it survives re-pushes and
is fetched once per VM.

### Runtime estimates (ESTIMATES — no GPU run has been made)

Basis: the record's own log, 1750 steps in **171.7 s on 8×H100-SXM5**
≈ 99 ms/step ≈ **792 GPU-ms/step**. Serializing onto one GPU removes comm but
also removes overlap, so the 1-GPU step is ~8× the record's wall step, scaled
by the device's throughput ratio to H100-SXM5. Added on top: first-run
`torch.compile` (~7–10 min per the vendor README) and 15 validation passes
(157M forward-only tokens ≈ 8% of training cost).

| Config | GPU-throughput assumption | est. train time | + compile | **est. total/run** | uncertainty |
|---|---|---|---|---|---|
| 1× H100-80G-PCIe, fp8 (record-faithful) | ~0.75× SXM5 | ~31 min | ~8 min | **~40 min** | ±30% |
| 2× H100-80G-PCIe, fp8 | ~1.8× scaling | ~17 min | ~8 min | **~25 min** | ±35% |
| 1× A100-80G-PCIe, bf16 head (not faithful) | ~0.45× H100 + no FP8 | ~75 min | ~10 min | **~1.5 h** | ±40% |

These are the numbers to replace with a measurement after the first run; the
first run should be launched with `max_steps: 50` on a dev seed to get a real
ms/step before committing to the full 1750.

### Cost

The project's only price rate verified in-repo is **$0.4067/h (RTX-A6000
spot)** (`docs/wp22-run-plan.md`). Hyperstack's A100/H100 rates are **not
exposed by the MCP tooling and are not asserted here** — the human reads them
at provisioning and fills `cost_usd` per CLAUDE.md rule 5. Cost is simply

```
cost/run = est. hours (above) × VM $/hr
```

| Rate ($/h) | 1×H100 run (0.67 h) | 3-seed set | 1×A100 run (1.5 h) | 3-seed set |
|---|---|---|---|---|
| 1.00 | $0.67 | $2.01 | $1.50 | $4.50 |
| 2.00 | $1.34 | $4.02 | $3.00 | $9.00 |
| 3.00 | $2.01 | $6.03 | $4.50 | $13.50 |

For scale: total project cloud spend to date is $9.72, so a 3-seed nanogpt set
is a step change in this project's burn rate and should be launched only after
a `max_steps: 50` timing probe. Spot preemption is covered by
`checkpoint.every_steps: 250` (VM-local, never synced).

---

## 4. Data footprint

- Source: `kjj0/fineweb10B-gpt2` (the same shards the vendored
  `data/cached_fineweb10B.py` fetches), GPT-2 tokens, uint16, 100M tokens +
  1 KiB header per shard = **0.2 GB/shard**.
- Budget: 1750 steps × 393,216 + 10 warmup steps = **692M train tokens**;
  ×1.15 for BOS-alignment span overhead and per-shard tail waste, +1 shard
  headroom → **9 train shards + 1 validation shard = 10 shards = 2.0 GB**.
  (The vendor README's own guidance for a run this long is `cached_fineweb10B.py 9`
  — the same answer.)
- `scripts/fetch_fineweb.py` fetches only those, resumably (`.part` + HTTP
  Range), and verifies each shard on every invocation: header magic
  `20240520` / version 1 / `num_tokens`, file size == `1024 + 2·num_tokens`,
  and a sha256 recorded in `data/fineweb10B/shard_manifest.json` on first
  download and re-checked afterwards.

---

## 5. Metrics written (results JSON `metrics`)

`val_curve` (step, tokens, val_loss, train_time_ms — the record's own trace
shape), `final_val_loss`, **`steps_to_target_loss`** (the metric of record:
val loss ≤ 3.28, linearly interpolated between the 125-step validation
points), `tokens_to_target_loss`, `train_time_s`, `tokens_per_step`,
`accumulation_factor`, `device_count`, `precision_mode`, `attention_impl`,
`record_faithful`, `deviations`, `peak_memory_mib`, and the full resolved
`nanogpt_config`. Provenance (git SHA, seed, gpu_type, wall time, cost) comes
from `scripts/run.py` + `src/results_io.py` as for every other experiment.

Interpolation note: the record's own trace is sampled every 125 steps, so
steps-to-target carries up to ~125 steps of sampling uncertainty **for both
sides equally**. For a comparison that needs finer resolution, lower
`val_loss_every` — but then it is no longer the record's eval cadence, and
that is a deviation to declare. The record's own n=20 logs interpolate to
1740.5 ± 4.5 steps.

---

## 6. Risks

1. **bf16 embedding-gradient accumulation** (§2) is the one numeric deviation
   the accumulation proof does not cover. If the overlay is off, test it first
   by running D=8-equivalent chunk counts with fp32 grad accumulation on the
   embeddings as a probe.

   **This risk has now materialised and the probe is built.** The seed-1701
   D=1 run finished +0.0112 above the record's n=20 ensemble mean and +0.0084
   above the record's observed *maximum*, with the deficit concentrated in the
   LR cooldown (10x denser per unit of loss removed than in the stable phase;
   all 6 cooldown segments underperform). The probe is
   `configs/wp02_nanogpt_fp32embed.yaml` — `fp32_embed_grad_accum: true`, seed
   1701, D=1, everything else identical to `configs/wp02_nanogpt_repro.yaml`.

   Pre-registered read (fixed before the run; see
   `reports/wp02-nanogpt-repro.md`):

   | final deficit vs the record's n=20 mean | conclusion |
   |---|---|
   | **<= +0.006** | bf16-accumulation suspect **confirmed** |
   | **unchanged at ~+0.011** | suspect **excluded**; residual is torch-version / kernel / hardware |

   An outcome between the two is partial attribution and is reported as such.
2. **Partial GPU verification.** CPU-verified: model forward/backward, the full
   training loop end-to-end (stubbed single-rank collectives), the same loop
   **with `compile: true` and `torch._dynamo.config.suppress_errors` forced
   False** (`test_training_loop_end_to_end_on_cpu_with_compile` — this is what
   catches dynamo/autograd-aliasing bugs of the O1 class), single-rank
   process-group bootstrap with an empty environment, accumulation arithmetic,
   data chunking, config parsing, the record-trace parser.
   **Still GPU-only-untested, in the order they would bite:**
   - the FP8 head — `torch.ops.routed_muon_nanogpt.mm` / `mm_backward`,
     `torch._scaled_mm`, `float8_e4m3fn` / `e5m2` quantization, and the custom
     op's `register_autograd`/`setup_context` wiring (CPU runs `precision_mode:
     bf16`, which takes the `F.linear` branch and never enters the custom op);
   - FlexAttention — `BlockMask.from_kv_blocks`, the `document_causal` mask_mod,
     and the long/short SWA masks (CPU runs the `sdpa` dense-mask fallback,
     which is the port's own code, not the record's kernel);
   - inductor's **CUDA** codegen for the compiled model and for
     `DistAdam.step` / `zeropower_via_newtonschulz5` (CPU exercises dynamo +
     AOTAutograd, i.e. the tracing/aliasing layer where O1 lived, but a
     different inductor backend);
   - NCCL — every collective is stubbed on CPU, so `reduce_scatter` /
     `reduce_scatter_tensor` / `all_gather` / `all_gather_into_tensor` with
     `ReduceOp.AVG`, and in particular the **input/output aliasing** at
     `world_size == 1` (`all_gather_into_tensor(p_view, p_slice)` where the two
     are the same memory), are unverified against a real backend;
   - `torch.cuda.max_memory_allocated`, and the peak-memory claim in §3.
3. **Torch-version drift** vs the record's 2025 nightly (§2).
4. **A100-only availability** would force `precision_mode: bf16`, i.e. no
   record-faithful reproduction at all — that is a WP0.2 finding to report to
   the human, not something to paper over.
5. **Runtime estimates are estimates** (±30–40%); the `max_steps: 50` probe
   converts them to measurements for ~$0.10.
6. **Interpolated steps-to-target** at 125-step cadence may be too coarse to
   resolve a 1% effect (≈17 steps) without either a finer eval cadence
   (declared deviation) or more seeds. This is the key design question for
   WP3.x and should be settled with the seed-variance number from the 3-seed
   set, via `scripts/analyze_nanogpt.py`'s power table.
