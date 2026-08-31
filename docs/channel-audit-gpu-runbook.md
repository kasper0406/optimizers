# Channel audit — GPU stage runbook (local RTX 5090)

Operator instructions for the 15-run frozen-tier re-measurement described in
`configs/dev/instrumented_airbench_demod.yaml`. Everything here runs on the
**local** GPU box (this is not a Hyperstack workload — see
`docs/hyperstack-runbook.md` for the cloud path).

The agent authored the configs and verified them on CPU. **Launching is the
human's call** (CLAUDE.md ground rule 4), and nothing in this document
evaluates a gate.

> ## Hard stop before any of this runs
>
> `reports/channel-audit-preregistration.md` pre-registers this program's
> Phase B and carries **`Status: DRAFT — not registered`**, with an empty
> threshold-freeze table in its Appendix (23 rows after its revision R1, no
> frozen values) and the standing instruction that *no Phase B run may launch
> while the file carries DRAFT status*.
>
> Freezing that table, flipping the status line and committing before the
> first run is what makes Phase B confirmatory. It is a **human** act
> (CLAUDE.md ground rules 1 and 3). Phase A is already peeked, so launching
> against an unfrozen table spends the only unpeeked surface this program
> has.
>
> The gate is mechanical — `scripts/preflight_channel_audit.py` (§2e) exits
> non-zero while the status line, the freeze table, the WP1.2 precondition,
> or config/pre-registration agreement is unresolved. **Run it; do not read
> this document's prose as the check.**
>
> The 15 runs described here *are* the 15 runs that document registers
> (§3 of the pre-registration, checked cell by cell by the same script). Its
> one registered scope limit is that the lr ladder stops at 0.48 while the
> Phase A DC excess peaks at 0.96; extending the grid is Appendix row 19 and
> is a human decision to be taken **before** any Phase B sidecar is read.
>
> **Revision R1 of the pre-registration (pre-launch, no Phase B sidecar in
> existence) changed what will be measured.** Ten defects found by internal
> review were repaired there: P1 now reads a same-probe alt-vs-DC *band
> contrast* rather than a raw AR(1)-calibrated ratio, P3's tau is mean-pooled
> and null-referenced and has a third (UNDECIDED) branch, P1's probe pool is
> the 9-run B = 2000 core rather than all 15 runs, and a new kill clause K6 is
> registered. The **run set** is untouched — the 15 runs below are still
> exactly the 15 that document registers, which is what check 4 verifies — but
> the analysis is not the one an earlier reading of this runbook described.
> §0 of that document carries the full repair record, including its own
> account of having been rewritten to match these configs.
>
> This runbook describes how to execute the run set that exists; it does not
> amend the pre-registration.

---

## 0. What is being run and why it is the confirmatory surface

`reports/frozen-probes.md` reports 864 frozen probes from 9 runs, every one of
them with ESS > n (pooled min 217.5 at n = 200, median 389.9, zero Newey-West
floorings). A follow-up **exploratory, peeked** analysis of the 218
tracked-direction sidecars already on disk fitted that to an AR(1) with
phi ~ -0.34, lr-invariant, whose implied inflation (1 - phi)/(1 + phi) = 2.03
matches the measured ESS/n = 1.95; under that null the demodulated
(alternating) channel sits at the null (median NW |t| ~ 0.75-0.85) while the DC
channel carries a monotone-in-lr excess (median |t| up to 3.77 at lr = 0.96)
confined almost entirely to the momentum-anchored `top` directions.

That analysis cannot confirm itself: it was run on data that had already been
looked at, its segment burn-in >= 5 is load-bearing (without it rho_2 reads
~-0.01 and the AR(1) looks wrong), and it needs a -1/n mean-subtraction bias
correction on finite refresh segments. The **frozen** tier is the confirmatory
surface because frozen probes are drawn once from the instrumentation seed and
are never refreshed, re-orthogonalised or reset, so they carry no
momentum-anchoring and no selection.

Two facts to carry into the analysis, both verified against the code:

- **The re-run's frozen probes are a fresh realisation, not a superset.**
  `torch.randn(m, k3)` fills row-major, so the k3 = 64 draw does not contain
  the k3 = 16 columns. Comparison with the published 864 probes is
  distributional only.
- **So is the trajectory: the original 9 runs are A6000 runs.** Every one of
  them carries `gpu_type: "NVIDIA RTX A6000"` (wall 98.0-108.4 s) and this set
  runs on the 5090 under an fp16 recipe, so identical seeds do not reproduce
  those iterates bitwise. Fixing the seeds, the config and `compile: false`
  preserves the design and the sampling, not the trajectory — a second,
  independent reason no comparison here may be read run-for-run.
- **The original 9 sidecars no longer exist on disk** (only their results
  JSONs survive; the `results/smoothness_sidecars/` staging dir is gone). The
  reference numbers are the summary blocks in `reports/frozen-probes.json`
  (`pooled`, `per_matrix`, `tracked_tier_final_abs_t`), not raw series.

The tracked tier is a free consistency check: the frozen bank draws from a
**separate** RNG stream (`FROZEN_SEED_OFFSET = 7919` plus a per-matrix stable
hash), so raising k3 from 16 to 64 cannot perturb the tracked subspace's
random draws.

## 1. The run set (15 runs, 5760 frozen probes)

| Sweep config | Variants | Seeds | Runs | B | epochs | steps | lr |
|---|---|---|---|---|---|---|---|
| `configs/dev/instrumented_airbench_demod.yaml` | 1 | 1300, 1301, 1302 | 3 | 2000 | 8 | 200 | 0.24 (stock) |
| `configs/dev/instrumented_airbench_demod_lr_ladder.yaml` | 3 (`lr` 0.12/0.24/0.48) | 1310, 1311 | 6 | 2000 | 8 | 200 | ladder |
| `configs/dev/instrumented_airbench_demod_b500.yaml` | 1 | 1320, 1321 | 2 | 500 | 2 | 200 | 0.24 |
| `configs/dev/instrumented_airbench_demod_b2000.yaml` | 1 | 1320, 1321 | 2 | 2000 | 8 | 200 | 0.24 |
| `configs/dev/instrumented_airbench_demod_b8000.yaml` | 1 | 1320, 1321 | 2 | 8000 | 32 | 192 | 0.24 |

Rows 1-2 reproduce the exact 9-run frozen tier behind
`reports/frozen-probes.json` (seeds and lr rungs read back from those runs'
results JSONs — the original ladder was {0.12, 0.24, 0.48} and never included
0.96). Rows 3-5 are the step-matched batch rider.

Instrumentation is identical in all five configs: `hvp: false`,
`smoothness.enabled: false`, `frozen_probes: {enabled: true, k3: 64,
max_lag: 32, decimate: 1}`, everything else as in the original
`instrumented_airbench_smoothness.yaml` (k1/k2 16/16, t_refresh 50,
snapshot_every 5, instrumentation seed 4242, classifier block,
`recipe.compile: false`).

**`max_lag: 32` does not widen the Newey-West window.** The automatic
Bartlett bandwidth is `L = min(max_lag, floor(4 (n/100)^(2/9)), n - 2)`, which
is **L = 4** at both n = 192 and n = 200. The logged `t_nw` / `ess` are
therefore unchanged by that setting, and neither are the sidecars:
`FrozenProbeBank.to_log()` exports `k3`, `max_lag`, `decimate`,
`n_observations`, `lag_truncation` and the L = 4 statistics — no lagged
cross-sum at any depth. The lag ladder out to lag 32 must be computed offline
from the raw per-step `s` series, which `decimate: 1` preserves. Do not report
these `t_nw` values as if they used a 32-lag window.

## 2. Preflight (all five checks, before anything is launched)

```bash
cd /home/knielsen/code/ml/optimizers

# (a) The GPU must be FREE. The audit needs the whole card; another process
#     holding ~30/32 GB will OOM the B=8000 rung. Expect ~0 MiB used.
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# (b) The venv must resolve torch 2.13.0+cu130 and must NOT be re-synced.
#     --no-sync is mandatory: a sync can replace the cu130 wheel.
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.13.0+cu130 True NVIDIA GeForce RTX 5090

# (c) Launch precondition for experiment 'airbench_instrumented'
#     (scripts/run.py refuses to launch without it).
test -f criteria/phase1_preregistration.md && echo "prereg OK"

# (d) Free disk for the sidecars (see §6): at least 1 GB headroom.
df -h .

# (e) The launch gate. Exits 0 only if criteria/phase1_preregistration.md
#     exists, the pre-registration reads REGISTERED, every row of its
#     Appendix threshold-freeze table carries a frozen value, and the five
#     configs still expand to the run set section 3 registers. It exits 1 and
#     names the blocker otherwise; nothing below may run until it is 0.
uv run --no-sync python scripts/preflight_channel_audit.py --verbose
```

Checks (a)-(d) are eyeball checks; (e) is the one that fails closed. As of
writing it exits 1 on `prereg_status` and `threshold_freeze_table` — both
cleared only by the human freezing the thresholds — while
`run_set_matches_prereg` and `instrumentation_matches` already pass. The
script reads; it never edits either document.

## 3. Materialise the sweeps

Already done by the agent; the trees exist under `sweeps/` (gitignored).
`scripts/sweep.py` **refuses to overwrite an existing `manifest.json`**, so if
you want to regenerate, delete the directory first:

```bash
rm -rf sweeps/instrumented_airbench_demod \
       sweeps/instrumented_airbench_demod_lr_ladder \
       sweeps/instrumented_airbench_demod_b500 \
       sweeps/instrumented_airbench_demod_b2000 \
       sweeps/instrumented_airbench_demod_b8000

for c in instrumented_airbench_demod \
         instrumented_airbench_demod_lr_ladder \
         instrumented_airbench_demod_b500 \
         instrumented_airbench_demod_b2000 \
         instrumented_airbench_demod_b8000; do
    uv run --no-sync python scripts/sweep.py "configs/dev/$c.yaml" --dry-run   # plan only
    uv run --no-sync python scripts/sweep.py "configs/dev/$c.yaml"             # writes run_all.sh
done
```

`--dry-run` prints the expansion and writes **nothing**; the second call is
what materialises the per-variant configs, `manifest.json` and `run_all.sh`.
Neither call trains.

## 4. Launch

**`UV_NO_SYNC=1` is not optional.** `scripts/sweep.py` emits
`uv run python scripts/run.py ...` with no `--no-sync` flag (hardcoded in
`run_command()`), so the environment variable is the only way to keep
`run_all.sh` from potentially re-syncing the venv. Export it in the shell that
runs the scripts.

```bash
cd /home/knielsen/code/ml/optimizers
export UV_NO_SYNC=1

# Pilot: ONE run first, timed. Do not launch the rest until this lands.
time uv run --no-sync python scripts/run.py \
    sweeps/instrumented_airbench_demod/instrumented_airbench_demod.yaml --seed 1300
```

Sanity-check the pilot before continuing:

- wall time **under ~150 s** (see §5; if it is far over, the k3 = 64 frozen
  tier is costing more than budgeted — stop and report rather than running 14
  more);
- one `results/airbench_instrumented_seed1300_*.json` and one matching
  `*.instrumentation.json` of roughly the expected size (§6: ~12.5 MB);
- `metrics.instrumentation_sidecar` present in the results JSON.

Then the remaining 14 runs. The base sweep's own `run_all.sh` is deliberately
not invoked (it would repeat the pilot's seed 1300); its other two seeds are
launched explicitly at the end, so the sequence below plus the pilot is
exactly 15 runs with no duplicate:

```bash
tmux new -s channel-audit
export UV_NO_SYNC=1
bash sweeps/instrumented_airbench_demod_lr_ladder/run_all.sh
bash sweeps/instrumented_airbench_demod_b500/run_all.sh
bash sweeps/instrumented_airbench_demod_b2000/run_all.sh
bash sweeps/instrumented_airbench_demod_b8000/run_all.sh
uv run --no-sync python scripts/run.py \
    sweeps/instrumented_airbench_demod/instrumented_airbench_demod.yaml --seed 1301
uv run --no-sync python scripts/run.py \
    sweeps/instrumented_airbench_demod/instrumented_airbench_demod.yaml --seed 1302
```

Each `run_all.sh` is `set -euo pipefail`, so a failing run stops that script.

## 5. Expected wall time — budget ~35 min total

Measured anchors from `results/` (all RTX 5090 unless noted), for the same
harness at 200 steps:

| Configuration | GPU | Wall time | n |
|---|---|---|---|
| B=2000, 200 steps, hvp **on** + smoothness **on**, k3=0 | 5090 | 41.3 s (median) | 51 |
| B=8000, 192 steps, hvp off + smoothness on, k3=0 | 5090 | 31.3 s (median) | 16 |
| B=8000, 48 steps, hvp off + smoothness on, k3=0 | 5090 | 11.9 s (median) | 16 |
| B=2000, 200 steps, hvp on + smoothness on, **k3=16** (the original frozen tier) | **A6000** | 98.3 s (median) | 9 |

This run set drops both the HVP double-backwards and the smoothness passes
(the two most expensive probes) and raises k3 from 16 to 64. The frozen tier
adds no forward or backward pass — it is O(k3) projections per matrix per step
plus a host-side accumulator — so the per-run cost should sit **below** the
41 s hvp+smoothness anchor plus the k3 = 64 projection overhead.

**Budget: ~2 min per run, ~35 min for all 15**, including CIFAR-10 load and
whitening at each launch and the sidecar writes. That is a deliberately loose
envelope: even the A6000 originals, which carried HVP *and* smoothness *and*
frozen probes, took 98-108 s. The last row is on a different GPU and must not
be compared to the 5090 rows directly; it is quoted only as an upper bound on
what the frozen tier can cost.

**Stop conditions:** a pilot run over ~150 s, or any run over ~5 min. Report
rather than improvise (CLAUDE.md ground rule 6).

**Correction to a cost figure quoted elsewhere.** §3 of
`reports/channel-audit-preregistration.md` attributes "98 s/run with HVP +
smoothness + frozen on" and "18 s/run with the probes off" to the RTX 5090.
Both are **RTX A6000** measurements: the 98 s figure is the 9 program-#4
frozen runs (`gpu_type: "NVIDIA RTX A6000"` in their results JSONs, medians
98.0-108.4 s) and the ~17.6 s figure is the WP1.2 A6000 batch. The
corresponding 5090 number for the same recipe with HVP + smoothness on and
frozen off is 41.3 s (n = 51). The prereg's 0.5 GPU-h all-in budget is
unaffected — it is loose either way — but the per-run anchors should not be
read as 5090 numbers.

## 6. Expected sidecar sizes

Sidecars are written by `src.instrument.schema.write_sidecar` with
`indent=2, sort_keys=True` (which costs ~1.93x the compact JSON size). Per
run, at k1=k2=16 over 6 Muon-managed filter matrices:

| Component | 200-step run | 192-step run (B=8000) |
|---|---|---|
| tracked tier (32 directions x 6 matrices) | 6.69 MB | ~6.4 MB |
| frozen tier, k3=64 x 6 matrices (simulated exactly) | 5.79 MB | 5.54 MB |
| smoothness block | absent (disabled) | absent |
| **total per sidecar** | **~12.5 MB** | **~12.0 MB** |

Basis: an existing 200-step sidecar with the identical tracked settings is
6.84 MB on disk including a 103 kB smoothness block and 44 kB of `lambda_hvp`,
both of which this set drops (6.69 MB without them); the frozen-tier figure is
a real `FrozenProbeBank(...).to_log()` at k3=64, max_lag=32, decimate=1,
snapshot_every=5 over 6 matrices, **spliced into that sidecar and re-dumped**
with `indent=2, sort_keys=True`. Sizing the frozen block standalone
under-counts it by ~1 MB: nested at `matrices.<name>.frozen_probes` it carries
6 more spaces of indent on ~160k lines.

**Total for 15 runs: ~186 MB of sidecars** (13 runs at 200 steps, 2 at 192)
plus 15 small results JSONs.
Sidecars are gitignored at the `results/` top level
(`results/*.instrumentation.json`) and are never committed.

Red flags: a sidecar under ~9 MB (frozen block missing or decimated), or
`frozen_probes_enabled: false` at the top level.

## 7. Post-run verification

```bash
cd /home/knielsen/code/ml/optimizers

# 15 new results JSONs from this run set, and their sidecars
uv run --no-sync python - <<'PY'
import glob, json, os
runs = []
for p in sorted(glob.glob('results/airbench_instrumented_*.json')):
    if p.endswith('.instrumentation.json'):
        continue
    d = json.load(open(p))
    if not d['config']['path'].startswith('sweeps/instrumented_airbench_demod'):
        continue
    side = os.path.join('results', d['metrics']['instrumentation_sidecar'])
    runs.append((os.path.basename(p), d['seed'], round(d['wall_time_s'], 1),
                 d['gpu_type'], round(os.path.getsize(side) / 1e6, 2)))
print(f'{len(runs)} runs')
for r in runs:
    print(*r)
PY

# schema validation + frozen-tier shape of every new sidecar
uv run --no-sync python - <<'PY'
import glob, json, os, sys
sys.path.insert(0, '.')
from src.instrument.schema import load_instrumentation
for p in sorted(glob.glob('results/airbench_instrumented_*.json')):
    if p.endswith('.instrumentation.json'):
        continue
    d = json.load(open(p))
    if not d['config']['path'].startswith('sweeps/instrumented_airbench_demod'):
        continue
    log = load_instrumentation(os.path.join('results', d['metrics']['instrumentation_sidecar']))
    fz = [m['frozen_probes'] for m in log['matrices'].values()]
    assert log['frozen_probes_enabled'] and not log['hvp_enabled'] and 'smoothness' not in log
    assert all(b['k3'] == 64 and b['max_lag'] == 32 and b['decimate'] == 1 for b in fz)
    print(os.path.basename(p), 'matrices', len(fz),
          'probes', sum(b['k3'] for b in fz),
          'n_obs', sorted({b['n_observations'] for b in fz}),
          'NW L', sorted({b['lag_truncation'] for b in fz}))
PY
```

Expected: 15 runs, 6 matrices and 384 probes each, `n_obs` 200 (192 for the
B=8000 rung), `NW L` 4 everywhere.

## 8. Staging for analysis

`scripts/analyze_frozen_probes.py` globs **every** `*.instrumentation.json` in
the directory it is given, and `results/` already holds 218 sidecars from
other programs. Stage the new ones into a separate directory first. Use a path
under `logs/` — it is gitignored operational state, whereas a new subdirectory
of `results/` would *not* be covered by the `results/*.instrumentation.json`
ignore rule and would dirty the tree.

```bash
mkdir -p logs/channel-audit-sidecars
uv run --no-sync python - <<'PY'
import glob, json, os
dst = 'logs/channel-audit-sidecars'
for p in sorted(glob.glob('results/airbench_instrumented_*.json')):
    if p.endswith('.instrumentation.json'):
        continue
    d = json.load(open(p))
    if not d['config']['path'].startswith('sweeps/instrumented_airbench_demod'):
        continue
    name = d['metrics']['instrumentation_sidecar']
    link = os.path.join(dst, name)
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join('results', name)), link)
print(len(os.listdir(dst)), 'sidecars staged in', dst)
PY

uv run --no-sync python scripts/analyze_frozen_probes.py \
    --sidecars logs/channel-audit-sidecars/ \
    --out-md reports/frozen-probes-channel-audit.md \
    --out-json reports/frozen-probes-channel-audit.json
```

Write the re-measurement to **new** report paths. `reports/frozen-probes.md` /
`.json` are the published record of the 9-run study and must not be
overwritten.

## 9. Standing constraints for this stage

- Descriptive output only. No pass/fail claim, no gate evaluation, no
  threshold introduced or adjusted (CLAUDE.md ground rules 1 and 3).
- Dev seeds only (1300-1302, 1310-1311, 1320-1321); nothing here is a
  comparison-table entry, and `probe_overrides` is the sanctioned Gate-1 A4
  deviation for the lr ladder.
- `results/` is append-only. Never edit or delete an existing results JSON or
  sidecar.
- The batch rider is **step-matched, not sample-budget-matched**, and three
  recipe quantities move with B (weight decay `2e-6 * B`, the whiten-bias
  warmup `min(ceil(3 * 50000//B), steps)`, and the number of examples seen).
  The rider is read on the autocovariance axis only, never as an accuracy
  ladder. The per-config headers list these confounds in full.
