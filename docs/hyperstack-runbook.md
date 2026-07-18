# Hyperstack runbook (human-executed cloud path)

How cloud runs happen for this project. The compute boundary (CLAUDE.md ground
rule 4): the agent writes code, configs, and these scripts; **the human**
provisions VMs, launches cloud runs, syncs results back, and fills in costs.
`scripts/launch_cloud.sh` automates everything up to but **not including**
provisioning.

## 0. Prerequisites

- A Hyperstack account with an SSH keypair registered.
- Local repo in the state you want to run (commit first — results record the
  git SHA, and a dirty tree is flagged `git_dirty` in every results JSON).
- `uv`, `ssh`, `rsync` locally.

## 1. Provision the VM (human-only)

Via the Hyperstack console (or MCP tooling). Sizing per the research plan
"Hardware" section:

| Workload | Flavor class | Notes |
|---|---|---|
| WP0.1 / WP2.x airbench sweeps | A6000 / L40-class, 1 GPU | runs are seconds each; one cheap VM left up through the whole seed sweep is the economical shape |
| WP0.2 / WP1.2 / WP3.x nanogpt | A100 / H100, 1 GPU (or 2-GPU DDP VM) | spot/preemptible is the cheap tier — see checkpoint-resume below |

- Pick an Ubuntu + NVIDIA docker image (Docker and the NVIDIA container
  toolkit preinstalled).
- Open inbound SSH (port 22) to your IP.
- Note the public IP; everything below uses `RM_VM=user@ip`
  (typically `ubuntu@<ip>`).

**GPU-type discipline (plan §0.1):** all runs inside a single comparison table
must use the same GPU type. If a sweep spans days, re-provision the *same
flavor*. `scripts/aggregate.py` enforces this and will refuse mixed-GPU
tables.

## 2. Push code and build the image

```bash
export RM_VM=ubuntu@<ip>
bash scripts/launch_cloud.sh push     # rsync repo -> VM, record local git SHA
bash scripts/launch_cloud.sh build    # docker build on the VM (tags RM_GIT_SHA)
```

`push` excludes `.git`, `results/`, and staging dirs; it writes the local
commit SHA to `GIT_SHA` on the VM so container runs record provenance via the
`RM_GIT_SHA` env var (see Dockerfile / `src/results_io.git_provenance`).

## 3. Run experiments

Single config (args after the config are forwarded to `scripts/run.py`):

```bash
bash scripts/launch_cloud.sh run configs/smoke.yaml --seed 1000
```

Sweep (the normal shape for WP0.1's 100-seed eval sweep):

```bash
bash scripts/launch_cloud.sh sweep configs/wp01_airbench_eval.yaml
```

Seed policy resolution happens **inside `scripts/sweep.py` at launch time**:
`seeds: eval` expands to seeds 0–99 passed via `run.py --seed N`; `seeds: dev`
expands to the documented dev range (1000, 1001, …). Literal eval seeds never
exist in any config file, and `sweep.py` refuses configs that contain them.

Results JSONs land in `~/routed-muon/results_out/` on the VM (mounted over the
container's `/workspace/results`). Runs write metrics JSONs only — checkpoints
never sync (see below).

For long runs, launch inside `tmux` on the VM so an SSH drop doesn't kill the
sweep:

```bash
ssh $RM_VM
tmux new -s sweep
# then run the docker command that `launch_cloud.sh sweep` prints/executes
```

## 4. Sync results back, fill costs, ingest

```bash
bash scripts/launch_cloud.sh pull    # VM results_out/ -> cloud_staging/ (local)

# Read the run's cost from Hyperstack billing (console or the billing API),
# then stamp it on the staged files (per-run share of the VM bill):
bash scripts/launch_cloud.sh fill-cost 1.84 cloud_staging/airbench_seed*.json

bash scripts/launch_cloud.sh ingest  # staged -> results/ (validated, append-only)
```

Why the staging hop: `results/` is append-only — files are never edited once
they are in there, and `cost_usd` must be human-filled for cloud runs
(CLAUDE.md rule 5). So costs get stamped on the *staged* copies, and `ingest`
refuses to move any cloud result whose `cost_usd` is still null, refuses to
overwrite existing files, and re-validates every JSON against the schema in
`src/results_io.py`.

## 5. Tear down (human-only)

Delete (or hibernate, if returning within hours) the VM once results are
ingested. VMs are disposable by design: everything needed to reproduce a run
is the pinned image + config + seed; everything worth keeping is in
`results/`.

## Checkpoint–resume

- **airbench (WP0.1, WP2.x):** runs take seconds each. No resume machinery —
  if a spot VM dies mid-sweep, `launch_cloud.sh sweep` is simply re-run;
  already-synced result files are skipped at `pull` time (`--ignore-existing`)
  and `ingest` never overwrites, so completed seeds are not lost. Re-executed
  seeds produce new timestamped files; keep the earlier one per (config, seed)
  since `aggregate.py` rejects duplicate seeds within a table.
- **nanogpt (WP0.2+):** runs are hours — resume is mandatory on the spot tier
  (plan "Hardware": every experiment must be resumable from checkpoint).
  Resume hooks belong in the nanogpt experiment runner registered in
  `EXPERIMENT_REGISTRY` in `scripts/run.py` (WP0.2 work): the config gets a
  `checkpoint:` block (directory + save interval in tokens), the runner
  checkpoints model/optimizer/dataloader state to **VM-local disk**
  (`~/routed-muon/checkpoints/`, mounted into the container) and, on start,
  resumes from the newest checkpoint matching (config sha, seed). Only metrics
  JSONs sync to durable storage — checkpoints stay on the VM and die with it;
  a preempted VM of the same flavor is re-provisioned, `push`/`build` re-run,
  and the volume re-attached or the run resumed from the last synced
  checkpoint if a persistent volume was mounted at `checkpoints/`.

## Troubleshooting

- `run`/`sweep` fails with a CUDA error: check `nvidia-smi` on the VM and that
  the image built from the CUDA base (`docker run --rm --gpus all routed-muon
  configs/smoke.yaml --seed 1000` is the smoke test; it runs on GPU-less
  machines too, on CPU).
- `ingest` refuses a file: the error names the schema problem or the missing
  cost. Fix on the staged copy — never in `results/`.
- `aggregate.py` refuses mixed GPU types: the sweep ran on more than one
  flavor; re-run the odd seeds on the table's flavor, or filter with
  `--gpu-type`.
