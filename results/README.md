# results/ — append-only, synced

Run results land here as JSON files, one per run. This directory is:

- **Append-only.** Never edit or delete an existing results file. The standing
  verification check (`git status` on results/) treats any modification as an
  error.
- **Synced.** Cloud (Hyperstack) runs write results remotely; the human syncs
  them here. Never assume a cloud run happened — check for its results file.

Every results JSON must validate against `src/results_io.py:validate()` and
contain at minimum: config (path + copy + hash), git_sha (+ dirty flag), seed,
gpu_type, wall_time_s, cost_usd (null locally, human-filled for cloud),
timestamps, and a metrics dict.
