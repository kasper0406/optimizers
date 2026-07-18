# WP0 independent verification pass — notes

Date: 2026-07-18. Verifier: report-only agent (no code/config/report modified
except this file). Machine: Apple-Silicon Mac, CPU-only torch; all GPU-dependent
behavior verified at the code/config level only.

## 1. Full test suite

`uv run pytest -q` from a clean state: **287 passed, 0 failed, 0 skipped**
(14 DeprecationWarnings from `torch.jit.script_method` via
`tests/test_optim_registry.py`, harmless). Wall time 47.6 s.

Notable: the previously self-reported failure
`tests/test_optim_muon.py::test_muon_state_dict_roundtrip_resumes_identically`
**now passes** (verified individually and in the full suite). That self-report
is stale — evidently fixed after it was filed.

## 2. Smoke run

`bash scripts/launch_local.sh configs/smoke.yaml` → exit 0, wrote
`results/smoke_seed1234_20260718T192228.json`. The file loads and validates
through `src.results_io.load_result` (schema v1, all required provenance keys
present: git_sha d777d224…, git_dirty true, seed 1234, gpu_type "cpu",
wall_time_s, cost_usd null, ISO timestamps). `max_param_delta == 0.0` confirms
the NoOp optimizer path (10-step training no-op as specified in WP0.0 DoD).

## 3. Standing verification (CLAUDE.md)

### results-append-only — PASS
`results/` is **untracked** in git (whole tree is pre-first-commit for these
paths), so `git status` alone cannot detect in-place edits. Verified instead by
SHA-1 snapshot of every file in `results/` before and after all verification
runs: the only delta is the *addition* of my own smoke-run JSON. No
modification or deletion of the 5 pre-existing JSONs or README.md.
`results_io.write_result` also structurally refuses overwrites
(FileExistsError).

### no-eval-seeds — PASS (with a factual note)
- configs/: every literal `seed:` is ≥ 1000 (smoke 1234; dev configs 1500–1600,
  probe seed 4242). The string "0-99" appears only in prose comments describing
  the abstract policy; `sweep.seeds: eval` / `dev` are abstract policies
  resolved at launch time. `tests/test_wp00_sanity.py` and
  `sweep.find_eval_seed_literals` enforce this programmatically.
- tests/: literal values in 0–99 appear **only** (a) as expected *output* of
  the launch-time resolver (`resolve_seed_policy("eval") == range(100)`), and
  (b) as inputs to negative tests asserting the tooling *refuses* eval-seed
  literals (`config["seed"] = 5`, `seeds=[0,1,2]`, `{"seed": 99}` in
  `tests/test_scripts_sweep.py`). No test trains or tunes on an eval seed.
  Stats-test seeds: 1234/5678/2024/4242 — all dev-range.

### criteria-untouched — PASS
`criteria/` contains exactly `README.md` + 4 `*.template` files. Each template
carries the required "HUMAN MUST AUTHOR — agent-drafted template, all
thresholds intentionally blank" header. Grep for numeric threshold values
outside TODO lines: none (TODO counts: airbench 11, nanogpt 12, phase1 20,
phase2 14). git blame is not applicable (files untracked, pre-first-commit);
no non-template criteria file exists.

### submodules-pinned — PASS
`git submodule status`:
- vendor/airbench @ 4c1b6d1e3889b037efadcfd5c0ea65b246592362 (clean)
- vendor/modded-nanogpt @ edf47a05a12062d661c4cfd4eef848c5ab5bed32 (clean)
- vendor/DynMuon @ 89baa66693819ef09e26915a5b46bccfc77913eb (clean)
All three match the expected SHAs exactly; no `+`/`-` dirty markers.

### no-stats-reimplementation — PASS
`src/instrument/tracker.py` imports `RegimeClassifier` from `src.stats` and
instantiates one per (direction, beta); grep for inline EMA update patterns
(`beta * … + (1-beta) * …`) across `src/instrument/*.py` finds nothing.
schema.py/plots.py only carry field names (`rho`, `t_stat`) as log schema;
subspace.py is linear algebra (power iteration), not statistics.

### no-hvp-in-update-path — PASS
`grep -riE 'hvp|hessian' src/ scripts/` hits **only** `src/instrument/*`
(tracker HVP callback + schema/plots fields). Zero hits in `src/optim/`; no
optimizer module imports `src.instrument`; `routed.py` is still a WP0.0
placeholder docstring (WP2.1-gated, correct). The tracker's `hvp_fn` is an
optional callback defaulting to None with the Phase-1-validation-only policy
documented at `src/instrument/tracker.py:22-26`.

## 4. Docker build

`docker build --platform linux/amd64 -t routed-muon:dev .` with an 8-minute
cap (Python wrapper; macOS has no `timeout(1)`): see structured output for
final status. Base image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`;
dependency layer via `uv sync --frozen`. Note the image ENTRYPOINT is
`scripts/run.py`, which today registers only the `smoke` experiment and `noop`
optimizer (see §5) — the container as-built can run `configs/smoke.yaml` but
not the airbench/dynmuon configs until those experiment runners are registered.

## 5. Cross-cutting reads

### Does scripts/run.py dispatch to the src/optim registry?
**Not directly.** `scripts/run.py` defines its own local
`OPTIMIZER_REGISTRY = {"noop": NoOpOptimizer}` and
`EXPERIMENT_REGISTRY = {"smoke": run_smoke}` and does **not** import
`src/optim/registry.py` (which holds muon/adamw/dynmuon/adamuon/normuon).
The wiring is runtime injection: `src/optim/airbench_zoo.py:main` updates
`run_mod.OPTIMIZER_REGISTRY` with the full zoo and registers the
`airbench_smoke` experiment, then delegates to `run.py:main` — verified by
`tests/test_optim_registry.py::test_entrypoint_wires_zoo_into_run_py` and
`::test_run_py_registration_comment_is_satisfied_at_runtime` (both pass).
The self-report on this is accurate but **slightly understated**: the runtime
registration covers only the `airbench_zoo` entrypoint path. Two paths do not
get it today:
1. `scripts/sweep.py` emits commands of the form
   `uv run python scripts/run.py <cfg> --seed N` (plain run.py), and
2. the Docker ENTRYPOINT is plain `scripts/run.py`.
So an *executed* wp01 sweep or a container run of a zoo config would fail with
"Unknown experiment". Not currently a defect — see next item — but it is the
gap that WP0.1 harness work must close.

### Do wp01/wp03 configs reference real entry points?
**Not yet — and both configs say so explicitly.**
- `configs/wp01_airbench_eval.yaml` names `experiment: airbench`, which is not
  registered anywhere (only `airbench_smoke` exists, via the zoo entrypoint).
  The config's own NOTE (lines 15–18) says the runner registration lands with
  the WP0.1 harness work; "sweep expansion itself works today" — confirmed:
  `sweep.py --dry-run` expands to exactly 100 runs, seeds 0–99, exit 0.
- `configs/wp03_dynmuon_repro.yaml` is explicitly "STATUS: SKELETON — DO NOT
  LAUNCH" with all science fields TBD; `experiment: dynmuon_repro` is
  unregistered by design. Dry-run expands 10 dev-seed runs (1000–1009), exit 0.
- dev configs use `experiment: airbench_smoke` (registered via zoo entrypoint)
  and `instrumented_mlp_smoke` (consumed by `scripts/bench_overhead.py`'s own
  runner, not run.py).

### Hand-computed optimizer test spot-check
Recomputed `tests/test_optim_adamw.py::test_adamw_three_steps_match_numpy_hand_computed`
**independently**: my own scalar-by-scalar float64 implementation of the
Loshchilov–Hutter decoupled-decay update (written from the paper formula, not
copied from the test) on the same fixed 4×3 matrix and 3 gradients. Result:
repo `AdamW` after 3 steps matches my expectation with max abs diff 8.0e-8
(float32 rounding level; e.g. element [0][0]: 0.400951043 expected vs
0.400951028 got). The test's expectation and the optimizer agree with an
independent derivation. The test additionally cross-checks against
`torch.optim.AdamW` at 1e-7 tolerance.

## 6. Verification of self-reported issues

| # | Self-reported issue | Verdict |
|---|---|---|
| 1 | src/stats pre-existing, verified-not-rewritten | Consistent; suite green; nothing to add |
| 2 | β=0.9 label flapping near ρ_osc boundary | Confirmed documented in `reports/wp05-stats-validation.md` §1 (occupancy ≥0.6 vs ≥0.9) |
| 3 | Spurious innovation resets at steps 377/379 etc. | Confirmed documented descriptively in wp05 report (§ switch tests), with the SIGNAL-prior failure-mode framing |
| 4 | run.py "WP0.4 registers…" comment / runtime registration | Real, accurate, **slightly understated** — see §5: sweep path and Docker ENTRYPOINT do not receive the runtime registration |
| 5 | torchvision absent from pyproject | Confirmed (`grep torchvision pyproject.toml` empty); suite including vendored-airbench import tests passes on CPU regardless |
| 6 | argv[0] fix for vendored airbench import | Covered by passing registry tests; no independent repro attempted (report-only) |
| 7 | state_dict aliasing note | Design note, not a defect; roundtrip test passes |
| 8 | DynMuon bf16-cast fidelity (dynmuon.py:595-597) | Confirmed the vendored code casts to bf16 "before communication" at the cited location; difference bf16-rounding-level as stated |
| 9–11 | Deliverables pre-existing / '0-99' prose-only / criteria README | All confirmed (see §3) |
| 12 | **Pre-existing muon state-dict test failure** | **Stale — the test passes now** (individually and in full suite) |
| 13–16 | arXiv-fetch provenance caveats, GPU-count caveat, quotes-report amendments | Report-content caveats, appropriately hedged in the reports; the overhead.tex off-by-one (line 41 header vs 42 first data row) is self-flagged and not independently checkable without re-fetching the arXiv archive |

## 7. Issues found by this pass (beyond the self-reports)

1. **Stale self-report**: the muon state-dict roundtrip failure no longer
   reproduces; whoever consumes the issue list should drop it.
2. **Registration gap is wider than the self-report phrasing**: neither the
   sweep-emitted `run_all.sh` commands nor the Docker ENTRYPOINT go through
   `airbench_zoo.main`, so today only `smoke` is executable via those paths.
   Both wp01/wp03 configs document their runners as pending, so nothing is
   silently broken — but WP0.1 must register `airbench` in run.py (or point
   sweep/Docker at the zoo entrypoint) before any eval sweep can execute.
3. **git-based standing checks are weak pre-first-commit**: results/,
   criteria/, configs/ are all untracked, so "git status shows no modification"
   and "git blame on criteria/" are vacuous until an initial commit lands.
   Verified by content hashing instead this session; recommend an early commit
   so the append-only and criteria checks have teeth.
4. Minor: `smoke.yaml` seed is 1234 and the launch script may be run twice in
   the same second producing a filename collision — `write_result` would then
   refuse (append-only), failing the run. Cosmetic; timestamp includes seconds.

No modifications were made to any file outside this report. Scratch artifacts
(hash snapshots, docker log, spot-check script) live in the session scratchpad.
