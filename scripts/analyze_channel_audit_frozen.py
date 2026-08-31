#!/usr/bin/env python
"""Phase B producer for program #23 -- the FROZEN-tier channel audit (prereg 6b).

    reports/channel-audit-preregistration.md 6b registers this file by name and
    K0(b) makes it a launch precondition: before this existed, no registered
    Phase B quantity had a producer at all (that document's repair R9).

It reads the 15 Phase B sidecars (``results/*.instrumentation.json`` written
with ``instrumentation.frozen_probes.enabled: true``), computes every
registered quantity of the section 6b table, and writes

    reports/channel-audit.{md,json}  +  reports/figures/channel-audit-*.png

DESCRIPTIVE ONLY.  Every number is printed next to the threshold the
pre-registration proposes for it, and the row of each registered outcome map
that the numbers mechanically fall in is reported as a LOOKUP, labelled
``adjudication: HUMAN``.  Nothing here evaluates a gate, amends a document, or
emits a verdict (CLAUDE.md ground rule 1).  The thresholds are carried in
``REGISTERED`` with their status -- at the time of writing every one of them is
``PROPOSED`` and unfrozen, and the report says so on its first page.

WHY A SEPARATE FILE.  ``scripts/analyze_channel_audit.py`` is the Phase A
producer (prereg 6a): tracked tier, 218 already-peeked sidecars, segment-level
unit, batched kernel, and three defaults that deliberately differ from the
registration.  Prereg 6b requires the confirmatory surface to have its own
producer so the validated Phase A script is not modified to serve a tier it
was not written for.  Nothing in that file is imported or changed here.

NO REIMPLEMENTATION (prereg 5 preamble, CLAUDE.md WP1.1).  Every estimator is
``src.stats.spectral`` called directly, per series: ``lag_ladder`` for the
ladder and phi_hat, ``channel_t`` for both channel means, ``ar1_surrogate_null``
for the nulls, ``ar1_streams`` for the synthetic controls and the tau
references, ``block_bootstrap_ci`` for every interval.  This module contributes
ingest, labelling, pooling, calibration and reporting -- no second estimator.

REGISTERED DEFAULTS (prereg 6b's table; asserted by
``tests/test_analyze_channel_audit_frozen.py`` by reading the parser):

    --max-lag           64        the ladder (the NW bandwidth is still L = 4)
    tau truncation      K in {8, 16, 32, 64}, K = 32 primary
    --null-reps         200000    (5.6: 2000 cannot resolve a 1e-4 rate)
    --null-seed         4242      plus 4243 for tau_white (5.9)
    --bootstrap-block   64        one frozen bank
    pooling             median, EXCEPT tau: mean (5.8)

TWO EXACTNESS NOTES, both stated rather than hidden:

* **The NW cap used for ``channel_t`` is 8, not 64, and the two are bitwise
  identical here.**  The bandwidth is ``L = min(max_lag, floor(4 (n/100)^(2/9)),
  n - 2)``, which is 4 at n = 195/187 and 3 at n = 45/37; every lag product
  above ``L`` is computed and then discarded.  Capping the *null's* and the
  observed series' NW call at ``NW_MAX_LAG = 8`` (the cap
  ``scripts/channel_audit_anchors.py`` already uses, and the one prereg 2's
  null-anchor table was drawn with) therefore returns the identical statistic
  ~3x faster.  The script asserts the equality of the two bandwidths at every
  series length it sees and refuses to run if it ever fails
  (``estimator.nw_bandwidth_check``), so this is a speed argument that cannot
  silently become an estimator change.
* **``ladder`` and ``phi_hat`` use ``--max-lag`` (64).**  ``rho_1`` is invariant
  to the cap (the +1/n bias correction does not depend on it), so prereg 5.5's
  ``lag_ladder(..., max_lag=32)['rho'][0]`` and this one agree exactly.

Cost.  The nulls dominate: ``ar1_surrogate_null`` at the registered 200000 reps
costs ~5 s per (phi, series length) on one CPU core, and the design needs one
per (matrix, batch rung, series length, burn-in) plus the phi +/- 0.05 and
phi = -0.34 sensitivities.  A full 15-run read is therefore tens of minutes of
single-core CPU, not seconds; nothing here needs a GPU.

Determinism: NumPy only, seeded RNGs only, no timestamps, sorted keys, figures
written with fixed metadata -- identical inputs produce byte-identical outputs.

Usage:
    uv run --no-sync python scripts/analyze_channel_audit_frozen.py \
        --sidecars results \
        --out-md reports/channel-audit.md \
        --out-json reports/channel-audit.json \
        --out-figdir reports/figures

    # K1's controls run the FULL pipeline on a generated stream:
    uv run --no-sync python scripts/analyze_channel_audit_frozen.py \
        --synthetic-control white --out-json /tmp/white.json --out-md /tmp/white.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.instrument.schema import (  # noqa: E402
    SIDECAR_SUFFIX,
    load_instrumentation,
)
from src.stats.spectral import (  # noqa: E402
    CHANNELS,
    ar1_streams,
    ar1_surrogate_null,
    block_bootstrap_ci,
    channel_t,
    lag_ladder,
    newey_west_bandwidth,
)

# --------------------------------------------------------------- registered

NW_MAX_LAG = 8  # cap on channel_t's Bartlett window; the rule gives L = 4 here
LADDER_MAX_LAG = 64  # prereg 6b: --max-lag, the ladder
TAU_LAGS = (8, 16, 32, 64)  # prereg 5.9 / 6b
TAU_PRIMARY_K = 32
REGISTERED_BURN_INS = (5, 15, 25)  # prereg 5.3, always reported together
PRIMARY_BURN_IN = 5
NULL_REPS = 200_000  # prereg 5.6
NULL_SEED = 4242
TAU_REFERENCE_SEED = 4243  # prereg 5.9, deliberately != NULL_SEED
BOOTSTRAP_REPS = 2000
BOOTSTRAP_BLOCK = 64  # prereg 5.8: exactly one (run, matrix) frozen bank
BOOTSTRAP_BLOCK_TRACKED = 4  # prereg 5.8: the 4 segments of one direction
PHI_SENSITIVITY_OFFSETS = (-0.05, 0.05)  # prereg 5.5
PHASE_A_PHI = -0.34  # prereg 5.5's fixed-phi sensitivity
PHI_CLAMP = 0.98  # ar1_surrogate_null requires |phi| < 1

# prereg 5.8: the pooling is registered per quantity and is load-bearing.
POOLING = {
    "excess_dc": "median",
    "frac_c": "rate",
    "frame_gain": "median",
    "phi_hat": "median",
    "ratio_c": "median",
    "tau": "mean",
}

# prereg 3: the registered run set.  Rows 1-2 are the 9-run B = 2000 core that
# P1/P2/P3 are read on; rows 3-5 are the step-matched batch rider.
CORE_SEEDS = (1300, 1301, 1302, 1310, 1311)
RIDER_SEEDS = (1320, 1321)
CORE_BATCH = 2000
RIDER_BATCHES = (500, 2000, 8000)
NYQUIST_EPOCH_HARMONIC_BATCH = 500  # Rider-2: epoch length 100 steps at B = 500

# prereg 4 / Appendix.  Frozen by the human on 2026-08-31 ("freeze as
# proposed": every appendix row adopted at its proposed value).  The status
# travels with the number into the report so the freeze is visible in every
# table; adjudication of every branch remains HUMAN.
THRESHOLD_STATUS = "FROZEN 2026-08-31 (human, as proposed)"
REGISTERED = {
    "k2_nw_floored_frac": 0.05,
    "k3_phi_window": (-0.60, -0.15),
    "k3_phi_spread_max": 0.35,
    "k4_frozen_median_t_dc_max": 2.0,
    "k6_channel_shape_divergence": 0.15,
    "p1_band_contrast_middle_edge": 1.15,
    "p1_band_contrast_pass": 1.30,
    "p1_frac_alt_floor": 0.010,
    "p1_min_dc_events": 10,
    "p1_tail_contrast_pass": 3.0,
    "p2_bulk_elevated_floor": 2.0,
    "p2_bulk_tracks_ceiling": 1.3,
    "p2_frame_gain_pass": 3.0,
    "p3_consistency_band": (0.75, 1.30),
    "p3_decisive_upper": 1.0,
    "rider1_fail_flat": 1.5,
    "rider1_pass_band": (2.8, 5.6),
    "rider1_vacuity_guard": 0.05,
    "rider2_invariance_max_over_min": 1.3,
    "rider2_sampling_ess_tolerance": 1.15,
    "t_exceedance": 4.0,
}

# prereg 2's peeked published anchors, printed next to the re-read (prereg 4's
# "ESS/n distribution vs the published pooled anchors", repair R4).  These are
# PEEKED priors, never a reference distribution.
PUBLISHED_ANCHORS = {
    "dc_frac_abs_t_ge_4": 0.00926,
    "dc_median_abs_t_nw": 0.8738,
    "ess_over_n_median": 1.9495,
    "ess_over_n_min": 1.0877,
    "n_nw_floored": 0,
    "n_probes": 864,
    "per_matrix_phi_hat_sorted": (-0.531, -0.460, -0.393, -0.355, -0.292, -0.267),
    "phi_hat_pooled": -0.384,
    "tau_at_L4_median": 0.513,
}

TIERS = ("frozen", "tracked")
POOLS = ("core", "rider")
QUANTILES = (0.25, 0.50, 0.75, 0.90)

# The registered tail contrast is read at ``theta = 4 / median_null,dc``, where
# the null's own exceedance rate is ~1e-4: on a pure null a 3456-probe pool
# holds ~0.3 DC exceedance events, so the registered denominator guard (prereg
# 4 P1, "< 10 events") fires by construction and ``tail_contrast`` is undefined
# there -- for ANY feasible pool size, since a +/-10% interval on a ratio of two
# 1e-4 rates needs ~1e7 probes.  K1 nevertheless requires the synthetic controls
# to return a tail contrast within 1.00 +/- 0.10 (prereg 7 K1, 6d).  The same
# statistic at a threshold the null CAN resolve is what makes that clause
# checkable: identical functional form (one calibrated threshold, common to both
# channels, taken from the null's own per-rep samples per prereg 5.7), read at
# the null's q75 / q90 / q99 instead of at 4 sigma.  Reported as a profile next
# to the registered value, never as a substitute for it.
TAIL_PROFILE_QUANTILES = (0.75, 0.90, 0.99)
# K1 reads the companion at q75 and not deeper for the reason prereg 5.6 gives
# for reps = 200000: a tolerance is only a test while it exceeds the estimator's
# own Monte-Carlo error.  At 3456 probes the ratio of two 10%-rates carries
# ~7.6% sd against K1's 10% tolerance (a ~30% false-alarm clause); at q75 it
# carries ~4.4%, so the clause fires on a broken pipeline rather than on noise.
K1_TAIL_QUANTILE = 0.75


def _f(x: Any, nd: int = 6) -> Optional[float]:
    """Round for stable JSON across platforms; ``None`` for non-finite."""
    if x is None:
        return None
    v = float(x)
    if not np.isfinite(v):
        return None
    return float(np.round(v, nd))


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """``num / den``, ``None`` on a missing or zero denominator.

    Every ratio in this file has a Monte-Carlo or count denominator that can
    legitimately be zero at reduced ``--null-reps`` or on a small pool; a
    reported non-number is a finding, a ZeroDivisionError is a crash.
    """
    if num is None or den is None:
        return None
    num, den = float(num), float(den)
    if den == 0.0 or not np.isfinite(den) or not np.isfinite(num):
        return None
    return num / den


def _nan_reduce(a: np.ndarray, fn, axis: int = 0) -> np.ndarray:
    """``fn`` ignoring NaN, returning NaN (not a warning) on an all-NaN slice.

    The ladder is NaN at every lag a series is too short for (lag 45..64 on a
    45-step tracked segment), so an all-NaN column is an expected, meaningful
    state -- "this lag does not exist at this length" -- and must not raise a
    RuntimeWarning on every report.
    """
    a = np.asarray(a, dtype=np.float64)
    ok = np.any(np.isfinite(a), axis=axis)
    out = np.full(a.shape[1 - axis] if a.ndim == 2 else 1, np.nan)
    if np.any(ok):
        cols = np.where(ok)[0]
        out[cols] = fn(a[:, cols] if axis == 0 else a[cols, :], axis=axis)
    return out


def _summary(values: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"max": None, "mean": None, "median": None, "min": None,
                "n": 0, "q25": None, "q75": None, "q90": None}
    q25, med, q75, q90 = (float(x) for x in np.quantile(v, QUANTILES))
    return {
        "max": _f(v.max()),
        "mean": _f(v.mean()),
        "median": _f(med),
        "min": _f(v.min()),
        "n": int(v.size),
        "q25": _f(q25),
        "q75": _f(q75),
        "q90": _f(q90),
    }


# ------------------------------------------------------------------ loading


def tombstoned_runs(dir_path: Path) -> set:
    """Run names listed in ``results/INVALID_RUNS.json`` (append-only).

    Same lookup the Phase A producer uses: that file's header says its entries
    "must be excluded by every analysis tool", so the exclusion is a lookup and
    never a deletion.
    """
    tomb = Path(dir_path) / "INVALID_RUNS.json"
    if not tomb.exists():
        return set()
    try:
        entries = json.loads(tomb.read_text()).get("invalid", [])
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt input
        raise SystemExit(f"unreadable {tomb}: {exc}")
    out = set()
    for e in entries:
        name = str(e.get("file", ""))
        if name.endswith(".json"):
            name = name[: -len(".json")]
        if name:
            out.add(name)
    return out


def select_sidecars(
    dir_path: Path,
    *,
    run_prefix: Optional[str] = None,
    min_seed: Optional[int] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Sorted ``*.instrumentation.json`` paths, filtered and reported.

    Filters, in order, each reported rather than assumed: ``INVALID_RUNS.json``
    tombstones; ``--run-prefix``; ``--min-seed`` (CLAUDE.md ground rule 2 --
    evaluation seeds 0-99 never enter development analysis).  The frozen-tier
    filter itself is NOT here: whether a sidecar carries a ``frozen_probes``
    block is a property of its contents, so it is applied in :func:`ingest`
    and reported as ``inputs.skipped``.
    """
    all_paths = sorted(Path(dir_path).glob(f"*{SIDECAR_SUFFIX}"))
    if not all_paths:
        raise SystemExit(f"no {SIDECAR_SUFFIX} files under {dir_path}")
    tombs = tombstoned_runs(dir_path)
    excluded: Dict[str, List[str]] = {"invalid_runs": [], "min_seed": [], "run_prefix": []}
    kept: List[Path] = []
    n_no_seed = 0
    for path in all_paths:
        name = path.name[: -len(SIDECAR_SUFFIX)]
        if name in tombs:
            excluded["invalid_runs"].append(path.name)
            continue
        if run_prefix and not name.startswith(run_prefix):
            excluded["run_prefix"].append(path.name)
            continue
        seed = None
        for part in name.split("_"):
            if part.startswith("seed") and part[4:].isdigit():
                seed = int(part[4:])
        if seed is None:
            n_no_seed += 1
        elif min_seed is not None and seed < min_seed:
            excluded["min_seed"].append(path.name)
            continue
        kept.append(path)
    if not kept:
        raise SystemExit(
            f"every {SIDECAR_SUFFIX} under {dir_path} was filtered out "
            f"(run_prefix={run_prefix!r}, min_seed={min_seed})"
        )
    if limit is not None and 0 < limit < len(kept):
        step = len(kept) / float(limit)
        kept = [kept[int(i * step)] for i in range(limit)]
    selection = {
        "excluded": {k: sorted(v) for k, v in sorted(excluded.items())},
        "min_seed": min_seed,
        "n_discovered": len(all_paths),
        "n_excluded": sum(len(v) for v in excluded.values()),
        "n_selected": len(kept),
        "name_without_seed": n_no_seed,
        "run_prefix": run_prefix,
    }
    return kept, selection


def run_metadata(sidecar: Path) -> Dict[str, Any]:
    """lr / batch_size / seed for a sidecar, from its paired results JSON.

    A run that cannot be labelled is reported as skipped, never guessed at
    (CLAUDE.md ground rule 6): the ``problem`` key carries the reason and the
    labels stay ``None``.
    """
    name = sidecar.name[: -len(SIDECAR_SUFFIX)]
    main = sidecar.with_name(name + ".json")
    out: Dict[str, Any] = {
        "batch_size": None, "lr": None, "problem": None, "run": name,
        "seed": None, "sidecar": sidecar.name,
    }
    if not main.exists():
        out["problem"] = f"missing main results file {main.name}"
        return out
    try:
        with open(main) as fh:
            res = json.load(fh)
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt input
        out["problem"] = f"unreadable main results file: {exc}"
        return out
    contents = (res.get("config") or {}).get("contents") or {}
    lr = (contents.get("probe_overrides") or {}).get("lr")
    if lr is None:
        lr = (res.get("metrics") or {}).get("optimizer_lr")
    batch = (contents.get("train") or {}).get("batch_size")
    out["batch_size"] = None if batch is None else int(batch)
    out["lr"] = None if lr is None else float(lr)
    out["seed"] = res.get("seed")
    pointer = (res.get("metrics") or {}).get("instrumentation_sidecar")
    if pointer is not None and pointer != sidecar.name:
        out["problem"] = (
            f"results metrics.instrumentation_sidecar={pointer!r} does not "
            f"name {sidecar.name!r}"
        )
    elif out["lr"] is None or out["batch_size"] is None:
        out["problem"] = "config carries no probe lr and/or train.batch_size"
    return out


def load_sidecars(paths: Sequence[Path]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """``[(metadata, instrumentation log)]`` for every selected sidecar."""
    return [(run_metadata(p), load_instrumentation(p)) for p in paths]


# ---------------------------------------------------------------- ingest


def boundaries_from_steps(
    steps: np.ndarray, cut_steps: Sequence[int]
) -> Tuple[Tuple[int, int], ...]:
    """Half-open index intervals of ``steps`` delimited by ``cut_steps``.

    A refresh at step ``r`` starts a new segment AT ``r`` (the tracker refreshes
    before the step is logged), so ``r`` is the first observation of the new
    segment.  Cuts outside the recorded range are ignored; leading and trailing
    remainders are segments of their own.
    """
    edges = {0, int(steps.size)}
    for r in cut_steps:
        idx = int(np.searchsorted(steps, int(r)))
        if 0 < idx < steps.size:
            edges.add(idx)
    ordered = sorted(edges)
    return tuple((a, b) for a, b in zip(ordered[:-1], ordered[1:]) if b > a)


def pool_of(seed: Any, batch: Any, core_seeds, rider_seeds, core_batch) -> str:
    """prereg 3's registered criterion pools, by (seed, batch rung).

    P1/P2/P3 are read on the 9-run B = 2000 core (rows 1-2 of the section 3
    table); Rider-1/Rider-2 on rows 3-5 (seeds 1320/1321, three batch rungs).
    A run in neither is labelled ``unassigned``, reported, and enters no
    criterion -- the pools are registered and do not move to fit the data.
    """
    if seed is None:
        return "unassigned"
    seed = int(seed)
    if seed in core_seeds and batch is not None and int(batch) == int(core_batch):
        return "core"
    if seed in rider_seeds:
        return "rider"
    return "unassigned"


class SeriesBank:
    """Every projection series entering the estimator, tier-agnostic.

    ``rows`` are ordered by (run, matrix, tier, probe/direction index, segment)
    -- the order prereg 5.8 registers for the block bootstrap, so a block of 64
    consecutive frozen rows is exactly one (run, matrix) frozen bank.
    """

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.runs: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, Any]] = []
        self.tier_contrast_raw: Dict[str, Dict[str, List[float]]] = {}
        self.parity: Dict[str, Any] = {}
        self.logged_finals: List[Dict[str, Any]] = []

    def where(self, **kw) -> List[int]:
        out = []
        for i, r in enumerate(self.rows):
            if all(r.get(k) == v for k, v in kw.items()):
                out.append(i)
        return out


def ingest(
    entries: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    *,
    min_n: int = 10,
    core_seeds: Sequence[int] = CORE_SEEDS,
    rider_seeds: Sequence[int] = RIDER_SEEDS,
    core_batch: int = CORE_BATCH,
    require_decimate_one: bool = True,
) -> SeriesBank:
    """Sidecars -> :class:`SeriesBank`, with the registered structural checks.

    Three refusals, all loud (prereg 1, 3 and 5.2):

    * a ``frozen_probes`` block written with ``decimate > 1`` -- the ladder and
      both channels are computed from the RAW per-step series, so a dropped
      step silently destroys the demodulation;
    * a raw-step sequence with a gap, for the same reason;
    * **mixed segment-start parity** -- ``channel_t`` fixes the demodulation
      sign from each supplied array's first element, so segments whose absolute
      starts differ in parity carry opposite alternating signs and cancel when
      pooled.  With ``t_refresh`` even every start is odd (1, 51, 101, 151) and
      the check is inert; it exists because a future even start would break the
      channel silently.
    """
    bank = SeriesBank()
    parities: List[int] = []
    for meta, log in sorted(entries, key=lambda e: str(e[0].get("run"))):
        run = str(meta.get("run"))
        if meta.get("problem"):
            bank.skipped.append({"reason": meta["problem"], "run": run})
            continue
        if not log.get("frozen_probes_enabled"):
            bank.skipped.append({"reason": "frozen_probes disabled in this run", "run": run})
            continue
        pool = pool_of(meta.get("seed"), meta.get("batch_size"), core_seeds, rider_seeds, core_batch)
        n_frozen = 0
        n_tracked = 0
        for name in sorted(log.get("matrices", {})):
            mat = log["matrices"][name]
            block = mat.get("frozen_probes")
            if block:
                if require_decimate_one and int(block.get("decimate", 1)) != 1:
                    raise SystemExit(
                        f"{run}/{name}: frozen_probes.decimate="
                        f"{block.get('decimate')} != 1. prereg 3 registers "
                        "decimate: 1 as load-bearing -- the ladder and both "
                        "channels are computed from the raw per-step series."
                    )
                raw_steps = np.asarray(block.get("raw_steps", []), dtype=np.int64)
                if raw_steps.size and not np.all(np.diff(raw_steps) == 1):
                    raise SystemExit(
                        f"{run}/{name}: frozen raw_steps are not consecutive; "
                        "the alternating channel is tied to the absolute step."
                    )
                start = int(raw_steps[0]) if raw_steps.size else 0
                for probe in sorted(block.get("probes", []), key=lambda p: int(p["index"])):
                    series = np.asarray(probe.get("s", []), dtype=np.float64)
                    if series.size - min_n < 0:
                        continue
                    parities.append(start % 2)
                    bank.rows.append({
                        "batch": meta.get("batch_size"),
                        "index": int(probe["index"]),
                        "kind": "frozen",
                        "lr": meta.get("lr"),
                        "matrix": name,
                        "pool": pool,
                        "run": run,
                        "seed": meta.get("seed"),
                        "segment": 0,
                        "series": series,
                        "start_step": start,
                        "tier": "frozen",
                    })
                    n_frozen += 1
                    final = probe.get("final") or {}
                    if final:
                        bank.logged_finals.append({
                            "logged": final,
                            "matrix": name,
                            "max_lag": int(block.get("max_lag", NW_MAX_LAG)),
                            "probe": int(probe["index"]),
                            "run": run,
                            "series": series,
                        })
            steps = np.asarray(mat.get("steps", []), dtype=np.int64)
            if steps.size:
                bounds = boundaries_from_steps(steps, mat.get("refresh_steps", []))
                for d in sorted(mat.get("directions", []), key=lambda x: int(x["index"])):
                    s_all = np.asarray(d.get("s", []), dtype=np.float64)
                    if s_all.size != steps.size:
                        continue
                    for si, (a, b) in enumerate(bounds):
                        if b - a < min_n:
                            continue
                        parities.append(int(steps[a]) % 2)
                        bank.rows.append({
                            "batch": meta.get("batch_size"),
                            "index": int(d["index"]),
                            "kind": str(d.get("kind", "unknown")),
                            "lr": meta.get("lr"),
                            "matrix": name,
                            "pool": pool,
                            "run": run,
                            "seed": meta.get("seed"),
                            "segment": si,
                            "series": s_all[a:b],
                            "start_step": int(steps[a]),
                            "tier": "tracked",
                        })
                        n_tracked += 1
                    for beta, series in sorted((d.get("per_beta") or {}).items()):
                        ts = series.get("t_stat") or []
                        if not ts:
                            continue
                        by_kind = bank.tier_contrast_raw.setdefault(str(beta), {})
                        by_kind.setdefault(str(d.get("kind", "unknown")), []).append(abs(float(ts[-1])))
                        by_kind.setdefault("all", []).append(abs(float(ts[-1])))
        bank.runs.append({
            "batch_size": meta.get("batch_size"),
            "lr": meta.get("lr"),
            "n_frozen_probes": n_frozen,
            "n_tracked_segments": n_tracked,
            "pool": pool,
            "run": run,
            "seed": meta.get("seed"),
        })
    n_odd = int(sum(parities))
    bank.parity = {
        "n_odd_starts": n_odd,
        "n_segment_starts": len(parities),
        "parity": "odd" if n_odd and n_odd == len(parities) else ("even" if parities else None),
    }
    if parities and n_odd not in (0, len(parities)):
        raise SystemExit(
            f"segment starts have MIXED parity ({n_odd} odd of {len(parities)}): "
            "channel_t fixes the demodulation sign from each supplied array's "
            "first element, so pooling series whose absolute starts differ in "
            "parity flips the alternating sign between them and cancels a real "
            "signal. prereg 1 registers this assertion; refusing to report."
        )
    if not bank.rows:
        raise SystemExit(
            "no usable frozen-tier series in these sidecars -- the runs were "
            "made with instrumentation.frozen_probes disabled, or every series "
            "was shorter than --min-n"
        )
    return bank


# ----------------------------------------------------- per-series estimator


def probe_table(rows: Sequence[Dict[str, Any]], burn_in: int, max_lag: int) -> Dict[str, Any]:
    """Per-series ladder + both channel statistics, via ``src.stats.spectral``.

    One ``lag_ladder`` and two ``channel_t`` calls per series -- the tested
    module, never a batched copy of it.  ``rho`` carries the registered
    bias-corrected ladder (``+1/n`` on every lag) and ``rho_raw`` the
    uncorrected one, so the correction stays auditable (prereg 5.4).
    """
    n = len(rows)
    out: Dict[str, Any] = {
        "burn_in": int(burn_in),
        "max_lag": int(max_lag),
        "n_kept": np.zeros(n, dtype=np.int64),
        "n_raw": np.zeros(n, dtype=np.int64),
        "rho": np.full((n, max_lag), np.nan),
        "rho_raw": np.full((n, max_lag), np.nan),
    }
    for ch in CHANNELS:
        out[ch] = {
            "abs_t": np.full(n, np.nan),
            "ess": np.full(n, np.nan),
            "ess_over_n": np.full(n, np.nan),
            "nw_floored": np.zeros(n, dtype=bool),
            "t_nw": np.full(n, np.nan),
        }
    for i, row in enumerate(rows):
        s = row["series"]
        out["n_raw"][i] = int(s.size)
        ladder = lag_ladder(s, max_lag=max_lag, burn_in=burn_in, bias_correct=True)
        out["n_kept"][i] = int(ladder["n"])
        out["rho"][i] = np.asarray(ladder["rho"], dtype=np.float64)
        out["rho_raw"][i] = np.asarray(ladder["rho_raw"], dtype=np.float64)
        for ch in CHANNELS:
            st = channel_t(s, ch, burn_in, max_lag=NW_MAX_LAG)
            out[ch]["t_nw"][i] = st["t_nw"]
            out[ch]["abs_t"][i] = abs(float(st["t_nw"]))
            out[ch]["ess"][i] = st["ess"]
            out[ch]["ess_over_n"][i] = float(st["ess"]) / max(int(st["n"]), 1)
            out[ch]["nw_floored"][i] = bool(st["nw_floored"])
    return out


def nw_bandwidth_check(n_raw_values: Sequence[int], burn_in: int, max_lag: int) -> Dict[str, Any]:
    """``NW_MAX_LAG`` must give the same Bartlett bandwidth as ``--max-lag``.

    That equality is what makes the 8-lag NW call bitwise identical to a
    64-lag one (every product above L is discarded).  It holds for every n
    below ~2260 and is asserted rather than assumed, so the speed argument can
    never silently become an estimator change (prereg 5.1's mirror contract).
    """
    rows = []
    ok = True
    for n_raw in sorted({int(v) for v in n_raw_values}):
        n = max(int(n_raw) - int(burn_in), 0)
        a = newey_west_bandwidth(n, NW_MAX_LAG)
        b = newey_west_bandwidth(n, int(max_lag))
        ok = ok and a == b
        rows.append({"L_at_max_lag": int(b), "L_at_nw_max_lag": int(a), "n_kept": n})
    return {"by_length": rows, "identical": bool(ok), "nw_max_lag": NW_MAX_LAG}


# ---------------------------------------------------------------- the nulls


def _null_seed(base: int, phi: float, n: int, burn_in: int) -> int:
    """Deterministic per-null seed from the null's own parameters.

    No timestamps, no global RNG state and no dependence on iteration order:
    two cells requesting the same (phi, n, burn_in) null get the same stream,
    which is also what makes the cache safe.
    """
    mix = int(base) * 1000003 + int(round(float(phi) * 1000.0)) * 10007 + int(n) * 101 + int(burn_in)
    return int(abs(mix) % (2**31 - 1))


class NullBank:
    """Cache of ``ar1_surrogate_null`` draws, keyed by (phi, n_raw, burn_in).

    prereg 5.6 registers the null at matched ``(phi_matrix, n, burn_in,
    max_lag)`` for every (matrix, series length) the design produces, and 5.7
    uses only its MEDIAN |t| (as the within-channel scale correction) and its
    per-rep samples (for the exceedance support).  At the registered
    ``reps = 200000`` one draw costs ~5 s of CPU, so the cache is what makes a
    per-matrix, per-length, per-burn-in, per-sensitivity-phi design affordable:
    every distinct (phi, n, burn_in) is drawn exactly once per process.

    ``phi`` is quantized to 1e-3 before it becomes a key, which is also the
    resolution of the seed derivation -- two matrices whose phi_hat agree to
    three decimals share a null, deliberately and reproducibly.
    """

    def __init__(self, reps: int, seed: int) -> None:
        self.reps, self.seed = int(reps), int(seed)
        self._cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.n_draws = 0

    def get(self, phi: float, n_raw: int, burn_in: int) -> Optional[Dict[str, Any]]:
        if phi is None or not np.isfinite(phi) or int(n_raw) - int(burn_in) < 3:
            return None
        phi = float(np.clip(float(phi), -PHI_CLAMP, PHI_CLAMP))
        key = (int(round(phi * 1000.0)), int(n_raw), int(burn_in))
        if key not in self._cache:
            self.n_draws += 1
            self._cache[key] = ar1_surrogate_null(
                key[0] / 1000.0,
                int(n_raw),
                self.reps,
                _null_seed(self.seed, phi, int(n_raw), int(burn_in)),
                burn_in=int(burn_in),
                max_lag=NW_MAX_LAG,
            )
        return self._cache[key]

    def median_abs_t(self, phi: float, n_raw: int, burn_in: int, channel: str) -> Optional[float]:
        null = self.get(phi, n_raw, burn_in)
        if null is None:
            return None
        return float(np.median(np.asarray(null[channel]["samples"]["abs_t_nw"])))

    def quantile_abs_t(
        self, phi: float, n_raw: int, burn_in: int, channel: str, q: float
    ) -> Optional[float]:
        """A |t| threshold read off the null's per-rep samples (prereg 5.7)."""
        null = self.get(phi, n_raw, burn_in)
        if null is None:
            return None
        return float(np.quantile(np.asarray(null[channel]["samples"]["abs_t_nw"]), float(q)))

    def describe(self, phi: float, n_raw: int, burn_in: int, channel: str) -> Optional[Dict[str, Any]]:
        """The section 6b ``null.<matrix>.<channel>.<n>`` record.

        The per-rep draws themselves are not serialized (200000 floats per
        channel per length); they are exactly reproducible from
        ``(phi, n, reps, seed)``, which is what the record carries, together
        with every summary any registered quantity reads off them.
        """
        null = self.get(phi, n_raw, burn_in)
        if null is None:
            return None
        s = np.asarray(null[channel]["samples"]["abs_t_nw"], dtype=np.float64)
        q25, med, q75, q90 = np.quantile(s, QUANTILES)
        return {
            "abs_t_nw": {
                "median": _f(med), "q25": _f(q25), "q75": _f(q75), "q90": _f(q90),
            },
            "burn_in": int(burn_in),
            "ess_over_n_median": _f(null[channel]["ess_over_n"]["median"]),
            "n_kept": int(n_raw) - int(burn_in),
            "n_raw": int(n_raw),
            "nw_floored_frac": _f(null[channel]["nw_floored_frac"]),
            "phi": _f(null["phi"], 4),
            "samples": {
                "frac_ge_2": _f(float(np.mean(s >= 2.0))),
                "frac_ge_3": _f(float(np.mean(s >= 3.0))),
                "frac_ge_4": _f(float(np.mean(s >= 4.0))),
                "note": "per-rep draws are reproducible from (phi, n, reps, seed)",
                "reps": int(null["reps"]),
                "seed": int(null["seed"]),
            },
        }


# -------------------------------------------------------------- calibration


def phi_hat_of(values: np.ndarray) -> Optional[float]:
    """prereg 5.5: ``phi_hat = median_p rho_1(p)`` on the bias-corrected DC ladder."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return float(np.median(v))


def group_keys(rows: Sequence[Dict[str, Any]], idx: Sequence[int]) -> List[Tuple[Any, ...]]:
    """The null group of each row: (tier, matrix, batch).

    prereg 5.5 registers the *cell* as (tier, matrix, batch rung, lr rung) and
    the *null* as "per matrix, at that matrix's own phi_hat".  RESOLVED HERE,
    and stated in the report: nulls are grouped by (tier, matrix, batch rung),
    pooling the lr rungs -- Phase A's A1 reports phi as lr-invariant, Rider-2's
    registered quantity is phi_hat BY BATCH RUNG, and the two frozen series
    lengths (195 / 187) are a batch-rung property.  The per-cell phi_hat, with
    lr, is reported separately under ``cells``.
    """
    return [(rows[i]["tier"], rows[i]["matrix"], rows[i]["batch"]) for i in idx]


def calibrate(
    rows: Sequence[Dict[str, Any]],
    table: Dict[str, Any],
    nulls: NullBank,
    burn_in: int,
    *,
    phi_override: Optional[float] = None,
    phi_offset: float = 0.0,
) -> Dict[str, Any]:
    """Null-calibrated ``T_c`` for every series (prereg 5.7).

    ``T_c(p) = |t_c,nw(p)| / median_null,c`` at that series' matched
    ``(phi_hat of its (tier, matrix, batch) group, n, burn_in)``.  Under a
    correct null ``median_p T_c ~ 1`` per group by construction, which is what
    K1's synthetic controls check.

    ``phi_override`` / ``phi_offset`` serve prereg 5.5's registered sensitivity
    (the same nulls at ``phi_hat +/- 0.05`` and at the fixed Phase A
    ``phi = -0.34``); with neither, every group uses its own fitted phi_hat.
    """
    n = len(rows)
    idx_all = list(range(n))
    keys = group_keys(rows, idx_all)
    by_group: Dict[Tuple[Any, ...], List[int]] = {}
    for i, k in zip(idx_all, keys):
        by_group.setdefault(k, []).append(i)
    phi_by_group: Dict[Tuple[Any, ...], Optional[float]] = {}
    out: Dict[str, Any] = {
        "burn_in": int(burn_in),
        "phi_by_group": {},
        "theta": np.full(n, np.nan),
        "theta_profile": {q: np.full(n, np.nan) for q in TAIL_PROFILE_QUANTILES},
    }
    for ch in CHANNELS:
        out[ch] = {"T": np.full(n, np.nan), "null_median": np.full(n, np.nan)}
    for key, members in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        fitted = phi_hat_of(table["rho"][members, 0])
        phi = fitted if phi_override is None else float(phi_override)
        if phi is not None:
            phi = float(np.clip(phi + float(phi_offset), -PHI_CLAMP, PHI_CLAMP))
        phi_by_group[key] = phi
        for n_raw in sorted({int(table["n_raw"][i]) for i in members}):
            sub = [i for i in members if int(table["n_raw"][i]) == n_raw]
            for ch in CHANNELS:
                med = nulls.median_abs_t(phi, n_raw, burn_in, ch)
                if med is None or med <= 0.0:
                    continue
                out[ch]["null_median"][sub] = med
                out[ch]["T"][sub] = table[ch]["abs_t"][sub] / med
            dc_med = out["dc"]["null_median"][sub[0]] if sub else np.nan
            if np.isfinite(dc_med) and dc_med > 0.0:
                out["theta"][sub] = REGISTERED["t_exceedance"] / dc_med
                for q in TAIL_PROFILE_QUANTILES:
                    thr = nulls.quantile_abs_t(phi, n_raw, burn_in, "dc", q)
                    if thr is not None:
                        out["theta_profile"][q][sub] = thr / dc_med
    out["phi_by_group"] = {
        "/".join(str(x) for x in k): _f(v, 4) for k, v in sorted(
            phi_by_group.items(), key=lambda kv: str(kv[0])
        )
    }
    out["_phi_by_group_raw"] = phi_by_group
    return out


# ---------------------------------------------------------------- intervals


def _bootstrap(
    values: np.ndarray, block: int, args: argparse.Namespace, statistic=np.median
) -> Optional[Dict[str, Any]]:
    """Block-bootstrap CI at the FULL requested rep count (prereg 5.8)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    return block_bootstrap_ci(
        v, int(min(block, v.size)), int(args.bootstrap_reps), int(args.bootstrap_seed),
        level=95.0, statistic=statistic,
    )


def _paired_bootstrap(
    fn, n: int, block: int, args: argparse.Namespace
) -> Optional[Dict[str, Any]]:
    """Block bootstrap of a statistic of SEVERAL same-length per-probe arrays.

    ``band_contrast`` and ``tail_contrast`` are functions of the alt and dc
    statistics OF THE SAME PROBES (prereg 5.10); resampling the two channels
    independently would destroy exactly the pairing the contrast exists to
    exploit.  The resampled object is therefore the probe INDEX, and ``fn``
    reads both channels at those indices, so the block structure (one frozen
    bank per block) and the pairing survive together.
    """
    if n < 2:
        return None
    idx = np.arange(n, dtype=np.float64)

    def stat(v, axis=None):
        return fn(np.rint(v).astype(np.int64), axis)

    return block_bootstrap_ci(
        idx, int(min(block, n)), int(args.bootstrap_reps), int(args.bootstrap_seed),
        level=95.0, statistic=stat,
    )


def _ci(entry: Optional[Dict[str, Any]]) -> Optional[List[Optional[float]]]:
    if not entry:
        return None
    return [_f(entry["ci_lo"]), _f(entry["ci_hi"])]


def _ratio_ci(num: Optional[Dict[str, Any]], den: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ratio of two INDEPENDENT bootstraps, with the delta-method interval.

    ``frame_gain`` divides a tracked-segment median by a frozen-probe median:
    different pools, different block structures, different generators, so the
    two bootstrap standard errors combine exactly to first order on the log
    scale, ``se(r)/r = sqrt((se_n/n)^2 + (se_d/d)^2)``.  A ratio printed with
    the numerator's error alone overstates its own resolution.
    """
    if not num or not den:
        return None
    a, b = float(num["point"]), float(den["point"])
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0.0:
        return None
    ratio = a / b
    rel = 0.0
    for point, entry in ((a, num), (b, den)):
        se = float(entry.get("se", float("nan")))
        if np.isfinite(se) and point > 0.0:
            rel += (se / point) ** 2
    se_ratio = ratio * float(np.sqrt(rel))
    return {
        "ci95": [_f(ratio - 1.959964 * se_ratio), _f(ratio + 1.959964 * se_ratio)],
        "point": _f(ratio),
        "se": _f(se_ratio),
    }


# ------------------------------------------------------------------ P1


def p1_band_contrast(
    rows, table, cal, idx: Sequence[int], args, *, label: str
) -> Dict[str, Any]:
    """prereg 4 P1: the same-probe alt-vs-dc band contrast (repair R1).

    ``ratio_c = median_p T_c``, ``band_contrast = ratio_alt / ratio_dc``,
    ``theta = 4 / median_null,dc`` (so on dc it is exactly the registered
    ``|t| >= 4`` and on alt it is ``4 * median_null,alt / median_null,dc``),
    ``frac_c = frac_p(T_c >= theta)`` and ``tail_contrast = frac_alt / frac_dc``.

    The raw ``ratio_alt`` / ``ratio_dc`` / ``frac(|t_c| >= 4)`` are reported next
    to every contrast, because the contrast is deliberately biased toward FAIL
    whenever the DC channel itself carries a persistent component (prereg 4) and
    that suppression has to stay visible.
    """
    idx = np.asarray(list(idx), dtype=np.int64)
    out: Dict[str, Any] = {"label": label, "n_probes": int(idx.size)}
    if idx.size == 0:
        out["available"] = False
        out["reason"] = "no probes in this pool"
        return out
    T = {ch: cal[ch]["T"][idx] for ch in CHANNELS}
    theta = cal["theta"][idx]
    ok = np.isfinite(T["alt"]) & np.isfinite(T["dc"]) & np.isfinite(theta)
    if int(ok.sum()) < 2:
        out["available"] = False
        out["reason"] = "no calibrated probes (the null could not be drawn)"
        return out
    idx, theta = idx[ok], theta[ok]
    T = {ch: T[ch][ok] for ch in CHANNELS}
    raw = {ch: table[ch]["abs_t"][idx] for ch in CHANNELS}
    exceed = {ch: T[ch] >= theta for ch in CHANNELS}
    ratio = {ch: float(np.median(T[ch])) for ch in CHANNELS}
    frac = {ch: float(np.mean(exceed[ch])) for ch in CHANNELS}
    n_events = {ch: int(exceed[ch].sum()) for ch in CHANNELS}
    band = _ratio(ratio["alt"], ratio["dc"])
    guard = n_events["dc"] < int(REGISTERED["p1_min_dc_events"])
    tail = None if guard else _ratio(frac["alt"], frac["dc"])

    T_alt, T_dc = T["alt"], T["dc"]
    band_ci = _paired_bootstrap(
        lambda i, axis: np.median(T_alt[i], axis=axis) / np.median(T_dc[i], axis=axis),
        int(idx.size), int(args.bootstrap_block), args,
    )
    ratio_ci = {
        ch: _bootstrap(T[ch], int(args.bootstrap_block), args) for ch in CHANNELS
    }
    floor_ok = frac["alt"] >= REGISTERED["p1_frac_alt_floor"]
    tail_ok = (tail is not None) and tail >= REGISTERED["p1_tail_contrast_pass"]
    p1b = bool(floor_ok and (tail_ok if not guard else True))
    if band is None:
        row, verdict = None, None
    else:
        if band >= REGISTERED["p1_band_contrast_pass"]:
            row = 1 if p1b else 2
        elif band >= REGISTERED["p1_band_contrast_middle_edge"]:
            row = 3 if p1b else 4
        else:
            row = 5 if p1b else 6
        verdict = {1: "PASS", 2: "MIDDLE BAND", 3: "MIDDLE BAND",
                   4: "MIDDLE BAND", 5: "MIDDLE BAND, flagged", 6: "FAIL"}[row]
    out.update({
        "available": True,
        "band_contrast": _f(band),
        "band_contrast_ci95": _ci(band_ci),
        "frac_alt": _f(frac["alt"]),
        "frac_dc": _f(frac["dc"]),
        "n_events": {ch: n_events[ch] for ch in CHANNELS},
        "n_probes_calibrated": int(idx.size),
        "outcome_row": row,
        "outcome_row_label": verdict,
        "p1a_bar": REGISTERED["p1_band_contrast_pass"],
        "p1a_holds": None if band is None else bool(band >= REGISTERED["p1_band_contrast_pass"]),
        "p1b_floor_holds": bool(floor_ok),
        "p1b_holds": p1b,
        "p1b_tail_holds": bool(tail_ok),
        "raw": {
            "frac_alt_abs_t_ge_4": _f(float(np.mean(raw["alt"] >= REGISTERED["t_exceedance"]))),
            "frac_dc_abs_t_ge_4": _f(float(np.mean(raw["dc"] >= REGISTERED["t_exceedance"]))),
            "median_abs_t_alt": _f(float(np.median(raw["alt"]))),
            "median_abs_t_dc": _f(float(np.median(raw["dc"]))),
        },
        "ratio_alt": _f(ratio["alt"]),
        "ratio_alt_ci95": _ci(ratio_ci["alt"]),
        "ratio_dc": _f(ratio["dc"]),
        "ratio_dc_ci95": _ci(ratio_ci["dc"]),
        "tail_contrast": _f(tail),
        "tail_contrast_denominator_guard": bool(guard),
        "tail_contrast_profile": {
            "q%d" % int(q * 100): {
                "frac_alt": _f(float(np.mean(T["alt"] >= cal["theta_profile"][q][idx]))),
                "frac_dc": _f(float(np.mean(T["dc"] >= cal["theta_profile"][q][idx]))),
                "n_events_dc": int(np.sum(T["dc"] >= cal["theta_profile"][q][idx])),
                "tail_contrast": _f(_ratio(
                    float(np.mean(T["alt"] >= cal["theta_profile"][q][idx])),
                    float(np.mean(T["dc"] >= cal["theta_profile"][q][idx])),
                )),
                "theta": _f(float(np.nanmedian(cal["theta_profile"][q][idx]))),
            }
            for q in TAIL_PROFILE_QUANTILES
            if np.any(np.isfinite(cal["theta_profile"][q][idx]))
        },
        "tail_contrast_profile_note": (
            "DIAGNOSTIC COMPANION, not the registered statistic: the same "
            "contrast at a threshold the null can resolve (K1 reads q%d). At "
            "the registered theta the null's own exceedance rate is ~1e-4, so "
            "a pure-null control cannot produce an estimable tail contrast at "
            "any feasible pool size and the denominator guard fires by "
            "construction" % int(K1_TAIL_QUANTILE * 100)
        ),
        "theta": _f(float(np.median(theta))),
        "theta_alt_raw_threshold": _f(float(np.median(
            theta * cal["alt"]["null_median"][idx]
        ))),
        "theta_by_matrix": {},
    })
    by_matrix: Dict[str, Any] = {}
    for name in sorted({rows[i]["matrix"] for i in idx}):
        sub = np.asarray([i for i in idx if rows[i]["matrix"] == name], dtype=np.int64)
        pos = np.isin(idx, sub)
        by_matrix[name] = {
            "alt_raw_threshold": _f(float(np.median(
                cal["theta"][sub] * cal["alt"]["null_median"][sub]
            ))),
            "band_contrast": _f(_ratio(
                float(np.median(T["alt"][pos])), float(np.median(T["dc"][pos]))
            )),
            "frac_alt": _f(float(np.mean(exceed["alt"][pos]))),
            "frac_dc": _f(float(np.mean(exceed["dc"][pos]))),
            "n_probes": int(pos.sum()),
            "theta": _f(float(np.median(cal["theta"][sub]))),
        }
    out["theta_by_matrix"] = by_matrix
    return out


# ------------------------------------------------------------------ P2


def p2_frame_gain(rows, table, cal, bank_idx: Sequence[int], args) -> Dict[str, Any]:
    """prereg 4 P2: ``frame_gain`` / ``bulk_gain``, both null-calibrated.

    Both sides are divided by a null at THEIR own ``(phi_hat, n, burn_in,
    max_lag)``, which is what makes the ratio a frame effect rather than a
    window-length effect: tracked segments contribute n = 45 post-burn-in and
    frozen probes n = 195, so raw |t| differs for that reason alone.
    """
    idx = np.asarray(list(bank_idx), dtype=np.int64)
    out: Dict[str, Any] = {"available": False}
    if idx.size == 0:
        out["reason"] = "no probes in the core pool"
        return out
    parts: Dict[str, np.ndarray] = {}
    for tier, kind in (("frozen", "frozen"), ("tracked", "top"), ("tracked", "bulk")):
        sel = np.asarray(
            [i for i in idx if rows[i]["tier"] == tier and rows[i]["kind"] == kind],
            dtype=np.int64,
        )
        parts[kind] = sel
    if parts["frozen"].size == 0 or parts["top"].size == 0:
        out["reason"] = "P2 needs both the frozen tier and tracked `top` segments"
        out["n_by_kind"] = {k: int(v.size) for k, v in sorted(parts.items())}
        return out
    boots = {}
    medians = {}
    for kind, sel in parts.items():
        if sel.size == 0:
            boots[kind], medians[kind] = None, None
            continue
        block = args.bootstrap_block if kind == "frozen" else args.bootstrap_block_tracked
        vals = cal["dc"]["T"][sel]
        boots[kind] = _bootstrap(vals, int(block), args)
        v = vals[np.isfinite(vals)]
        medians[kind] = float(np.median(v)) if v.size else None
    frame = _ratio(medians.get("top"), medians.get("frozen"))
    bulk = _ratio(medians.get("bulk"), medians.get("frozen"))
    gain_hi = frame is not None and frame >= REGISTERED["p2_frame_gain_pass"]
    if bulk is None:
        row = None
    elif bulk <= REGISTERED["p2_bulk_tracks_ceiling"]:
        row = "A" if gain_hi else "D"
    elif bulk >= REGISTERED["p2_bulk_elevated_floor"]:
        row = "B" if gain_hi else "E"
    else:
        row = "C" if gain_hi else "F"
    labels = {
        "A": "gain, bulk frozen", "B": "gain, bulk elevated",
        "C": "gain, bulk ambiguous", "D": "no gain",
        "E": "no gain, bulk elevated", "F": "no gain, bulk ambiguous",
    }
    by_lr: Dict[str, Any] = {}
    for lr in sorted({rows[i]["lr"] for i in idx if rows[i]["lr"] is not None}):
        med = {}
        for kind, sel in parts.items():
            sub = [i for i in sel if rows[i]["lr"] == lr]
            v = cal["dc"]["T"][np.asarray(sub, dtype=np.int64)] if sub else np.array([])
            v = v[np.isfinite(v)]
            med[kind] = float(np.median(v)) if v.size else None
        by_lr["%g" % lr] = {
            "bulk_gain": _f(_ratio(med.get("bulk"), med.get("frozen"))),
            "frame_gain": _f(_ratio(med.get("top"), med.get("frozen"))),
            "median_T_dc": {k: _f(v) for k, v in sorted(med.items())},
        }
    out.update({
        "available": True,
        "bulk_gain": _f(bulk),
        "bulk_gain_ci95": (_ratio_ci(boots.get("bulk"), boots.get("frozen")) or {}).get("ci95"),
        "by_lr": by_lr,
        "frame_gain": _f(frame),
        "frame_gain_ci95": (_ratio_ci(boots.get("top"), boots.get("frozen")) or {}).get("ci95"),
        "median_T_dc": {k: _f(v) for k, v in sorted(medians.items())},
        "n_by_kind": {k: int(v.size) for k, v in sorted(parts.items())},
        "outcome_row": row,
        "outcome_row_label": None if row is None else labels[row],
        "reason": None,
    })
    return out


# ------------------------------------------------------------------ P3


def _tau_from_rho(rho: np.ndarray, k: int) -> np.ndarray:
    """``tau_p(K) = 1 + 2 sum_{j<=K} rho_j(p)`` on the bias-corrected ladder."""
    return 1.0 + 2.0 * np.sum(rho[:, :k], axis=1)


_TAU_REF_LADDERS: Dict[Tuple[Any, ...], np.ndarray] = {}


def _tau_reference_ladder(
    phi: float, n_raw: int, burn_in: int, probes: int, seed: int, max_lag: int
) -> np.ndarray:
    """The reference streams' bias-corrected ladders, computed once per key.

    The ladder does not depend on the truncation K, so the four registered K
    share one draw; the cache key carries every parameter the draw depends on,
    so a hit is bit-identical to a recomputation and the result stays a pure
    function of (phi, n, burn_in, probes, seed, max_lag).
    """
    key = (round(float(phi), 6), int(n_raw), int(burn_in), int(probes), int(seed), int(max_lag))
    if key not in _TAU_REF_LADDERS:
        rng = np.random.default_rng(int(seed))
        if float(phi) == 0.0:
            streams = rng.standard_normal((int(probes), int(n_raw)))
        else:
            streams = ar1_streams(rng, float(phi), int(n_raw), int(probes))
        out = np.empty((int(probes), int(max_lag)), dtype=np.float64)
        for i in range(int(probes)):
            out[i] = np.asarray(
                lag_ladder(streams[i], max_lag=max_lag, burn_in=burn_in,
                           bias_correct=True)["rho"]
            )
        _TAU_REF_LADDERS[key] = out
    return _TAU_REF_LADDERS[key]


def _tau_reference(
    phi: float, n_raw: int, burn_in: int, probes: int, seed: int, max_lag: int, k: int
) -> float:
    """The same estimator on a synthetic stream at matched (n, K, burn_in, N).

    ``tau_white`` is drawn at ``--tau-reference-seed`` (4243), deliberately
    different from the 4242 used everywhere else: with a shared seed the
    synthetic white-noise control would return ``tau_cal == 1.000`` by
    construction and would test nothing (prereg 5.9, the bbp-prereg A2 failure
    mode).  Mean-pooled, because a 32-lag sum is right-skewed and its median is
    biased ~12% low (prereg 5.8, repair R2).
    """
    rho = _tau_reference_ladder(phi, n_raw, burn_in, probes, seed, max_lag)
    return float(np.mean(1.0 + 2.0 * np.nansum(rho[:, :int(k)], axis=1)))


def p3_tau(rows, table, cal, idx: Sequence[int], args, burn_in: int) -> Dict[str, Any]:
    """prereg 4 P3: tau, its white and AR(1) references, and the K ladder.

    ``tau_cal(K) = tau_hat(K) / tau_white(K)``, MEAN-pooled over probes, with
    the block-bootstrap interval at every K in {8, 16, 32, 64}; a verdict is
    registered only if it is identical at all four (repair R2's K-stability
    requirement), otherwise the row is UNDECIDED.
    """
    idx = np.asarray(list(idx), dtype=np.int64)
    out: Dict[str, Any] = {"available": False, "burn_in": int(burn_in)}
    if idx.size < 2:
        out["reason"] = "no frozen probes in the core pool"
        return out
    lengths = sorted({int(table["n_raw"][i]) for i in idx})
    phi_hat = phi_hat_of(table["rho"][idx, 0])
    tau_lags = tuple(int(k) for k in args.tau_lags)
    if max(tau_lags) > int(args.max_lag):
        raise SystemExit(
            f"--tau-lags {tau_lags} exceeds --max-lag {args.max_lag}; the "
            "registered ladder is 64 and K = 64 is a registered truncation"
        )
    by_k: Dict[str, Any] = {}
    verdicts = []
    for k in tau_lags:
        tau_p = _tau_from_rho(table["rho"][idx], k)
        white = {
            n: _tau_reference(0.0, n, burn_in, int(np.sum(table["n_raw"][idx] == n)),
                              args.tau_reference_seed, int(args.max_lag), k)
            for n in lengths
        }
        ar1 = {
            n: _tau_reference(phi_hat or 0.0, n, burn_in,
                              int(np.sum(table["n_raw"][idx] == n)),
                              args.null_seed, int(args.max_lag), k)
            for n in lengths
        } if phi_hat is not None else {n: float("nan") for n in lengths}
        ref = np.asarray([white[int(table["n_raw"][i])] for i in idx], dtype=np.float64)
        tau_cal_p = tau_p / ref
        ci = _bootstrap(tau_cal_p, int(args.bootstrap_block), args, statistic=np.mean)
        weights = np.asarray([np.sum(table["n_raw"][idx] == n) for n in lengths], dtype=np.float64)
        tau_white = float(np.average([white[n] for n in lengths], weights=weights))
        tau_ar1 = float(np.average([ar1[n] for n in lengths], weights=weights))
        finite = np.isfinite(tau_p)
        tau_hat = float(np.mean(tau_p[finite])) if finite.any() else float("nan")
        if ci is None:
            verdict = None
        elif ci["ci_hi"] < REGISTERED["p3_decisive_upper"]:
            verdict = "DECISIVE"
        elif ci["ci_lo"] > REGISTERED["p3_decisive_upper"]:
            verdict = "FAIL"
        else:
            verdict = "UNDECIDED"
        verdicts.append(verdict)
        by_k["K%d" % k] = {
            "tau_ar1": _f(tau_ar1),
            "tau_cal": _f(None if ci is None else ci["point"]),
            "tau_cal_ci95": _ci(ci),
            "tau_hat": _f(tau_hat),
            "tau_white": _f(tau_white),
            "verdict": verdict,
        }
    primary = "K%d" % int(args.tau_primary_k)
    k_stable = len({v for v in verdicts}) == 1 and verdicts[0] is not None
    consistency = _ratio(by_k[primary]["tau_hat"], by_k[primary]["tau_ar1"])
    lo, hi = REGISTERED["p3_consistency_band"]
    out.update({
        "available": True,
        "by_K": by_k,
        "consistency_band": [lo, hi],
        "consistency_holds": None if consistency is None else bool(lo <= consistency <= hi),
        "consistency_ratio": _f(consistency),
        "k_stable": bool(k_stable),
        "n_probes": int(idx.size),
        "phi_hat": _f(phi_hat, 4),
        "primary_K": int(args.tau_primary_k),
        "series_lengths": lengths,
        "tau_ar1": by_k[primary]["tau_ar1"],
        "tau_cal": by_k[primary]["tau_cal"],
        "tau_cal_ci95": by_k[primary]["tau_cal_ci95"],
        "tau_hat": by_k[primary]["tau_hat"],
        "tau_white": by_k[primary]["tau_white"],
        "verdict": by_k[primary]["verdict"] if k_stable else "UNDECIDED",
        "verdict_by_K": {"K%d" % k: v for k, v in zip(tau_lags, verdicts)},
        "verdict_reason": None if k_stable else "the four K do not agree (repair R2)",
    })
    return out


# --------------------------------------------------------------- the riders


def riders(rows, table, cal, idx: Sequence[int], args) -> Dict[str, Any]:
    """Rider-1 (DC excess vs B) and Rider-2 (phi_hat / ESS-per-n vs B).

    Rider-1's vacuity guard is the ``bbp-prereg.md`` A2 lesson made mechanical:
    a ratio whose denominator is pinned by construction measures nothing, so
    ``excess_dc(500) < 0.05`` is reported as "excess unmeasurable at B = 500"
    and no ratio is printed.
    """
    idx = [i for i in idx if rows[i]["tier"] == "frozen"]
    out: Dict[str, Any] = {"available": False}
    if not idx:
        out["reason"] = "no rider-pool frozen probes (seeds 1320/1321)"
        return out
    batches = sorted({rows[i]["batch"] for i in idx if rows[i]["batch"] is not None})
    excess: Dict[int, Optional[float]] = {}
    ess_over_n: Dict[int, Optional[float]] = {}
    phi_by_batch: Dict[int, Optional[float]] = {}
    detail: Dict[str, Any] = {}
    for b in batches:
        sel = np.asarray([i for i in idx if rows[i]["batch"] == b], dtype=np.int64)
        T = cal["dc"]["T"][sel]
        T = T[np.isfinite(T)]
        med = float(np.median(T)) if T.size else None
        excess[b] = None if med is None else med - 1.0
        ess = table["dc"]["ess_over_n"][sel]
        ess = ess[np.isfinite(ess)]
        ess_over_n[b] = float(np.median(ess)) if ess.size else None
        phi_by_batch[b] = phi_hat_of(table["rho"][sel, 0])
        boot = _bootstrap(cal["dc"]["T"][sel], int(args.bootstrap_block), args)
        detail[str(b)] = {
            "excess_dc": _f(excess[b]),
            "ess_over_n_median": _f(ess_over_n[b]),
            "median_T_dc": _f(med),
            "median_T_dc_ci95": _ci(boot),
            "n_kept_median": _f(float(np.median(table["n_kept"][sel]))),
            "n_probes": int(sel.size),
            "nyquist_is_epoch_harmonic": bool(b == NYQUIST_EPOCH_HARMONIC_BATCH),
            "phi_hat": _f(phi_by_batch[b], 4),
        }
    lo_b, hi_b = (min(batches), max(batches)) if batches else (None, None)
    guard = (
        lo_b is None
        or excess.get(lo_b) is None
        or excess[lo_b] < REGISTERED["rider1_vacuity_guard"]
    )
    ratio = None if guard else _ratio(excess.get(hi_b), excess.get(lo_b))
    band = REGISTERED["rider1_pass_band"]
    if guard:
        r1 = "GUARD_FIRED"
    elif ratio is None:
        r1 = None
    elif band[0] <= ratio <= band[1]:
        r1 = "PASS"
    elif ratio < REGISTERED["rider1_fail_flat"]:
        r1 = "FAIL_FLAT"
    else:
        r1 = "AMBIGUOUS"
    vals = [ess_over_n[b] for b in batches if ess_over_n.get(b) is not None]
    spread = _ratio(max(vals), min(vals)) if vals else None
    monotone = len(vals) == len(batches) and all(
        vals[i] > vals[i + 1] for i in range(len(vals) - 1)
    )
    top = ess_over_n.get(hi_b)
    near_one = top is not None and top > 0 and max(top, 1.0 / top) <= REGISTERED["rider2_sampling_ess_tolerance"]
    if spread is not None and spread < REGISTERED["rider2_invariance_max_over_min"]:
        r2 = "B_INVARIANT"
    elif monotone and near_one:
        r2 = "SAMPLING_CONSISTENT"
    else:
        r2 = "MIXED"
    out.update({
        "available": True,
        "batches": [int(b) for b in batches],
        "ess_over_n_by_batch": {str(b): _f(ess_over_n[b]) for b in batches},
        "ess_over_n_max_over_min": _f(spread),
        "excess_by_batch": {str(b): _f(excess[b]) for b in batches},
        "phi_by_batch": {str(b): _f(phi_by_batch[b], 4) for b in batches},
        "ratio": _f(ratio),
        "rider1_branch": r1,
        "rider1_pass_band": list(band),
        "rider2_branch": r2,
        "rider2_ess_monotone_decreasing": bool(monotone),
        "vacuity_guard_fired": bool(guard),
        "vacuity_guard_note": (
            "excess unmeasurable at B = %s" % (lo_b,) if guard else None
        ),
        "by_batch": detail,
    })
    return out


# ------------------------------------------------------------- diagnostics


def channel_shape(cal, idx: Sequence[int]) -> Dict[str, Any]:
    """K6: the two channels' calibrated quantile profiles (repair R1).

    P1's contrast assumes the instrument's inflation is channel-common.  Under
    that assumption the profiles ``q(T_c)/median(T_c)`` coincide; under the
    AR(1) null they already do to within ~1% through q90.  K6 fires when q25,
    q75 or q90 differ between the channels by more than the registered 15%, and
    the reading is that P1 is reported UNREAD.

    DISCLOSED, and flagged for the human rather than repaired here (changing
    the clause would be an amendment, prereg 7): the diagnostic is calibrated
    under the NULL only, so it also fires under the alternative.  A homogeneous
    planted alternating mean is a location shift of |t_alt|, which COMPRESSES
    that channel's quantile-over-median profile; measured on this pipeline, a
    plant of A = 0.25 (band contrast 3.3, deep inside P1's PASS row) drives the
    divergence to 0.48 and trips K6, while every weaker plant (A <= 0.10, and
    the sparse plants that produce rows 3 and 5) leaves it at 0.05-0.09 and
    does not.  So on a STRONG homogeneous signal K6 and P1a point in opposite
    directions by construction.  Both numbers are reported, ``p1.read`` records
    the clause, and the P1 row is still computed and printed so the human sees
    what was suppressed.
    """
    idx = np.asarray(list(idx), dtype=np.int64)
    out: Dict[str, Any] = {"available": False}
    if idx.size < 4:
        out["reason"] = "too few probes for a quantile profile"
        return out
    prof: Dict[str, Dict[str, Optional[float]]] = {}
    for ch in CHANNELS:
        v = cal[ch]["T"][idx]
        v = v[np.isfinite(v)]
        if v.size < 4:
            out["reason"] = "channel %s has no calibrated probes" % ch
            return out
        q = np.quantile(v, QUANTILES) / np.median(v)
        prof[ch] = {"q25": _f(q[0]), "q50": _f(q[1]), "q75": _f(q[2]), "q90": _f(q[3])}
    div = {
        k: _f(abs((prof["alt"][k] / prof["dc"][k]) - 1.0))
        for k in ("q25", "q75", "q90")
    }
    worst = max(v for v in div.values() if v is not None)
    return {
        "alt": prof["alt"],
        "available": True,
        "bar": REGISTERED["k6_channel_shape_divergence"],
        "dc": prof["dc"],
        "n_probes": int(idx.size),
        "divergence": div,
        "k6_fires": bool(worst > REGISTERED["k6_channel_shape_divergence"]),
        "max_divergence": _f(worst),
        "note": (
            "calibrated under the null only: a strong homogeneous alternating "
            "signal also compresses the alt profile and trips this clause, so "
            "a fire is 'the contrast has no null value of 1 HERE', not "
            "'there is no signal'. The P1 row is printed either way. The 15% "
            "bar is ~4 sd of this profile at the registered 3456-probe pool "
            "and only ~1.3 sd at a few hundred probes, so read n_probes with it"
        ),
    }


def nw_floored_diagnostic(rows, table, idx: Sequence[int]) -> Dict[str, Any]:
    """K2: the Newey-West flooring rate, per channel and per pool.

    Registered as expected-inert on dc (published 0/864) and live on alt and
    the rider rungs; a rate above 5% makes that channel's criteria unread.
    """
    idx = np.asarray(list(idx), dtype=np.int64)
    out: Dict[str, Any] = {"bar": REGISTERED["k2_nw_floored_frac"], "by_channel": {}}
    fires = False
    for ch in CHANNELS:
        if idx.size:
            flag = table[ch]["nw_floored"][idx]
            n_fl, frac = int(flag.sum()), float(np.mean(flag))
        else:
            n_fl, frac = 0, None
        ch_fires = frac is not None and frac > REGISTERED["k2_nw_floored_frac"]
        fires = fires or ch_fires
        out["by_channel"][ch] = {
            "frac": _f(frac), "k2_fires": bool(ch_fires), "n_floored": n_fl,
            "n_probes": int(idx.size),
        }
    out["k2_fires"] = bool(fires)
    return out


def phi_diagnostic(cal, table, rows, idx: Sequence[int]) -> Dict[str, Any]:
    """K3: the per-matrix phi_hat window and spread (repair R7)."""
    idx = np.asarray(list(idx), dtype=np.int64)
    by_matrix: Dict[str, Optional[float]] = {}
    for name in sorted({rows[i]["matrix"] for i in idx}):
        sel = np.asarray([i for i in idx if rows[i]["matrix"] == name], dtype=np.int64)
        by_matrix[name] = phi_hat_of(table["rho"][sel, 0])
    vals = [v for v in by_matrix.values() if v is not None]
    lo, hi = REGISTERED["k3_phi_window"]
    spread = (max(vals) - min(vals)) if vals else None
    fires = bool(
        vals and (any(v < lo or v > hi for v in vals)
                  or (spread is not None and spread > REGISTERED["k3_phi_spread_max"]))
    )
    return {
        "by_matrix": {k: _f(v, 4) for k, v in sorted(by_matrix.items())},
        "k3_fires": fires,
        "phi_spread": _f(spread, 4),
        "spread_bar": REGISTERED["k3_phi_spread_max"],
        "window": list(REGISTERED["k3_phi_window"]),
    }


def mirror_check(bank: SeriesBank, budget: int) -> Dict[str, Any]:
    """The logged accumulator statistic vs ``channel_t`` on the same series.

    prereg 5.1's mirror contract says ``spectral.channel_t`` reproduces
    ``FrozenProbeAccumulator.stats`` exactly.  The sidecar carries the
    accumulator's own final ``t_nw`` / ``ess`` over the full (burn-in 0) series,
    so the contract is checkable on real Phase B data rather than only in the
    module's unit tests.  A deterministic evenly spaced sample is used so the
    check costs O(budget), not O(probes).
    """
    finals = bank.logged_finals
    out: Dict[str, Any] = {"max_abs_dev_ess": None, "max_abs_dev_t_nw": None, "n_checked": 0}
    if not finals or budget <= 0:
        return out
    step = max(1, len(finals) // int(budget))
    picked = finals[::step][: int(budget)]
    dev_t, dev_e = [], []
    for entry in picked:
        st = channel_t(entry["series"], "dc", 0, max_lag=entry["max_lag"])
        logged = entry["logged"]
        if np.isfinite(st["t_nw"]) and logged.get("t_nw") is not None:
            dev_t.append(abs(float(st["t_nw"]) - float(logged["t_nw"])))
        if np.isfinite(st["ess"]) and logged.get("ess") is not None:
            dev_e.append(abs(float(st["ess"]) - float(logged["ess"])))
    out["n_checked"] = len(picked)
    out["max_abs_dev_t_nw"] = _f(max(dev_t), 12) if dev_t else None
    out["max_abs_dev_ess"] = _f(max(dev_e), 12) if dev_e else None
    return out


# ------------------------------------------------------ synthetic controls


def synthetic_entries(
    kind: str,
    *,
    probes: int,
    matrices: int,
    runs: int,
    n_steps: int,
    seed: int,
    phi: float = PHASE_A_PHI,
    tracked: int = 0,
    t_refresh: int = 50,
    batch: int = CORE_BATCH,
    seeds: Sequence[int] = CORE_SEEDS,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Sidecar-shaped synthetic runs, for K1's controls and for the test suite.

    ``kind='white'`` draws iid N(0, 1); ``kind='ar1'`` draws the registered
    zero-mean AR(1) at ``phi`` from ``spectral.ar1_streams`` -- the tested
    generator, never a re-implementation of the recursion, so a control cannot
    drift away from the null it calibrates (the module's own contract).

    The streams carry NOTHING to detect in either channel, which is the point:
    K1 requires the pipeline to return ``band_contrast`` and ``tail_contrast``
    within 1.00 +/- 0.10 and ``tau_cal`` within 1.00 +/- 0.10 at every K on
    white noise, and the AR(1) control to return both contrasts within
    1.00 +/- 0.10.  A pipeline that cannot return 1 on its own null cannot
    report a number that means anything.
    """
    rng = np.random.default_rng(int(seed))
    entries = []
    for r in range(int(runs)):
        run_seed = int(seeds[r % len(seeds)])
        run = "synthetic_%s_seed%d_%02d" % (kind, run_seed, r)
        matrices_log: Dict[str, Any] = {}
        for m in range(int(matrices)):
            name = "synthetic.matrix%d.weight" % m
            if kind == "white":
                X = rng.standard_normal((int(probes), int(n_steps)))
            else:
                X = ar1_streams(rng, float(phi), int(n_steps), int(probes))
            steps = list(range(1, int(n_steps) + 1))
            block = {
                "decimate": 1,
                "k3": int(probes),
                "lag_truncation": newey_west_bandwidth(int(n_steps), NW_MAX_LAG),
                "max_lag": NW_MAX_LAG,
                "n_observations": int(n_steps),
                "probes": [
                    {"ess": [], "final": {}, "index": j, "mean": [], "s": [float(v) for v in X[j]],
                     "t_naive": [], "t_nw": [], "var": []}
                    for j in range(int(probes))
                ],
                "raw_steps": steps,
                "snapshot_steps": [],
            }
            directions = []
            if tracked:
                if kind == "white":
                    Y = rng.standard_normal((int(tracked), int(n_steps)))
                else:
                    Y = ar1_streams(rng, float(phi), int(n_steps), int(tracked))
                for j in range(int(tracked)):
                    directions.append({
                        "index": j,
                        "kind": "top" if j < int(tracked) // 2 else "bulk",
                        "lambda_hvp": {"step": [], "value": []},
                        "per_beta": {},
                        "refresh_alignment": {"step": [], "value": []},
                        "reset_steps": [],
                        "s": [float(v) for v in Y[j]],
                        "sigma": {"step": [], "value": []},
                    })
            matrices_log[name] = {
                "align_min": 0.9,
                "directions": directions,
                "frozen_probes": block,
                "grad_fro_norm": [1.0] * int(n_steps),
                "k1": int(tracked) // 2,
                "k2": int(tracked) - int(tracked) // 2,
                "refresh_steps": list(range(1, int(n_steps) + 1, int(t_refresh))),
                "shape": [8, 8],
                "snapshot_every": 5,
                "steps": steps,
                "t_refresh": int(t_refresh),
                "top_sigma_m": [1.0] * int(n_steps),
            }
        log = {
            "betas": [],
            "frozen_probes_enabled": True,
            "hvp_enabled": False,
            "instrumentation_schema_version": 2,
            "matrices": matrices_log,
        }
        meta = {
            "batch_size": int(batch), "lr": 0.24, "problem": None, "run": run,
            "seed": run_seed, "sidecar": run + SIDECAR_SUFFIX,
        }
        entries.append((meta, log))
    return entries


def k1_controls(args, nulls_reps: int) -> Dict[str, Any]:
    """K1 as run: the same pipeline, on white noise and on AR(1) phi = -0.34.

    Reported next to K1's registered tolerances; whether the clause fires is a
    HUMAN reading, so the block states the numbers and the mechanical
    comparison, not a decision.
    """
    out: Dict[str, Any] = {
        "note": (
            "prereg 7 K1; adjudication is HUMAN. `contrasts_within_tolerance` "
            "reads band_contrast at the registered statistic and the tail "
            "contrast at the null's q75: at the registered theta a pure null "
            "holds ~0.3 DC exceedance events per 3456 probes, so the "
            "registered denominator guard fires by construction and no "
            "feasible control pool can estimate that ratio (see "
            "TAIL_PROFILE_QUANTILES)"
        ),
        "probes": int(args.control_probes),
        "tolerance": 0.10,
    }
    for label, kind, phi in (("ar1_phi_m0.34", "ar1", PHASE_A_PHI), ("white", "white", 0.0)):
        entries = synthetic_entries(
            kind, probes=max(int(args.control_probes) // 2, 2), matrices=2, runs=1,
            n_steps=int(args.control_steps), seed=int(args.control_seed), phi=phi,
        )
        bank = ingest(entries)
        table = probe_table(bank.rows, int(args.primary_burn_in), int(args.max_lag))
        nulls = NullBank(nulls_reps, int(args.null_seed))
        cal = calibrate(bank.rows, table, nulls, int(args.primary_burn_in))
        idx = [i for i, r in enumerate(bank.rows) if r["tier"] == "frozen"]
        p1 = p1_band_contrast(bank.rows, table, cal, idx, args, label=label)
        p3 = p3_tau(bank.rows, table, cal, idx, args, int(args.primary_burn_in))
        profile = p1.get("tail_contrast_profile") or {}
        tail_est = (profile.get("q%d" % int(K1_TAIL_QUANTILE * 100)) or {}).get("tail_contrast")
        contrasts_ok = all(
            v is not None and abs(v - 1.0) <= 0.10
            for v in (p1.get("band_contrast"), tail_est)
        )
        tau_ok = all(
            e["tau_cal"] is not None and abs(e["tau_cal"] - 1.0) <= 0.10
            for e in p3.get("by_K", {}).values()
        ) if p3.get("available") else False
        out[label] = {
            "band_contrast": p1.get("band_contrast"),
            "contrasts_within_tolerance": bool(contrasts_ok),
            "outcome_row": p1.get("outcome_row"),
            "phi": _f(phi, 4),
            "tail_contrast": p1.get("tail_contrast"),
            "tail_contrast_at_null_q%d" % int(K1_TAIL_QUANTILE * 100): tail_est,
            "tail_contrast_denominator_guard": p1.get("tail_contrast_denominator_guard"),
            "tail_contrast_profile": profile,
            "tau_cal_by_K": {k: v["tau_cal"] for k, v in sorted(p3.get("by_K", {}).items())},
            "tau_cal_within_tolerance_every_K": bool(tau_ok) if label == "white" else None,
            "tau_verdict": p3.get("verdict"),
        }
    return out


# ---------------------------------------------------------------- analysis


def analyze_at(
    bank: SeriesBank, table: Dict[str, Any], burn_in: int, nulls: NullBank, args
) -> Dict[str, Any]:
    """Every registered quantity at one burn-in."""
    rows = bank.rows
    cal = calibrate(rows, table, nulls, burn_in)
    core_frozen = [i for i, r in enumerate(rows) if r["pool"] == "core" and r["tier"] == "frozen"]
    core_all = [i for i, r in enumerate(rows) if r["pool"] == "core"]
    rider_all = [i for i, r in enumerate(rows) if r["pool"] == "rider"]

    p1 = p1_band_contrast(rows, table, cal, core_frozen, args, label="core B=2000")
    by_batch: Dict[str, Any] = {}
    for b in sorted({rows[i]["batch"] for i in rider_all if rows[i]["batch"] is not None}):
        sel = [i for i in rider_all if rows[i]["batch"] == b and rows[i]["tier"] == "frozen"]
        entry = p1_band_contrast(rows, table, cal, sel, args, label="rider B=%s" % b)
        entry["criterion"] = False
        entry["nyquist_is_epoch_harmonic"] = bool(b == NYQUIST_EPOCH_HARMONIC_BATCH)
        entry["note"] = (
            "DESCRIPTIVE EXTENSION ONLY (prereg 3, repair R8): the rider rungs "
            "cannot create or destroy a P1 verdict"
        )
        by_batch[str(b)] = entry
    p1["by_batch"] = by_batch

    shape = channel_shape(cal, core_frozen)
    p1["k6_fires"] = shape.get("k6_fires")
    p1["read"] = not bool(shape.get("k6_fires"))
    p1["unread_reason"] = (
        "K6 fired, so P1 is reported UNREAD: the two channels are not the same "
        "instrument at different scales and the contrast has no null value of 1 "
        "(prereg 7 K6). The row below is still printed; see "
        "diagnostics.channel_shape.note for what a fire does and does not mean"
        if shape.get("k6_fires") else None
    )

    cells: Dict[str, Any] = {}
    for i, r in enumerate(rows):
        key = "%s/%s/b%s/lr%s" % (r["tier"], r["matrix"], r["batch"], r["lr"])
        cells.setdefault(key, []).append(i)
    cell_out: Dict[str, Any] = {}
    for key, members in sorted(cells.items()):
        sel = np.asarray(members, dtype=np.int64)
        cell_out[key] = {
            "median_T_alt": _f(float(np.nanmedian(cal["alt"]["T"][sel]))),
            "median_T_dc": _f(float(np.nanmedian(cal["dc"]["T"][sel]))),
            "median_abs_t_alt": _f(float(np.nanmedian(table["alt"]["abs_t"][sel]))),
            "median_abs_t_dc": _f(float(np.nanmedian(table["dc"]["abs_t"][sel]))),
            "median_ess_over_n_dc": _f(float(np.nanmedian(table["dc"]["ess_over_n"][sel]))),
            "n_kept_median": _f(float(np.median(table["n_kept"][sel]))),
            "n_series": int(sel.size),
            "phi_hat": _f(phi_hat_of(table["rho"][sel, 0]), 4),
            "pool": rows[members[0]]["pool"],
            "rho_1_raw": _f(phi_hat_of(table["rho_raw"][sel, 0]), 4),
        }

    null_out: Dict[str, Any] = {}
    for key, phi in sorted(cal["_phi_by_group_raw"].items(), key=lambda kv: str(kv[0])):
        tier, matrix, batch = key
        lengths = sorted({
            int(table["n_raw"][i]) for i, r in enumerate(rows)
            if (r["tier"], r["matrix"], r["batch"]) == key
        })
        for ch in CHANNELS:
            for n_raw in lengths:
                d = nulls.describe(phi, n_raw, burn_in, ch)
                if d is None:
                    continue
                null_out.setdefault("%s/%s/b%s" % (tier, matrix, batch), {}).setdefault(
                    ch, {}
                )[str(n_raw - burn_in)] = d

    frozen_all = [i for i, r in enumerate(rows) if r["tier"] == "frozen"]
    ladder: Dict[str, Any] = {}
    for tier in TIERS:
        sel = np.asarray([i for i, r in enumerate(rows) if r["tier"] == tier], dtype=np.int64)
        if sel.size == 0:
            continue
        rho = table["rho"][sel]
        rho_raw = table["rho_raw"][sel]
        ladder[tier] = {
            "n_series": int(sel.size),
            "q25": [_f(v) for v in _nan_reduce(rho, lambda a, axis: np.nanquantile(a, 0.25, axis=axis))],
            "q75": [_f(v) for v in _nan_reduce(rho, lambda a, axis: np.nanquantile(a, 0.75, axis=axis))],
            "rho": [_f(v) for v in _nan_reduce(rho, np.nanmedian)],
            "rho_raw": [_f(v) for v in _nan_reduce(rho_raw, np.nanmedian)],
        }

    channels_out: Dict[str, Any] = {}
    for ch in CHANNELS:
        sel = np.asarray(core_frozen or frozen_all, dtype=np.int64)
        channels_out[ch] = {
            "T": _summary(cal[ch]["T"][sel]),
            "ess": _summary(table[ch]["ess"][sel]),
            "ess_over_n": _summary(table[ch]["ess_over_n"][sel]),
            "n_nw_floored": int(table[ch]["nw_floored"][sel].sum()),
            "nw_floored": {
                "frac": _f(float(np.mean(table[ch]["nw_floored"][sel]))) if sel.size else None,
                "n": int(table[ch]["nw_floored"][sel].sum()),
                "n_probes": int(sel.size),
            },
            "t_nw": _summary(table[ch]["t_nw"][sel]),
            "abs_t_nw": _summary(table[ch]["abs_t"][sel]),
        }

    tier_contrast: Dict[str, Any] = {"calibrated_nw_median_T_dc": {}, "raw_final_abs_t_stat": {}}
    for beta, by_kind in sorted(bank.tier_contrast_raw.items()):
        tier_contrast["raw_final_abs_t_stat"][beta] = {
            k: _f(float(np.median(v))) for k, v in sorted(by_kind.items())
        }
    for kind in ("bulk", "frozen", "top"):
        sel = np.asarray([i for i, r in enumerate(rows) if r["kind"] == kind], dtype=np.int64)
        if sel.size:
            tier_contrast["calibrated_nw_median_T_dc"][kind] = _f(
                float(np.nanmedian(cal["dc"]["T"][sel]))
            )
    tier_contrast["note"] = (
        "the raw block is the published per-beta EMA t_stat (an EMA statistic, "
        "not this program's Newey-West one); the calibrated block is this "
        "program's null-calibrated T_dc by tier, which is the comparison "
        "prereg 4 asks for ('the object A5 showed was never calibrated')"
    )

    ess = table["dc"]["ess_over_n"][np.asarray(core_frozen or frozen_all, dtype=np.int64)]
    descriptive = {
        "by_kind": {
            kind: {
                "median_T_alt": _f(float(np.nanmedian(cal["alt"]["T"][sel]))),
                "median_T_dc": _f(float(np.nanmedian(cal["dc"]["T"][sel]))),
                "n_series": int(len(sel)),
            }
            for kind in sorted({r["kind"] for r in rows})
            for sel in [np.asarray([i for i, r in enumerate(rows) if r["kind"] == kind], dtype=np.int64)]
        },
        "by_lr": {
            "%g" % lr: {
                "median_T_alt": _f(float(np.nanmedian(cal["alt"]["T"][sel]))),
                "median_T_dc": _f(float(np.nanmedian(cal["dc"]["T"][sel]))),
                "n_series": int(len(sel)),
            }
            for lr in sorted({r["lr"] for r in rows if r["lr"] is not None})
            for sel in [np.asarray(
                [i for i, r in enumerate(rows) if r["lr"] == lr and r["tier"] == "frozen"],
                dtype=np.int64)]
            if len(sel)
        },
        "by_matrix": {
            name: {
                "median_T_alt": _f(float(np.nanmedian(cal["alt"]["T"][sel]))),
                "median_T_dc": _f(float(np.nanmedian(cal["dc"]["T"][sel]))),
                "n_series": int(len(sel)),
                "phi_hat": _f(phi_hat_of(table["rho"][sel, 0]), 4),
            }
            for name in sorted({r["matrix"] for r in rows})
            for sel in [np.asarray(
                [i for i, r in enumerate(rows) if r["matrix"] == name and r["tier"] == "frozen"],
                dtype=np.int64)]
            if len(sel)
        },
        "ess_over_n": {
            "observed": _summary(ess),
            "published_anchor": {
                "median": PUBLISHED_ANCHORS["ess_over_n_median"],
                "min": PUBLISHED_ANCHORS["ess_over_n_min"],
                "note": "PEEKED prior (prereg 0); a re-read, never a reference",
            },
        },
        "tier_contrast": tier_contrast,
    }

    core_dc_median = channels_out["dc"]["T"]["median"]
    return {
        "cells": cell_out,
        "channels": channels_out,
        "descriptive": descriptive,
        "diagnostics": {
            "channel_shape": shape,
            "k3_phi": phi_diagnostic(cal, table, rows, core_frozen or frozen_all),
            "k4_frame_gain_denominator": {
                "bar": REGISTERED["k4_frozen_median_t_dc_max"],
                "k4_fires": bool(
                    core_dc_median is not None
                    and core_dc_median > REGISTERED["k4_frozen_median_t_dc_max"]
                ),
                "median_T_dc_frozen": core_dc_median,
            },
            "n_nw_floored": nw_floored_diagnostic(rows, table, core_frozen or frozen_all),
        },
        "estimator": {
            "burn_in": int(burn_in),
            "max_lag": int(args.max_lag),
            "n_kept": _summary(table["n_kept"].astype(np.float64)),
            "n_raw": _summary(table["n_raw"].astype(np.float64)),
            "nw_bandwidth_check": nw_bandwidth_check(table["n_raw"], burn_in, args.max_lag),
            "phi_hat": {
                "by_group": cal["phi_by_group"],
                "by_matrix": phi_diagnostic(cal, table, rows, core_frozen or frozen_all)["by_matrix"],
                "ci95": _ci(_bootstrap(
                    table["rho"][np.asarray(core_frozen or frozen_all, dtype=np.int64), 0],
                    int(args.bootstrap_block), args,
                )),
                "point": _f(phi_hat_of(
                    table["rho"][np.asarray(core_frozen or frozen_all, dtype=np.int64), 0]
                ), 4),
                "rho_1_raw": _f(phi_hat_of(
                    table["rho_raw"][np.asarray(core_frozen or frozen_all, dtype=np.int64), 0]
                ), 4),
            },
            "series_parity": bank.parity,
        },
        "ladder": ladder,
        "null": null_out,
        "p1": p1,
        "p2": p2_frame_gain(rows, table, cal, core_all, args),
        "p3": p3_tau(rows, table, cal, core_frozen, args, burn_in),
        "rider": riders(rows, table, cal, rider_all, args),
    }


def compact(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The point estimates a sensitivity row needs -- no cells, no nulls."""
    p1, p2, p3, rider = analysis["p1"], analysis["p2"], analysis["p3"], analysis["rider"]
    return {
        "band_contrast": p1.get("band_contrast"),
        "bulk_gain": p2.get("bulk_gain"),
        "frac_alt": p1.get("frac_alt"),
        "frac_dc": p1.get("frac_dc"),
        "frame_gain": p2.get("frame_gain"),
        "median_T_alt": analysis["channels"]["alt"]["T"]["median"],
        "median_T_dc": analysis["channels"]["dc"]["T"]["median"],
        "p1_outcome_row": p1.get("outcome_row"),
        "p2_outcome_row": p2.get("outcome_row"),
        "p3_verdict": p3.get("verdict"),
        "phi_hat": analysis["estimator"]["phi_hat"]["point"],
        "ratio_alt": p1.get("ratio_alt"),
        "ratio_dc": p1.get("ratio_dc"),
        "rider1_branch": rider.get("rider1_branch"),
        "rider2_branch": rider.get("rider2_branch"),
        "tail_contrast": p1.get("tail_contrast"),
        "tau_cal": p3.get("tau_cal"),
    }


def phi_sensitivity(bank, table, nulls, args, burn_in: int) -> Dict[str, Any]:
    """prereg 5.5: the same nulls at phi_hat +/- 0.05 and at phi = -0.34.

    Reported on BOTH channels, always: the alternating channel's phi
    sensitivity is ~1.5x the DC channel's in the median and ~4x in the >= 4
    tail (prereg 2, repair R7), and P1 is read on alt.
    """
    rows = bank.rows
    core_frozen = [i for i, r in enumerate(rows) if r["pool"] == "core" and r["tier"] == "frozen"]
    idx = core_frozen or [i for i, r in enumerate(rows) if r["tier"] == "frozen"]
    out: Dict[str, Any] = {}
    settings: List[Tuple[str, Optional[float], float]] = [
        ("phi_hat%+.2f" % off, None, off) for off in PHI_SENSITIVITY_OFFSETS
    ]
    settings.append(("phi_fixed_%.2f" % PHASE_A_PHI, PHASE_A_PHI, 0.0))
    for label, override, offset in settings:
        cal = calibrate(rows, table, nulls, burn_in, phi_override=override, phi_offset=offset)
        p1 = p1_band_contrast(rows, table, cal, idx, args, label=label)
        out[label] = {
            "band_contrast": p1.get("band_contrast"),
            "frac_alt": p1.get("frac_alt"),
            "frac_dc": p1.get("frac_dc"),
            "outcome_row": p1.get("outcome_row"),
            "ratio_alt": p1.get("ratio_alt"),
            "ratio_dc": p1.get("ratio_dc"),
            "tail_contrast": p1.get("tail_contrast"),
            "theta": p1.get("theta"),
        }
    return out


def build_report(bank: SeriesBank, args) -> Dict[str, Any]:
    """The full section 6b output dict, deterministic and sorted."""
    burn_ins = tuple(int(b) for b in args.burn_ins)
    primary = int(args.primary_burn_in)
    if primary not in burn_ins:
        raise SystemExit(f"--primary-burn-in {primary} is not in --burn-ins {list(burn_ins)}")
    nulls = NullBank(int(args.null_reps), int(args.null_seed))
    tables = {b: probe_table(bank.rows, b, int(args.max_lag)) for b in burn_ins}
    check = nw_bandwidth_check(tables[primary]["n_raw"], primary, int(args.max_lag))
    if not check["identical"]:  # pragma: no cover - unreachable below n ~ 2260
        raise SystemExit(
            "the Newey-West bandwidth at --max-lag differs from the one at "
            f"NW_MAX_LAG={NW_MAX_LAG}: {check['by_length']}. The fast path is "
            "only exact while they agree; refusing to report."
        )
    analyses = {b: analyze_at(bank, tables[b], b, nulls, args) for b in burn_ins}
    report = dict(analyses[primary])
    report["sensitivity"] = {
        "burn_in": {str(b): compact(analyses[b]) for b in burn_ins},
        "burn_in_note": (
            "prereg 5.3: criteria are read at b = 5 and the b in {5, 15, 25} "
            "sweep is reported alongside every registered quantity; a verdict "
            "that flips across the sweep belongs in the headline"
        ),
        "phi": phi_sensitivity(bank, tables[primary], nulls, args, primary),
        "phi_note": (
            "prereg 5.5: nulls redrawn at phi_hat +/- 0.05 and at the fixed "
            "Phase A phi = -0.34, reported on BOTH channels"
        ),
    }
    report["inputs"] = {
        "n_runs": len(bank.runs),
        "n_series": len(bank.rows),
        "n_skipped": len(bank.skipped),
        "pools": {
            pool: {
                "n_frozen_probes": sum(
                    1 for r in bank.rows if r["pool"] == pool and r["tier"] == "frozen"
                ),
                "n_runs": sum(1 for r in bank.runs if r["pool"] == pool),
                "n_tracked_segments": sum(
                    1 for r in bank.rows if r["pool"] == pool and r["tier"] == "tracked"
                ),
            }
            for pool in sorted({r["pool"] for r in bank.runs} | set(POOLS))
        },
        "selection": getattr(args, "_selection", None),
        "skipped": sorted(bank.skipped, key=lambda d: str(d.get("run"))),
        "synthetic_control": args.synthetic_control,
    }
    report["runs"] = sorted(bank.runs, key=lambda d: str(d["run"]))
    report["diagnostics"]["mirror_check"] = mirror_check(bank, int(args.verify_probes))
    report["registered"] = {
        "core_seeds": list(CORE_SEEDS),
        "note": (
            "prereg 4 / Appendix. Every threshold below is a PROPOSAL until a "
            "human freezes it; this script prints quantities next to them and "
            "adjudicates nothing (CLAUDE.md ground rule 1)."
        ),
        "pooling": dict(sorted(POOLING.items())),
        "rider_seeds": list(RIDER_SEEDS),
        "status": THRESHOLD_STATUS,
        "thresholds": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in sorted(REGISTERED.items())
        },
    }
    report["settings"] = {
        "bootstrap_block": int(args.bootstrap_block),
        "bootstrap_block_tracked": int(args.bootstrap_block_tracked),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "burn_ins": list(burn_ins),
        "max_lag": int(args.max_lag),
        "null_draws": int(nulls.n_draws),
        "null_reps": int(args.null_reps),
        "null_seed": int(args.null_seed),
        "nw_max_lag": NW_MAX_LAG,
        "primary_burn_in": primary,
        "tau_lags": [int(k) for k in args.tau_lags],
        "tau_primary_k": int(args.tau_primary_k),
        "tau_reference_seed": int(args.tau_reference_seed),
    }
    if args.controls and args.synthetic_control == "none":
        report["controls"] = k1_controls(args, int(args.null_reps))
    else:
        report["controls"] = {
            "note": (
                "skipped: this run IS a synthetic control"
                if args.synthetic_control != "none" else "skipped by --no-controls"
            )
        }
    return report


# ------------------------------------------------------------------ report


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _pair(v: Optional[Sequence[Any]], nd: int = 3) -> str:
    if not v:
        return "n/a"
    return "[%s, %s]" % (_fmt(v[0], nd), _fmt(v[1], nd))


def to_markdown(report: Dict[str, Any]) -> str:
    p1, p2, p3, rider = report["p1"], report["p2"], report["p3"], report["rider"]
    th = report["registered"]["thresholds"]
    lines: List[str] = [
        "# Channel audit, Phase B - frozen tier (descriptive)",
        "",
        "Output of `scripts/analyze_channel_audit_frozen.py`, the Phase B",
        "producer registered in `reports/channel-audit-preregistration.md` 6b.",
        "",
        "**Descriptive only.** Every registered quantity is printed next to the",
        "threshold the pre-registration proposes for it, and the row of each",
        "registered outcome map that the numbers mechanically fall in is a",
        "LOOKUP, not a verdict: adjudication is HUMAN (CLAUDE.md ground rule 1).",
        "",
        f"Threshold status: **{report['registered']['status']}**.",
        "",
        "| input | value |",
        "| --- | --- |",
        f"| runs | {report['inputs']['n_runs']} |",
        f"| series (frozen probes + tracked segments) | {report['inputs']['n_series']} |",
        f"| core pool (B = 2000, seeds {CORE_SEEDS}) | "
        f"{report['inputs']['pools'].get('core', {}).get('n_frozen_probes', 0)} frozen probes |",
        f"| rider pool (seeds {RIDER_SEEDS}) | "
        f"{report['inputs']['pools'].get('rider', {}).get('n_frozen_probes', 0)} frozen probes |",
        f"| burn-in (primary / sweep) | {report['settings']['primary_burn_in']} / "
        f"{report['settings']['burn_ins']} |",
        f"| ladder max lag / NW cap | {report['settings']['max_lag']} / "
        f"{report['settings']['nw_max_lag']} (bandwidth L in "
        + str(sorted({row["L_at_max_lag"]
                      for row in report["estimator"]["nw_bandwidth_check"]["by_length"]}))
        + f", identical at both caps: "
        f"{report['estimator']['nw_bandwidth_check']['identical']}) |",
        f"| null reps / seed | {report['settings']['null_reps']} / "
        f"{report['settings']['null_seed']} ({report['settings']['null_draws']} draws) |",
        f"| tau reference seed | {report['settings']['tau_reference_seed']} |",
        f"| bootstrap block / reps | {report['settings']['bootstrap_block']} "
        f"(tracked {report['settings']['bootstrap_block_tracked']}) / "
        f"{report['settings']['bootstrap_reps']} |",
        f"| segment-start parity (prereg 1) | "
        f"{report['estimator']['series_parity']['parity']}, "
        f"{report['estimator']['series_parity']['n_odd_starts']} odd of "
        f"{report['estimator']['series_parity']['n_segment_starts']} |",
        "",
        "## P1 - band artifact (frozen tier, 9-run B = 2000 core)",
        "",
        "| quantity | value | CI95 | proposed threshold |",
        "| --- | --- | --- | --- |",
        f"| ratio_alt | {_fmt(p1.get('ratio_alt'))} | {_pair(p1.get('ratio_alt_ci95'))} | - |",
        f"| ratio_dc | {_fmt(p1.get('ratio_dc'))} | {_pair(p1.get('ratio_dc_ci95'))} | - |",
        f"| **band_contrast** | {_fmt(p1.get('band_contrast'))} | "
        f"{_pair(p1.get('band_contrast_ci95'))} | >= {th['p1_band_contrast_pass']} "
        f"(middle band from {th['p1_band_contrast_middle_edge']}) |",
        f"| theta (dc units) | {_fmt(p1.get('theta'))} | - | |t| >= {th['t_exceedance']} |",
        f"| theta, alt raw threshold | {_fmt(p1.get('theta_alt_raw_threshold'))} | - | - |",
        f"| frac_alt | {_fmt(p1.get('frac_alt'), 5)} | - | >= {th['p1_frac_alt_floor']} |",
        f"| frac_dc | {_fmt(p1.get('frac_dc'), 5)} | - | - |",
        f"| tail_contrast | {_fmt(p1.get('tail_contrast'))} | - | >= "
        f"{th['p1_tail_contrast_pass']} |",
        f"| DC exceedance events | {p1.get('n_events', {}).get('dc')} | - | guard at "
        f"{th['p1_min_dc_events']} |",
        "",
        f"Registered outcome-map row: **{p1.get('outcome_row')}** "
        f"({p1.get('outcome_row_label')}); P1 read: {p1.get('read')}"
        + (f" ({p1.get('unread_reason')})" if p1.get("unread_reason") else "")
        + ".",
        "",
        "Raw (uncalibrated) companions, printed so P1's deliberate FAIL-ward",
        "suppression stays visible: median |t_alt| "
        f"{_fmt((p1.get('raw') or {}).get('median_abs_t_alt'))}, median |t_dc| "
        f"{_fmt((p1.get('raw') or {}).get('median_abs_t_dc'))}, "
        f"frac(|t_alt| >= 4) {_fmt((p1.get('raw') or {}).get('frac_alt_abs_t_ge_4'), 5)}, "
        f"frac(|t_dc| >= 4) {_fmt((p1.get('raw') or {}).get('frac_dc_abs_t_ge_4'), 5)}.",
        "",
        "## P2 - frame gain (tracked `top` / frozen, same runs)",
        "",
        "| quantity | value | CI95 | proposed threshold |",
        "| --- | --- | --- | --- |",
        f"| frame_gain | {_fmt(p2.get('frame_gain'))} | {_pair(p2.get('frame_gain_ci95'))} "
        f"| >= {th['p2_frame_gain_pass']} |",
        f"| bulk_gain | {_fmt(p2.get('bulk_gain'))} | {_pair(p2.get('bulk_gain_ci95'))} "
        f"| <= {th['p2_bulk_tracks_ceiling']} or >= {th['p2_bulk_elevated_floor']} |",
        "",
        f"Registered outcome-map row: **{p2.get('outcome_row')}** "
        f"({p2.get('outcome_row_label')}).",
        "",
        "## P3 - integrated autocorrelation time (re-read + the K > 4 extension)",
        "",
        "| K | tau_hat | tau_white | tau_ar1 | tau_cal | CI95 | branch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, e in sorted((p3.get("by_K") or {}).items(), key=lambda kv: int(kv[0][1:])):
        lines.append(
            f"| {key[1:]} | {_fmt(e['tau_hat'], 4)} | {_fmt(e['tau_white'], 4)} | "
            f"{_fmt(e['tau_ar1'], 4)} | {_fmt(e['tau_cal'], 4)} | "
            f"{_pair(e['tau_cal_ci95'], 4)} | {e['verdict']} |"
        )
    lines += [
        "",
        f"K-stable: {p3.get('k_stable')}; branch at K = {p3.get('primary_K')}: "
        f"**{p3.get('verdict')}**"
        + (f" ({p3.get('verdict_reason')})" if p3.get("verdict_reason") else "")
        + f". Consistency clause tau_hat(K)/tau_ar1(K) = "
        f"{_fmt(p3.get('consistency_ratio'))} against the proposed band "
        f"{p3.get('consistency_band')}: {p3.get('consistency_holds')}.",
        "",
        "## Riders (secondary; batch axis, seeds 1320/1321)",
        "",
        "| B | excess_dc | median T_dc | ESS/n | phi_hat | probes | Nyquist = epoch harmonic |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b, e in sorted((rider.get("by_batch") or {}).items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {b} | {_fmt(e['excess_dc'])} | {_fmt(e['median_T_dc'])} | "
            f"{_fmt(e['ess_over_n_median'])} | {_fmt(e['phi_hat'])} | "
            f"{e['n_probes']} | {e['nyquist_is_epoch_harmonic']} |"
        )
    lines += [
        "",
        f"Rider-1 ratio {_fmt(rider.get('ratio'))} against the proposed pass band "
        f"{th['rider1_pass_band']} / flat bar {th['rider1_fail_flat']} -> "
        f"**{rider.get('rider1_branch')}**"
        + (f" ({rider.get('vacuity_guard_note')})" if rider.get("vacuity_guard_fired") else "")
        + ".",
        f"Rider-2 ESS/n max/min {_fmt(rider.get('ess_over_n_max_over_min'))} against "
        f"{th['rider2_invariance_max_over_min']} -> **{rider.get('rider2_branch')}**.",
        "",
        "## Kill-clause diagnostics (reported whether or not they fire)",
        "",
        "| clause | quantity | value | proposed bar | fires |",
        "| --- | --- | --- | --- | --- |",
    ]
    diag = report["diagnostics"]
    for ch in CHANNELS:
        e = diag["n_nw_floored"]["by_channel"][ch]
        lines.append(
            f"| K2 ({ch}) | NW-floored fraction | {_fmt(e['frac'], 5)} | "
            f"{th['k2_nw_floored_frac']} | {e['k2_fires']} |"
        )
    k3 = diag["k3_phi"]
    lines += [
        f"| K3 | per-matrix phi_hat spread | {_fmt(k3['phi_spread'], 4)} | "
        f"{th['k3_phi_spread_max']} (window {k3['window']}) | {k3['k3_fires']} |",
        f"| K4 | frozen median T_dc | {_fmt(diag['k4_frame_gain_denominator']['median_T_dc_frozen'])} "
        f"| {th['k4_frozen_median_t_dc_max']} | "
        f"{diag['k4_frame_gain_denominator']['k4_fires']} |",
        f"| K6 | max channel-shape divergence | "
        f"{_fmt(diag['channel_shape'].get('max_divergence'), 4)} | "
        f"{th['k6_channel_shape_divergence']} | {diag['channel_shape'].get('k6_fires')} |",
        "",
    ]
    controls = report.get("controls") or {}
    if "white" in controls:
        lines += [
            "## K1 - the estimator's own controls, run through this pipeline",
            "",
            "| control | band_contrast | tail_contrast (null q%d) | tau_cal by K | within 1.00 +/- 0.10 |"
            % int(K1_TAIL_QUANTILE * 100),
            "| --- | --- | --- | --- | --- |",
        ]
        for label in sorted(k for k in controls if k in ("ar1_phi_m0.34", "white")):
            c = controls[label]
            lines.append(
                f"| {label} | {_fmt(c['band_contrast'])} | "
                f"{_fmt(c['tail_contrast_at_null_q%d' % int(K1_TAIL_QUANTILE * 100)])} | "
                + ", ".join(
                    f"{k[1:]}:{_fmt(v, 3)}" for k, v in sorted(
                        c["tau_cal_by_K"].items(), key=lambda kv: int(kv[0][1:])
                    )
                )
                + f" | contrasts {c['contrasts_within_tolerance']}"
                + (f", tau {c['tau_cal_within_tolerance_every_K']}"
                   if c.get("tau_cal_within_tolerance_every_K") is not None else "")
                + " |"
            )
        lines.append("")
    lines += [
        "## Sensitivities (they can qualify a number, never create one)",
        "",
        "| burn-in | band_contrast | tail_contrast | frame_gain | tau_cal | phi_hat |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for b, e in sorted(report["sensitivity"]["burn_in"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {b} | {_fmt(e['band_contrast'])} | {_fmt(e['tail_contrast'])} | "
            f"{_fmt(e['frame_gain'])} | {_fmt(e['tau_cal'])} | {_fmt(e['phi_hat'], 4)} |"
        )
    lines += [
        "",
        "| null phi | ratio_alt | ratio_dc | band_contrast | frac_alt | frac_dc |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, e in sorted(report["sensitivity"]["phi"].items()):
        lines.append(
            f"| {label} | {_fmt(e['ratio_alt'])} | {_fmt(e['ratio_dc'])} | "
            f"{_fmt(e['band_contrast'])} | {_fmt(e['frac_alt'], 5)} | "
            f"{_fmt(e['frac_dc'], 5)} |"
        )
    lines += [
        "",
        "## Descriptive (no criteria, no thresholds)",
        "",
        "ESS/n on the core pool: median "
        f"{_fmt(report['descriptive']['ess_over_n']['observed']['median'])} "
        f"(published, PEEKED: {PUBLISHED_ANCHORS['ess_over_n_median']}); "
        "per-matrix phi_hat "
        + ", ".join(f"{_fmt(v, 3)}" for v in sorted(
            v for v in k3["by_matrix"].values() if v is not None
        ))
        + ".",
        "",
        "Mirror check (logged accumulator statistic vs `spectral.channel_t` on "
        "the same series): "
        f"{report['diagnostics']['mirror_check']['n_checked']} probes, max |dev| "
        f"t_nw {report['diagnostics']['mirror_check']['max_abs_dev_t_nw']}.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- plots


def make_figures(report: Dict[str, Any], outdir: Path, prefix: str = "channel-audit-") -> List[Path]:
    """The registered ``reports/figures/channel-audit-*.png`` set.

    Deterministic: fixed figure size, no timestamps, and PNG metadata pinned so
    two runs on the same input produce byte-identical files.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a repo dependency
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    meta = {"Software": None}

    def save(fig, name: str) -> None:
        path = outdir / f"{prefix}{name}.png"
        fig.savefig(path, dpi=140, metadata=meta)
        plt.close(fig)
        made.append(path)

    # 1. the lag ladder, corrected and raw, per tier
    ladder = report.get("ladder") or {}
    if ladder:
        fig, axes = plt.subplots(1, len(ladder), figsize=(5.0 * len(ladder), 3.4), squeeze=False)
        for ax, (tier, e) in zip(axes.ravel(), sorted(ladder.items())):
            k = np.arange(1, len(e["rho"]) + 1)
            rho = np.asarray([np.nan if v is None else v for v in e["rho"]], dtype=float)
            q25 = np.asarray([np.nan if v is None else v for v in e["q25"]], dtype=float)
            q75 = np.asarray([np.nan if v is None else v for v in e["q75"]], dtype=float)
            raw = np.asarray([np.nan if v is None else v for v in e["rho_raw"]], dtype=float)
            ax.fill_between(k, q25, q75, color="#4477aa", alpha=0.25, label="IQR")
            ax.plot(k, rho, color="#4477aa", lw=1.2, label="rho (bias-corrected)")
            ax.plot(k, raw, color="#cc3311", lw=0.9, ls="--", label="rho_raw")
            ax.axhline(0.0, color="#666666", lw=0.7)
            ax.set_title(f"{tier} tier (n = {e['n_series']})", fontsize=9)
            ax.set_xlabel("lag k")
            ax.set_ylabel("rho_k")
            ax.legend(fontsize=7)
        fig.suptitle("Lag ladder to K = 64 (median over series)", fontsize=10)
        fig.tight_layout()
        save(fig, "ladder")

    # 2. tau against its two references
    by_k = (report.get("p3") or {}).get("by_K") or {}
    if by_k:
        ks = sorted(int(k[1:]) for k in by_k)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for key, color, label in (
            ("tau_hat", "#4477aa", "tau_hat"),
            ("tau_white", "#666666", "tau_white (seed 4243)"),
            ("tau_ar1", "#cc3311", "tau_ar1(phi_hat)"),
        ):
            vals = [by_k["K%d" % k].get(key) for k in ks]
            ax.plot(ks, [np.nan if v is None else v for v in vals], "o-", color=color, label=label)
        ax.axhline(1.0, color="#000000", lw=0.7, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("truncation K")
        ax.set_ylabel("tau")
        ax.set_title("Integrated autocorrelation time and its references", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, "tau")

    # 3. the two channels' calibrated quantile profiles (K6's diagnostic)
    shape = (report.get("diagnostics") or {}).get("channel_shape") or {}
    if shape.get("available"):
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        qs = ["q25", "q50", "q75", "q90"]
        x = np.arange(len(qs))
        for ch, color in (("alt", "#cc3311"), ("dc", "#4477aa")):
            ax.plot(x, [shape[ch][q] for q in qs], "o-", color=color, label=ch)
        ax.set_xticks(x)
        ax.set_xticklabels(qs)
        ax.set_ylabel("quantile / own median")
        ax.set_title(
            "K6: channel shape (max divergence %s, bar %s)"
            % (shape.get("max_divergence"), shape.get("bar")), fontsize=9,
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, "channel-shape")

    # 4. the batch rider
    by_batch = (report.get("rider") or {}).get("by_batch") or {}
    if by_batch:
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        bs = sorted(int(b) for b in by_batch)
        ax.plot(bs, [by_batch[str(b)]["excess_dc"] for b in bs], "o-", color="#4477aa",
                label="excess_dc")
        ax.axhline(0.0, color="#666666", lw=0.7)
        ax.axhline(REGISTERED["rider1_vacuity_guard"], color="#cc3311", lw=0.8, ls="--",
                   label="vacuity guard")
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b) for b in bs])
        ax.set_xlabel("batch size")
        ax.set_ylabel("median T_dc - 1")
        ax.set_title("Rider-1: DC excess vs batch (step-matched)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, "rider")
    return made


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so a test can assert its defaults.

    prereg 6b: "the Phase B producer's defaults must equal the registered
    values and its test must assert that they do -- a default that silently
    disagrees with the registration is how a report ends up quoting neither
    quantity."  ``tests/test_analyze_channel_audit_frozen.py`` reads this
    parser and pins every row of that table.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sidecars", type=Path, default=Path("results"),
                    help="directory of *.instrumentation.json Phase B sidecars")
    ap.add_argument("--out-md", type=Path, default=Path("reports/channel-audit.md"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/channel-audit.json"))
    ap.add_argument("--out-figdir", type=Path, default=Path("reports/figures"),
                    help="reports/figures/channel-audit-*.png (prereg 6b)")
    ap.add_argument("--run-prefix", type=str, default="airbench_instrumented_")
    ap.add_argument("--min-seed", type=int, default=1000,
                    help="CLAUDE.md ground rule 2: 0-99 are evaluation seeds")
    ap.add_argument("--limit", type=int, default=None,
                    help="use an evenly spaced subset of N sidecars (smoke run)")
    ap.add_argument("--min-n", type=int, default=10,
                    help="drop series with fewer raw observations than this")
    ap.add_argument("--max-lag", type=int, default=LADDER_MAX_LAG,
                    help="the ladder (prereg 6b registers 64); the Newey-West "
                         "bandwidth is L = 4 regardless")
    ap.add_argument("--tau-lags", type=int, nargs="+", default=list(TAU_LAGS),
                    help="prereg 5.9: all of {8, 16, 32, 64} computed, all "
                         "required to agree on the branch")
    ap.add_argument("--tau-primary-k", type=int, default=TAU_PRIMARY_K)
    ap.add_argument("--burn-ins", type=int, nargs="+", default=list(REGISTERED_BURN_INS),
                    help="prereg 5.3: reported alongside every quantity, always")
    ap.add_argument("--primary-burn-in", type=int, default=PRIMARY_BURN_IN)
    ap.add_argument("--null-reps", type=int, default=NULL_REPS,
                    help="prereg 5.6: 2000 cannot resolve a 1e-4 exceedance rate")
    ap.add_argument("--null-seed", type=int, default=NULL_SEED)
    ap.add_argument("--tau-reference-seed", type=int, default=TAU_REFERENCE_SEED,
                    help="prereg 5.9: deliberately != --null-seed, so the white "
                         "control is not a tautology")
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    ap.add_argument("--bootstrap-block", type=int, default=BOOTSTRAP_BLOCK,
                    help="prereg 5.8: exactly one (run, matrix) frozen bank")
    ap.add_argument("--bootstrap-block-tracked", type=int, default=BOOTSTRAP_BLOCK_TRACKED,
                    help="prereg 5.8: the segments of one tracked direction")
    ap.add_argument("--bootstrap-seed", type=int, default=NULL_SEED)
    ap.add_argument("--verify-probes", type=int, default=64,
                    help="probes re-checked against the logged accumulator "
                         "statistic (prereg 5.1's mirror contract)")
    ap.add_argument("--synthetic-control", choices=("none", "white", "ar1"), default="none",
                    help="run the FULL pipeline on a generated stream instead of "
                         "sidecars (prereg 7 K1 / 6d)")
    ap.add_argument("--control-probes", type=int, default=2048,
                    help="probes per K1 control stream")
    ap.add_argument("--control-steps", type=int, default=200)
    ap.add_argument("--control-seed", type=int, default=1301)
    ap.add_argument("--no-controls", dest="controls", action="store_false",
                    help="skip the embedded K1 controls (they cost one extra "
                         "null draw per control)")
    ap.set_defaults(controls=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args._selection = None
    if args.synthetic_control != "none":
        defaults = build_parser().parse_args([])
        if args.out_md == defaults.out_md or args.out_json == defaults.out_json:
            raise SystemExit(
                "--synthetic-control writes a CONTROL, not the Phase B report: "
                "pass explicit --out-md/--out-json paths so a control can never "
                "land on reports/channel-audit.{md,json}"
            )
        entries = synthetic_entries(
            args.synthetic_control,
            probes=max(int(args.control_probes) // 2, 2),
            matrices=2, runs=1, n_steps=int(args.control_steps),
            seed=int(args.control_seed),
            phi=0.0 if args.synthetic_control == "white" else PHASE_A_PHI,
            tracked=8,
        )
    else:
        paths, selection = select_sidecars(
            args.sidecars, run_prefix=args.run_prefix or None,
            min_seed=None if args.min_seed is None or args.min_seed < 0 else args.min_seed,
            limit=args.limit,
        )
        args._selection = selection
        entries = load_sidecars(paths)

    bank = ingest(entries, min_n=int(args.min_n))
    report = build_report(bank, args)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(report))
    if args.out_figdir:
        made = make_figures(report, args.out_figdir)
        if not made:
            print("matplotlib unavailable; skipped the figures", file=sys.stderr)

    p1, p2, p3 = report["p1"], report["p2"], report["p3"]
    print(
        "P1 band_contrast=%s tail_contrast=%s -> row %s (%s); P1 read: %s"
        % (_fmt(p1.get("band_contrast")), _fmt(p1.get("tail_contrast")),
           p1.get("outcome_row"), p1.get("outcome_row_label"), p1.get("read"))
    )
    print(
        "P2 frame_gain=%s bulk_gain=%s -> row %s (%s)"
        % (_fmt(p2.get("frame_gain")), _fmt(p2.get("bulk_gain")),
           p2.get("outcome_row"), p2.get("outcome_row_label"))
    )
    print(
        "P3 tau_cal(K=%s)=%s -> %s (K-stable: %s)"
        % (p3.get("primary_K"), _fmt(p3.get("tau_cal")), p3.get("verdict"),
           p3.get("k_stable"))
    )
    print(
        "Rider-1 ratio=%s -> %s; Rider-2 -> %s"
        % (_fmt(report["rider"].get("ratio")), report["rider"].get("rider1_branch"),
           report["rider"].get("rider2_branch"))
    )
    print(
        "thresholds are %s; every row above is a lookup, not a verdict "
        "(adjudication is HUMAN)" % THRESHOLD_STATUS
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
