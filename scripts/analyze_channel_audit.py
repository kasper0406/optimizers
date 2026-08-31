#!/usr/bin/env python
"""Two-channel (DC / alternating) audit of the tracked-direction tier.

EXPLORATORY TIER -- READ THE HEADER BEFORE THE NUMBERS.  This script is
offline analysis of the 218 ALREADY-RECORDED frontier sidecars
(``results/*.instrumentation.json``, dev seeds 1400-1401 / 1410-1414, program
#6/#6b lr x batch ladders).  Those sidecars have **already been peeked**: the
lag-ladder / channel readings quoted as A1-A5 in
``reports/channel-audit-preregistration.md`` §2 were produced by ad-hoc
in-session analysis over exactly this data before any criterion existed.
Everything this script prints is therefore a PRIOR, not evidence: it is
unblinded, uncontrolled for multiplicity, and computed on a tier whose
period-50 refresh cadence has a harmonic sitting exactly on the Nyquist
frequency the alternating channel reads.  The confirmatory surface is the
frozen-probe tier re-run on GPU (prereg §3), which does not exist on disk.

DESCRIPTIVE OUTPUT ONLY: quantities, never verdicts.  No gate is evaluated,
no threshold is compared, no pass/fail is emitted (CLAUDE.md ground rule 1).

WHAT IS MEASURED
----------------
For matrix ``m`` and tracked direction slot ``j`` the sidecar stores the raw
per-step projection ``s_t = u_j^T G_t v_j`` at every step.  The tracked
subspace is re-derived every ``t_refresh`` steps, so ``s`` is a concatenation
of fixed-direction SEGMENTS.  Each segment is cut out, its first ``burn_in``
observations are dropped (the re-anchoring transient at the head of a segment
is load-bearing, prereg §5.2 / A2), and the segment is read twice:

* **DC channel**      ``x_t = s_t``            -- the published mean test;
* **alternating**     ``x_t = (-1)^t s_t``     -- the Nyquist end of the band,

with ``t`` the ABSOLUTE training step from the sidecar's ``steps`` array, so
parity is consistent across matrices, segments, runs and batch sizes (prereg
§1).  Both channels go through the identical Newey-West estimator, and both
are calibrated against an AR(1) surrogate null matched to the cell's own
fitted phi and median segment length -- a zig-zag stream has ESS > n and a
raw |t| threshold is not on the N(0, 1) scale, so an uncalibrated |t| is
uninterpretable.

Per segment: the bias-corrected lag ladder rho_1..rho_max_lag
(``src.stats.spectral.lag_ladder``) and the per-channel Newey-West t
(``src.stats.spectral.channel_t``).  Segments are stratified by
kind (top/bulk) x lr x batch_size and every cell reports medians, the
null-calibrated ratio, exceedance fractions and block-bootstrap intervals.

Per SLOT (§7 of the report): the same channels read on the prereg §5.1 unit
-- a slot's post-burn-in segments pooled with lag products never crossing a
boundary (``src.stats.spectral.segmented_channel_t``), against the matching
segmented null (``ar1_segmented_null``).  That section exists because the
per-segment ``ratio`` cannot identify what it looks like it identifies; see
WHAT ``ratio`` DOES NOT MEAN below.

WHAT ``ratio`` DOES NOT MEAN
---------------------------
``ratio`` = observed median |t_nw| / null median |t_nw| is a SCALE
correction, not an identification.  Its numerator is the zero-frequency
content of a 45-observation window; its denominator integrates lags 1..L
with Bartlett weights and L = newey_west_bandwidth(45, 8) = 3.  Any
zero-mean component with a correlation time longer than ~3 steps is
therefore absent from the denominator and fully present in the numerator,
and an AR(1) null fitted to rho_1 has a correlation time under one step by
construction and cannot calibrate it away.  Demonstrated in
``tests/test_analyze_channel_audit.py`` and in the spectral module
docstring: a ZERO-MEAN stream (fast AR(1) phi = -0.40 plus slow AR(1)
phi = 0.97 at variance 0.15) pushed through this exact pipeline returns
rho_1 = -0.33, dc median |t| = 1.78, dc ratio = 2.71, alt ratio = 1.04 --
numerically the reported top-tier cell, with no mean anywhere in it.

The discriminator is the SLOT-level growth factor (report §7): over k
pooled segments a mean that survives a subspace refresh grows |t| like
sqrt(k), while within-segment low-frequency power does not, because L grows
with N and starts absorbing it.  Both branches are pinned as tests.

ESTIMATOR NOTES / DEVIATIONS FROM THE DRAFT PRE-REGISTRATION
------------------------------------------------------------
This list is meant to be COMPLETE.  Every registered §5 choice this script
does not implement is here; if a reviewer finds a seventh, that is a defect
in this block, not a licensed deviation (CLAUDE.md ground rule 3).

* **Primary unit of analysis is the SEGMENT**, not the direction slot.  The
  DRAFT prereg §5.1 registers the slot-level estimator (a slot's
  post-burn-in segments concatenated, lag products never crossing a
  boundary, N ~ 720 at burn-in 5).  Sections 1-5 report the per-segment unit
  (N ~ 45), which is a different, shorter-window statistic: |t| scales like
  sqrt(N), so those numbers are NOT comparable rung-for-rung with a
  slot-level reading.  Section 7 adds the registered slot-level unit back as
  a diagnostic (N ~ 270), with its own matched null.
* **`--null-reps` defaults to 2000, not the registered 200000** (§5.6).
  The registration states verbatim why: the exceedance rates involved are of
  order 1e-4 and 2000 draws cannot resolve them.  Consequences, both
  reported rather than hidden: `null.frac_abs_t_ge['4']` has a resolution
  floor of 1/2000 = 5e-4 and is emitted with `null.frac_abs_t_ge_floor`
  beside it; and the null MEDIAN carries ~2.8% (dc) / ~3.1% (alt) Monte-Carlo
  error at 2000 reps, which propagates into every `ratio`.  Both the null
  median's own bootstrap interval and the combined `ratio_ci95` are emitted
  and printed, so no `ratio` appears without its reproducibility floor.
* **`--bootstrap-block` defaults to 16, which is NOT the registered block**
  (§5.8).  For tracked per-segment statistics the registration says "the
  block is the 4 segments of one direction"; rows here are ordered
  (run, matrix, segment, slot), so 16 consecutive rows are one
  (run, matrix, segment, kind) cluster of direction SLOTS -- an orthogonal
  clustering of the same table, wide in slots and narrow in time rather than
  the other way round.  It was chosen because slots of one matrix at one
  step share the momentum matrix that selected them; it is a different
  dependence unit from the registered one and the intervals cover
  accordingly.  Both are reported: `--bootstrap-block-alt` (default 4, the
  registered direction-block) is run alongside on every pooled interval.
* **The surrogate null is drawn per DISTINCT raw segment length** in each
  cell and mixed at the cell's own length frequencies, rather than at "every
  distinct series length in the design" globally (§5.6).  This is a repair,
  not a deviation, of an earlier defect: drawing one null at the cell's
  MEDIAN raw length put every b04000 cell's null at n_raw = 48, a length
  that does not occur in it at all (its segments are 46 and 50, 306 each),
  and never drew the registered n = 42 -> 37 null.  The mixture weights and
  per-length nulls are emitted under `null.by_segment_len`.
* **Burn-in 0 is carried in the sweep** in addition to the registered
  {5, 15, 25} (§5.3).  It is the only value that can exhibit the A2
  re-anchoring transient the burn-in exists to remove, and prereg §2's A2
  anchor is stated at burn-in 0; without it the burn-in table cannot test
  its own annotation.  Every criterion is still read at 5 and no registered
  row is removed.
* **Segment boundaries.**  ``--segment-at refresh`` (default) cuts at every
  entry of the matrix's ``refresh_steps``: the tracked pair is re-derived from
  the momentum matrix at every refresh, so those are the only intervals over
  which the direction is guaranteed fixed.  ``--segment-at reset`` cuts at the
  direction's own ``reset_steps`` instead, which is the DRAFT prereg §5.1
  choice; ``reset_steps`` is the subset of refreshes whose alignment fell
  below ``align_min``, i.e. it treats a well-aligned refresh as direction
  continuity.  The fraction of refreshes that did not reset is reported under
  ``diagnostics``.  Neither mode ever cuts at a data-dependent boundary
  interior to a segment: conditioning a cut on the observations themselves
  would select on the very fluctuation being measured.
* **Bias correction.**  ``rho_k = c_k/c_0 + 1/n`` per segment, the
  registered (§5.4) process-independent mean-subtraction term.  ``rho_raw``
  medians are reported next to it -- computed by the SAME degenerate-row
  logic, not reconstructed as ``rho - 1/n``, which would report -1/n for a
  variance-floored row the estimator left at exactly 0.  The correction is
  first-order and its residual is MEASURED, not assumed: +0.014 at
  phi = -0.34 and n = 45, which is several times the block-bootstrap CI
  half-width this script prints on rho_1, so a reported rho_1 of -0.343
  corresponds to a true phi near -0.358 (spectral.py module docstring has
  the map, and §5.4's quoted +0.0075 and its residual formula are both
  wrong -- an amendment is owed to the registration, which this script does
  not write).
* **Vectorization / mirror contract.**  The per-segment and per-slot
  statistics are computed by batched NumPy kernels over all direction slots
  of a matrix at once (~12.6M observations do not survive a per-observation
  Python loop).  The kernels are not second estimators: ``--verify-blocks``
  re-computes a seeded UNIFORM sample (:class:`DeterministicSampler`) of
  segments and slots through ``src.stats.spectral`` itself, walking the
  checked slot index so `top` and `bulk` positions and short segments are all
  reachable, and reports the maximum deviation AND the coverage (which slot
  positions, which raw lengths) under ``diagnostics.mirror_check``.

Deterministic: sorted keys, no timestamps, seeded RNGs only, NumPy only, no
GPU, no network -- identical inputs produce byte-identical outputs.

Usage (prereg §6a; the Phase B filenames are RESERVED and refused here):
    uv run --no-sync python scripts/analyze_channel_audit.py \
        --sidecars results \
        --out-md reports/channel-audit-phase-a.md \
        --out-json reports/channel-audit-phase-a.json \
        [--limit 5]           # deterministic evenly-spaced smoke subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.instrument.schema import (  # noqa: E402
    SIDECAR_SUFFIX,
    load_instrumentation,
)
from src.stats.spectral import (  # noqa: E402
    CHANNELS,
    DEFAULT_MAX_LAG,
    VAR_FLOOR,
    ar1_segmented_null,
    ar1_streams,
    ar1_surrogate_null,
    block_bootstrap_ci,
    channel_t,
    lag_ladder,
    newey_west_bandwidth,
    segment_mean_persistence,
    segmented_channel_t,
)

REGISTERED_BURN_INS = (5, 15, 25)  # prereg 5.3 sensitivity sweep
BURN_INS = (0,) + REGISTERED_BURN_INS  # 0 added for the A2 transient anchor
PRIMARY_BURN_IN = 5
T_THRESHOLDS = (2.0, 4.0)
TAU_LAGS = (4, 8)  # tau = 1 + 2 sum_{k<=K} rho_k, K capped at max_lag
PHI_CLAMP = 0.98  # ar1_surrogate_null requires |phi| < 1
KINDS = ("bulk", "top")
# reports/channel-audit-preregistration.md 6b reserves these names for the
# CONFIRMATORY Phase B producer; this is the Phase A producer (6a).
PHASE_B_RESERVED = ("channel-audit.md", "channel-audit.json")

# prereg 2: the peeked Phase A anchors this script is registered to reproduce.
# 'quantity' names what the script emits; None means the script cannot produce
# it at all, which is itself the reportable fact (the obligation stays OPEN).
PHASE_A_ANCHORS = (
    {
        "anchor": "A1",
        "claim": "clean AR(1), phi ~ -0.34 on the tracked lag ladder, "
                 "LR-INVARIANT across the ladder",
        "quantity": "phi_hat by kind x lr at burn-in 5 (report section 1)",
    },
    {
        "anchor": "A2",
        "claim": "burn-in is load-bearing: at burn-in 0 rho_2 reads ~ -0.01 "
                 "against the AR(1) prediction +0.116 at phi = -0.34",
        "quantity": "rho_2 at burn-in 0 vs burn-in 5 (report section 5)",
    },
    {
        "anchor": "A3",
        "claim": "alternating channel at the null: median |t_alt,nw| ~ "
                 "0.75-0.85 across every lr, burn-in and kind",
        "quantity": "alt median |t| over every cell (report sections 2-5)",
    },
    {
        "anchor": "A4",
        "claim": "DC excess monotone in lr, peaking at 3.77 at lr = 0.96, "
                 "2.83 at burn-in 25; confined to top",
        "quantity": "dc median |t| by kind x lr and by burn-in "
                    "(report sections 2 and 5)",
    },
    {
        "anchor": "A5",
        "claim": "the published beta = 0.9 vs 0.99 tier contrast (0.596 vs "
                 "0.400) is ~20% reproduced by a zero-mean AR(1) surrogate",
        "quantity": None,
    },
)


# ------------------------------------------------------------------ loading


def tombstoned_runs(dir_path: Path) -> set:
    """Run names listed in ``results/INVALID_RUNS.json``.

    That file's own header says its entries "must be excluded by every
    analysis tool"; it is append-only, so the exclusion is a lookup, never a
    deletion.  Names are returned with the ``.json`` suffix stripped so both
    a main results file and its sidecar can be matched against them.
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
    limit: Optional[int],
    stride: int,
    *,
    run_prefix: Optional[str] = None,
    min_seed: Optional[int] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Sorted ``*.instrumentation.json`` paths, filtered and optionally thinned.

    Three filters, applied in this order and each reported rather than
    assumed, because the Phase B runs of the pre-registration write their own
    ``*.instrumentation.json`` into this same directory and re-running the
    documented command must not silently mix tiers:

    1. ``results/INVALID_RUNS.json`` tombstones (:func:`tombstoned_runs`);
    2. ``--run-prefix`` (default ``airbench_instrumented_``), the Phase A
       run family;
    3. ``--min-seed`` (default 1000), CLAUDE.md ground rule 2 -- evaluation
       seeds 0-99 never enter development analysis.  The seed is read from
       the run name's ``seed<NNNN>`` field; a name that carries none is kept
       and reported under ``name_without_seed``.

    ``--limit N`` then keeps an evenly spaced subset of the surviving sorted
    list rather than its first N entries, so a smoke run spans the lr x batch
    grid instead of one corner of it; the choice is a pure function of the
    sorted names and is therefore reproducible.
    """
    all_paths = sorted(Path(dir_path).glob(f"*{SIDECAR_SUFFIX}"))
    if not all_paths:
        raise SystemExit(f"no {SIDECAR_SUFFIX} files under {dir_path}")
    tombs = tombstoned_runs(dir_path)
    excluded: Dict[str, List[str]] = {
        "invalid_runs": [], "min_seed": [], "run_prefix": [],
    }
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
    if stride > 1:
        kept = kept[::stride]
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
        "stride": int(stride),
    }
    return kept, selection


def run_metadata(sidecar: Path) -> Dict[str, Any]:
    """lr / batch_size / seed for a sidecar, from its paired results JSON.

    Returns a dict with a ``problem`` key set (and the labels None) when the
    main results file is missing or unreadable -- a run that cannot be
    labelled is reported as skipped, never guessed at (CLAUDE.md ground rule
    6).
    """
    name = sidecar.name[: -len(SIDECAR_SUFFIX)]
    main = sidecar.with_name(name + ".json")
    out: Dict[str, Any] = {
        "batch_size": None,
        "lr": None,
        "problem": None,
        "run": name,
        "seed": None,
        "sidecar": sidecar.name,
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


# ------------------------------------------------------------- segmentation


def boundaries_from_steps(
    steps: np.ndarray, cut_steps: Sequence[int]
) -> Tuple[Tuple[int, int], ...]:
    """Half-open index intervals of ``steps`` delimited by ``cut_steps``.

    A cut at step ``r`` starts a new segment AT ``r`` (the tracker refreshes
    before the step is logged), so ``r`` is the first observation of the new
    segment.  Cuts outside the recorded range are ignored and the leading /
    trailing remainders are kept as segments of their own.
    """
    edges = {0, int(steps.size)}
    for r in cut_steps:
        idx = int(np.searchsorted(steps, int(r)))
        if 0 < idx < steps.size:
            edges.add(idx)
    ordered = sorted(edges)
    return tuple((a, b) for a, b in zip(ordered[:-1], ordered[1:]) if b > a)


def direction_groups(
    mat: Dict[str, Any], steps: np.ndarray, mode: str
) -> List[Tuple[Tuple[int, int], ...]]:
    """[(boundaries, [direction positions])] for one matrix.

    ``mode='refresh'`` puts every direction in one group (all slots share the
    matrix's refresh cadence).  ``mode='reset'`` groups slots by their own
    ``reset_steps`` tuple, which keeps the batched kernel wide even though
    resets are per-direction.
    """
    dirs = mat.get("directions", [])
    if mode == "refresh":
        bounds = boundaries_from_steps(steps, mat.get("refresh_steps", []))
        return [(bounds, list(range(len(dirs))))]
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for pos, d in enumerate(dirs):
        cuts = tuple(int(x) for x in d.get("reset_steps", []))
        groups.setdefault(cuts, []).append(pos)
    return [
        (boundaries_from_steps(steps, key), members)
        for key, members in sorted(groups.items())
    ]


# ------------------------------------------------------- batched estimators


def _lag_sums(Y: np.ndarray, max_lag: int) -> np.ndarray:
    """``S_j = sum_t y_t y_{t-j}``, j = 0..max_lag, for every row of ``Y``."""
    n = Y.shape[1]
    S = np.full((Y.shape[0], max_lag + 1), np.nan)
    S[:, 0] = np.einsum("ij,ij->i", Y, Y)
    for j in range(1, max_lag + 1):
        if n - j < 1:
            break
        S[:, j] = np.einsum("ij,ij->i", Y[:, j:], Y[:, :-j])
    return S


def _moments(
    Y: np.ndarray, max_lag: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise mean, autocovariance ladder c_0..c_max_lag, and raw lag sums.

    Exactly the definition of :meth:`FrozenProbeAccumulator.stats` mirrored by
    ``src.stats.spectral``: ``c_0 = S_0/n - mean^2`` (divisor n, floored at 0)
    and ``c_j = S_j/(n-j) - mean^2`` (divisor n - j), all about the row's own
    post-burn-in mean.  The uncentred ``S`` is returned as well because the
    slot-level (prereg 5.1) pooling adds S across a slot's segments and only
    then subtracts the pooled mean -- centring per segment first would not be
    the same statistic.
    """
    n = Y.shape[1]
    mean = Y.sum(axis=1) / n
    S = _lag_sums(Y, max_lag)
    mm = mean * mean
    c = np.full_like(S, np.nan)
    c[:, 0] = np.maximum(S[:, 0] / n - mm, 0.0)
    for j in range(1, max_lag + 1):
        if n - j < 1:
            break
        c[:, j] = S[:, j] / (n - j) - mm
    return mean, c, S


def _ladder_from_moments(
    c: np.ndarray, n: int, max_lag: int, *, bias_correct: bool = True
) -> np.ndarray:
    """``rho_k = c_k/c_0 (+ 1/n)`` for every row.

    Rows whose variance is at or below ``VAR_FLOOR`` report rho = 0 at every
    lag and take no bias correction, the :class:`DirectionStats` convention
    that ``src.stats.spectral.lag_ladder`` follows (the correlation of a
    constant segment is undefined, not zero-with-error).  ``bias_correct``
    exists so the raw ladder is produced by the SAME degenerate-row logic:
    reconstructing it downstream as ``rho - 1/n`` would report -1/n for rows
    the estimator deliberately left at exactly 0.
    """
    c0 = c[:, 0]
    live = c0 > VAR_FLOOR
    rho = np.zeros((c.shape[0], max_lag))
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = c[:, 1:] / np.where(live, c0, 1.0)[:, None]
    rho[live] = raw[live] + (1.0 / n if bias_correct else 0.0)
    return rho


def _channel_from_moments(
    mean: np.ndarray, c: np.ndarray, n: int, max_lag: int
) -> Dict[str, Any]:
    """Newey-West readout of a channel mean, batched over rows.

    Mirrors :func:`src.stats.spectral.channel_t`: Bartlett kernel, Newey-West
    (1994) automatic bandwidth ``L = min(max_lag, floor(4 (n/100)^(2/9)),
    n-2)`` recomputed on the POST-burn-in length, and a fallback to ``c_0``
    with ``nw_floored`` set when the truncated long-run variance is
    non-positive.
    """
    c0 = c[:, 0]
    L = newey_west_bandwidth(n, max_lag)
    sigma = c0.copy()
    for j in range(1, L + 1):
        if n - j <= 0:
            break
        cj = c[:, j]
        sigma += 2.0 * (1.0 - j / (L + 1.0)) * np.where(np.isfinite(cj), cj, 0.0)
    floored = sigma <= 0.0
    sigma = np.where(floored, c0, sigma)
    pos = sigma > 0.0
    safe_sigma = np.where(pos, sigma, 1.0)
    live = c0 > 0.0
    safe_c0 = np.where(live, c0, 1.0)
    return {
        "ess": np.where(pos, n * c0 / safe_sigma, float(n)),
        "lag_truncation": L,
        "mean": mean,
        "nw_floored": floored,
        "t_naive": np.where(live, mean / np.sqrt(safe_c0 / n), 0.0),
        "t_nw": np.where(pos, mean / np.sqrt(safe_sigma / n), 0.0),
    }


def segment_block(
    raw: np.ndarray, parity: np.ndarray, burn_in: int, max_lag: int
) -> Dict[str, Any]:
    """All per-segment statistics for one (matrix, segment) block of slots.

    ``raw`` is (n_slots, segment_length) of the PRE-burn-in segment and
    ``parity`` the matching ``(-1)^step`` signs.  The alternating channel is
    formed on the full segment and the burn-in is dropped afterwards, so a
    given absolute step keeps its demodulation sign whatever burn-in is used.

    ``sums`` carries the uncentred lag sums of both channels so a caller can
    pool a slot's segments (report section 7) without recomputing anything.
    """
    dc = raw[:, burn_in:]
    alt = (raw * parity[None, :])[:, burn_in:]
    n = dc.shape[1]
    mean_dc, c_dc, S_dc = _moments(dc, max_lag)
    mean_alt, c_alt, S_alt = _moments(alt, max_lag)
    return {
        "alt": _channel_from_moments(mean_alt, c_alt, n, max_lag),
        "dc": _channel_from_moments(mean_dc, c_dc, n, max_lag),
        "n_kept": n,
        "n_raw": raw.shape[1],
        "rho": _ladder_from_moments(c_dc, n, max_lag),
        "rho_raw": _ladder_from_moments(c_dc, n, max_lag, bias_correct=False),
        "sums": {"alt": S_alt, "dc": S_dc},
    }


class SlotAccumulator:
    """Pool a slot's segments into the prereg 5.1 slot-level statistic.

    One instance per (matrix, direction-group, burn-in).  ``add`` folds in
    one :func:`segment_block`; ``finish`` returns the batched equivalent of
    :func:`src.stats.spectral.segmented_channel_t` over every slot at once.

    Lag products never cross a segment boundary: the uncentred sums S_j are
    added across segments, the pair counts P_j = sum_i (n_i - j) with them,
    and the pooled mean is subtracted only at the end --

        c_0 = S_0 / N - mean^2 ,  c_j = S_j / P_j - mean^2 ,
        L   = min(max_lag, floor(4 (N/100)^(2/9)), N - 2) ,

    i.e. exactly the same arithmetic on a longer window.  The point of the
    longer window is stated in the module docstring: over k segments a mean
    that survives a subspace refresh grows |t| like sqrt(k) while
    within-segment low-frequency power does not, and the per-segment ratio
    cannot tell those apart.
    """

    def __init__(self, n_slots: int, max_lag: int) -> None:
        self.max_lag = int(max_lag)
        self.n = 0
        self.n_segments = 0
        self.pairs = np.zeros(max_lag + 1, dtype=np.int64)
        self.sums = {ch: np.zeros((n_slots, max_lag + 1)) for ch in CHANNELS}
        self.totals = {ch: np.zeros(n_slots) for ch in CHANNELS}
        self.seg_means: Dict[str, List[np.ndarray]] = {ch: [] for ch in CHANNELS}

    def add(self, block: Dict[str, Any]) -> None:
        n = int(block["n_kept"])
        self.n += n
        self.n_segments += 1
        for j in range(self.max_lag + 1):
            self.pairs[j] += max(n - j, 0)
        for ch in CHANNELS:
            S = np.where(np.isfinite(block["sums"][ch]), block["sums"][ch], 0.0)
            self.sums[ch] += S
            mean = np.asarray(block[ch]["mean"], dtype=np.float64)
            self.totals[ch] += mean * n
            self.seg_means[ch].append(mean.copy())

    def finish(self) -> Optional[Dict[str, Any]]:
        if self.n < 2 or self.n_segments == 0:
            return None
        N, L = self.n, newey_west_bandwidth(self.n, self.max_lag)
        out: Dict[str, Any] = {
            "lag_truncation": L,
            "n": N,
            "n_segments": self.n_segments,
        }
        for ch in CHANNELS:
            mean = self.totals[ch] / N
            mm = mean * mean
            c0 = np.maximum(self.sums[ch][:, 0] / N - mm, 0.0)
            sigma = c0.copy()
            for j in range(1, L + 1):
                if self.pairs[j] <= 0:
                    continue
                sigma += 2.0 * (1.0 - j / (L + 1.0)) * (
                    self.sums[ch][:, j] / self.pairs[j] - mm
                )
            floored = sigma <= 0.0
            sigma = np.where(floored, c0, sigma)
            pos = sigma > 0.0
            live = c0 > 0.0
            means = np.stack(self.seg_means[ch], axis=0)  # (n_segments, n_slots)
            out[ch] = {
                "ess": np.where(pos, N * c0 / np.where(pos, sigma, 1.0), float(N)),
                "mean": mean,
                "nw_floored": floored,
                "segment_mean_acf1": _acf1_about_zero(means),
                "segment_mean_coherence": _sign_coherence(means),
                "t_naive": np.where(
                    live, mean / np.sqrt(np.where(live, c0, 1.0) / N), 0.0
                ),
                "t_nw": np.where(
                    pos, mean / np.sqrt(np.where(pos, sigma, 1.0) / N), 0.0
                ),
            }
        return out


def _sign_coherence(means: np.ndarray) -> np.ndarray:
    """``max(#positive, #negative) / k`` per column of a (k, n_slots) block.

    Column-wise :func:`src.stats.spectral.segment_mean_persistence`, batched;
    the scalar function is the tested definition and the mirror check holds
    this against it.
    """
    k = means.shape[0]
    pos = np.sum(means > 0.0, axis=0)
    neg = np.sum(means < 0.0, axis=0)
    return np.maximum(pos, neg) / float(k)


def _acf1_about_zero(means: np.ndarray) -> np.ndarray:
    """``sum_i m_i m_{i+1} / sum_i m_i^2`` per column, NaN on a null column."""
    if means.shape[0] < 2:
        return np.full(means.shape[1], np.nan)
    denom = np.einsum("ij,ij->j", means, means)
    num = np.einsum("ij,ij->j", means[1:], means[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0.0, num / np.where(denom > 0.0, denom, 1.0), np.nan)


# --------------------------------------------------------- mirror contract


def mirror_deviation(
    raw: np.ndarray, parity: np.ndarray, burn_in: int, max_lag: int, row: int
) -> Dict[str, float]:
    """Max |batched - src.stats.spectral| on one segment of one slot.

    The batched kernel is an optimization, not a second estimator; this
    re-runs the canonical scalar functions on a single row so the equivalence
    is a reported number rather than an assertion in a comment.  ``alt`` is
    checked twice: against ``channel_t(signed, 'dc')`` (absolute-step parity,
    the convention used here) and, on |t| only, against
    ``channel_t(raw, 'alt')`` (segment-relative parity, the module default) --
    the two differ by a global sign whenever the segment starts on an odd
    step, and never in magnitude.
    """
    batched = segment_block(raw[row : row + 1], parity, burn_in, max_lag)
    ladder = lag_ladder(raw[row], max_lag, burn_in)
    ref_dc = channel_t(raw[row], "dc", burn_in, max_lag=max_lag)
    ref_alt = channel_t(raw[row] * parity, "dc", burn_in, max_lag=max_lag)
    ref_alt_rel = channel_t(raw[row], "alt", burn_in, max_lag=max_lag)
    dev = {
        "rho": float(np.nanmax(np.abs(batched["rho"][0] - np.asarray(ladder["rho"])))),
        "rho_raw": float(
            np.nanmax(np.abs(batched["rho_raw"][0] - np.asarray(ladder["rho_raw"])))
        ),
        "t_nw_dc": abs(float(batched["dc"]["t_nw"][0] - ref_dc["t_nw"])),
        "t_nw_alt": abs(float(batched["alt"]["t_nw"][0] - ref_alt["t_nw"])),
        "abs_t_nw_alt_parity": abs(
            abs(float(batched["alt"]["t_nw"][0])) - abs(float(ref_alt_rel["t_nw"]))
        ),
        "ess_dc": abs(float(batched["dc"]["ess"][0] - ref_dc["ess"])),
        "ess_alt": abs(float(batched["alt"]["ess"][0] - ref_alt["ess"])),
    }
    return dev


def slot_mirror_deviation(
    segments: Sequence[np.ndarray],
    parities: Sequence[np.ndarray],
    burn_in: int,
    max_lag: int,
    row: int,
) -> Dict[str, float]:
    """Max |SlotAccumulator - src.stats.spectral| on one slot.

    The slot-level pooling has the same status as the per-segment kernel: an
    optimization over :func:`src.stats.spectral.segmented_channel_t`, never a
    second estimator.  ``alt`` is checked in the absolute-step convention
    (the signed series pushed through the ``dc`` branch), matching what
    :class:`SlotAccumulator` accumulates.
    """
    acc = SlotAccumulator(1, max_lag)
    for raw, parity in zip(segments, parities):
        acc.add(segment_block(raw[row : row + 1], parity, burn_in, max_lag))
    got = acc.finish()
    dev: Dict[str, float] = {}
    if got is None:
        return dev
    for ch, series in (
        ("dc", [raw[row] for raw in segments]),
        ("alt", [raw[row] * par for raw, par in zip(segments, parities)]),
    ):
        ref = segmented_channel_t(series, "dc", burn_in, max_lag=max_lag)
        per = segment_mean_persistence(ref["segment_means"])
        dev[f"slot_t_nw_{ch}"] = abs(float(got[ch]["t_nw"][0] - ref["t_nw"]))
        dev[f"slot_ess_{ch}"] = abs(float(got[ch]["ess"][0] - ref["ess"]))
        dev[f"slot_coherence_{ch}"] = abs(
            float(got[ch]["segment_mean_coherence"][0] - per["coherence"])
        )
        dev[f"slot_acf1_{ch}"] = abs(
            float(got[ch]["segment_mean_acf1"][0] - per["acf1"])
        )
    return dev


# --------------------------------------------------------- estimator control


SELF_TEST_CASES = (
    # (label, phi, alt_mean, dc_mean, slow_phi, slow_var)
    ("K1a phi=0.00", 0.0, 0.0, 0.0, None, 0.0),
    ("K1b phi=-0.34", -0.34, 0.0, 0.0, None, 0.0),
    ("K1c phi=+0.50", 0.5, 0.0, 0.0, None, 0.0),
    ("K2a phi=-0.34 +alt mean", -0.34, 0.5, 0.0, None, 0.0),
    ("K2b phi=-0.40 +slot dc mean", -0.40, 0.0, 0.30, None, 0.0),
    ("K3 zero-mean slow power", -0.40, 0.0, 0.0, 0.97, 0.15),
)


def self_test(
    reps: int,
    burn_in: int,
    max_lag: int,
    seg_len: int,
    n_segments: int,
    nulls: "NullBank",
) -> Dict[str, Any]:
    """Synthetic controls for the estimator + null calibration path.

    Every case is ``reps`` synthetic direction SLOTS of ``n_segments``
    segments of ``seg_len`` steps, pushed through the identical
    :func:`segment_block` / :class:`SlotAccumulator` / :class:`NullBank`
    path the real sidecars use.  Nothing here is asserted -- the assertions
    live in ``tests/test_analyze_channel_audit.py``; this table exists so the
    report carries its own controls next to its own numbers.

    K1 (the DRAFT pre-registration §7 control): on a zero-mean stream there
    is nothing to detect, so the null-calibrated ``ratio`` must sit at 1 in
    BOTH channels whatever the autocorrelation --

    * ``K1a`` white noise: tau returns 1, ratios 1;
    * ``K1b`` phi = -0.34, the peeked tracked-tier value: ratios still 1
      even though ESS/n is ~1.9 in DC and ~0.6 in ALT, which is the whole
      reason a raw |t| threshold is not interpretable;
    * ``K1c`` phi = +0.5, the opposite sign, so a control that only worked
      on anti-correlated input would show up here.

    K2, the positive controls, each of which must move exactly one column:

    * ``K2a`` planted ALTERNATING mean -- ``ratio_alt`` leaves 1, ``ratio_dc``
      stays at it;
    * ``K2b`` planted per-slot DC mean, i.e. a mean that survives a refresh
      -- ``ratio_dc`` leaves 1 AND the slot-level growth goes to sqrt(k).

    K3 is the discriminating control and the reason section 7 exists: a
    stream with NO mean in either channel, built as a fast AR(1) plus a slow
    AR(1) whose correlation time exceeds the L = 3 Newey-West bandwidth of a
    45-step window.  Its per-segment reading is numerically indistinguishable
    from K2b -- large ``ratio_dc``, ``ratio_alt`` at 1 -- and its slot-level
    growth is NOT sqrt(k).  A control that cannot fail measures nothing;
    K2b and K3 are the two branches of the same measurement.
    """
    out: Dict[str, Any] = {
        "burn_in": int(burn_in),
        "n_segments": int(n_segments),
        "reps": int(reps),
        "segment_length": int(seg_len),
        "sqrt_n_segments": float(np.sqrt(n_segments)),
    }
    parity = np.where(np.arange(1, seg_len + 1) % 2 == 0, 1.0, -1.0)
    reps, n_segments = int(reps), int(n_segments)
    for label, phi, alt_mean, dc_mean, slow_phi, slow_var in SELF_TEST_CASES:
        rng = np.random.default_rng(_null_seed(nulls.seed, phi, seg_len, burn_in))
        raw = ar1_streams(rng, phi, seg_len, reps * n_segments)
        if alt_mean:
            raw = raw + alt_mean * parity[None, :]
        if slow_phi is not None:
            slow = ar1_streams(rng, slow_phi, seg_len, reps * n_segments)
            raw = raw + slow / float(slow.std()) * np.sqrt(slow_var)
        raw = raw.reshape(reps, n_segments, seg_len)
        if dc_mean:
            raw = raw + dc_mean * rng.standard_normal(reps)[:, None, None]
        acc = SlotAccumulator(reps, max_lag)
        blocks = []
        for i in range(n_segments):
            block = segment_block(raw[:, i, :], parity, burn_in, max_lag)
            acc.add(block)
            blocks.append(block)
        slot = acc.finish()
        rho_med = [
            float(np.median(np.concatenate([b["rho"][:, k] for b in blocks])))
            for k in range(max_lag)
        ]
        entry: Dict[str, Any] = {
            "phi": phi,
            "planted_alt_mean": alt_mean,
            "planted_slot_dc_mean": dc_mean,
            "rho_1_hat": rho_med[0],
            "slow_phi": slow_phi,
            "slow_var": slow_var,
            "tau_4": 1.0 + 2.0 * float(np.sum(rho_med[:4])),
        }
        seg_null = nulls.get(rho_med[0], seg_len, burn_in)
        slot_null = nulls.get_slot(rho_med[0], seg_len, n_segments, burn_in)
        for ch in sorted(CHANNELS):
            obs = float(np.median(np.abs(np.concatenate(
                [b[ch]["t_nw"] for b in blocks]
            ))))
            e: Dict[str, Any] = {
                "median_abs_t": obs,
                "ratio": None,
                "slot_growth_calibrated": None,
                "slot_median_abs_t": None,
                "slot_ratio": None,
            }
            if slot is not None:
                e["slot_median_abs_t"] = float(np.median(np.abs(slot[ch]["t_nw"])))
                e["slot_segment_mean_coherence"] = float(
                    np.median(slot[ch]["segment_mean_coherence"])
                )
                e["slot_segment_mean_acf1"] = float(
                    np.nanmedian(slot[ch]["segment_mean_acf1"])
                )
            if seg_null is not None:
                ref = float(np.median(np.abs(seg_null[ch]["samples"]["t_nw"])))
                e["null_median_abs_t"] = ref
                e["ratio"] = obs / ref if ref > 0.0 else None
            if slot_null is not None and slot is not None:
                ref = float(np.median(np.abs(slot_null[ch]["samples"]["t_nw"])))
                e["slot_null_median_abs_t"] = ref
                e["slot_null_segment_mean_coherence"] = slot_null[ch][
                    "segment_mean_coherence"
                ]["median"]
                if ref > 0.0:
                    e["slot_ratio"] = e["slot_median_abs_t"] / ref
            if e["ratio"] and e["slot_ratio"]:
                e["slot_growth_calibrated"] = e["slot_ratio"] / e["ratio"]
            entry[ch] = e
        out[label] = entry
    return out


# ------------------------------------------------------------- accumulation


class CellStore:
    """Per-segment statistics bucketed by (kind, lr, batch_size, burn_in).

    Rows are appended in ingest order -- run, matrix, segment, direction slot
    -- and that order is what the primary block bootstrap resamples: a block
    of ``--bootstrap-block`` (16) consecutive rows is exactly one
    (run, matrix, segment, kind) cluster of direction slots.

    That is NOT the block the pre-registration names.  Prereg §5.8 registers,
    for tracked per-segment statistics, "the block is the 4 segments of one
    direction" -- an orthogonal clustering of the same table (narrow in slots,
    wide in time, where this one is the reverse).  Because the two are not
    nested, neither dominates, so both are computed: every row also carries a
    ``slot_rank`` sort key, and :meth:`finalize` returns the permutation that
    puts the rows in (run, matrix, slot, segment) order so the registered
    direction-block interval can be formed from the same data.
    """

    def __init__(self, max_lag: int) -> None:
        self.max_lag = max_lag
        self.rows: Dict[Tuple[Any, ...], Dict[str, List[np.ndarray]]] = {}
        self.runs: Dict[Tuple[Any, ...], set] = {}

    def add(
        self,
        key: Tuple[Any, ...],
        run: str,
        block: Dict[str, Any],
        sel: np.ndarray,
        slot_rank: np.ndarray,
    ) -> None:
        if not sel.any():
            return
        bucket = self.rows.setdefault(key, {})
        n_rows = int(sel.sum())
        ones = np.full(n_rows, float(block["n_kept"]))
        payload = {
            "ess_alt": block["alt"]["ess"][sel],
            "ess_dc": block["dc"]["ess"][sel],
            "n_kept": ones,
            "n_raw": np.full(n_rows, float(block["n_raw"])),
            "nw_floored_alt": block["alt"]["nw_floored"][sel].astype(np.float64),
            "nw_floored_dc": block["dc"]["nw_floored"][sel].astype(np.float64),
            "slot_rank": slot_rank[sel].astype(np.float64),
            "t_naive_alt": block["alt"]["t_naive"][sel],
            "t_naive_dc": block["dc"]["t_naive"][sel],
            "t_nw_alt": block["alt"]["t_nw"][sel],
            "t_nw_dc": block["dc"]["t_nw"][sel],
            "rho": block["rho"][sel],
            "rho_raw": block["rho_raw"][sel],
        }
        for field, values in payload.items():
            bucket.setdefault(field, []).append(values)
        self.runs.setdefault(key, set()).add(run)

    def finalize(self) -> Dict[Tuple[Any, ...], Dict[str, np.ndarray]]:
        return {
            key: {f: np.concatenate(chunks) for f, chunks in sorted(bucket.items())}
            for key, bucket in sorted(self.rows.items())
        }


class SlotStore:
    """Slot-level (prereg §5.1) statistics bucketed by the same cell key."""

    def __init__(self) -> None:
        self.rows: Dict[Tuple[Any, ...], Dict[str, List[np.ndarray]]] = {}
        self.runs: Dict[Tuple[Any, ...], set] = {}

    def add(
        self,
        key: Tuple[Any, ...],
        run: str,
        slot: Dict[str, Any],
        sel: np.ndarray,
        shape_id: int,
    ) -> None:
        if not sel.any():
            return
        bucket = self.rows.setdefault(key, {})
        n_rows = int(sel.sum())
        payload = {
            "acf1_alt": slot["alt"]["segment_mean_acf1"][sel],
            "acf1_dc": slot["dc"]["segment_mean_acf1"][sel],
            "coherence_alt": slot["alt"]["segment_mean_coherence"][sel],
            "coherence_dc": slot["dc"]["segment_mean_coherence"][sel],
            "ess_alt": slot["alt"]["ess"][sel],
            "ess_dc": slot["dc"]["ess"][sel],
            "lag_truncation": np.full(n_rows, float(slot["lag_truncation"])),
            "n": np.full(n_rows, float(slot["n"])),
            "n_segments": np.full(n_rows, float(slot["n_segments"])),
            "shape_id": np.full(n_rows, float(shape_id)),
            "t_naive_alt": slot["alt"]["t_naive"][sel],
            "t_naive_dc": slot["dc"]["t_naive"][sel],
            "t_nw_alt": slot["alt"]["t_nw"][sel],
            "t_nw_dc": slot["dc"]["t_nw"][sel],
        }
        for field, values in payload.items():
            bucket.setdefault(field, []).append(np.asarray(values, dtype=np.float64))
        self.runs.setdefault(key, set()).add(run)

    def finalize(self) -> Dict[Tuple[Any, ...], Dict[str, np.ndarray]]:
        return {
            key: {f: np.concatenate(chunks) for f, chunks in sorted(bucket.items())}
            for key, bucket in sorted(self.rows.items())
        }


class DeterministicSampler:
    """Deterministic, single-pass, uniform subsample of a stream.

    Seeded reservoir sampling (Algorithm R): the result is a uniformly
    random subset of the WHOLE stream of the requested size, reproducible
    from ``seed`` alone, with one pass and O(budget) memory.

    Why not "keep every k-th": this stream is strongly periodic -- blocks
    arrive grouped by (matrix, direction group, burn-in), and the group
    sizes are 1, 2, 4, 8, 16 segments -- so any power-of-two stride aliases
    against it and can miss a 6% subpopulation entirely.  Measured: an
    evenly spaced 64-block sample landed on raw length 50 sixty-four times
    out of sixty-four, i.e. it never checked a short segment.  Taking the
    first N is worse still: it put every check on one run, two matrices and
    slot 16, a BULK slot, so the kernel behind every headline `top` number
    was never checked on a `top` slot.

    Items are returned in stream order so the report can say which slot
    positions and segment lengths the sample actually covered.
    """

    def __init__(self, budget: int, seed: int) -> None:
        self.budget = max(int(budget), 0)
        self.seen = 0
        self.rng = np.random.default_rng(int(seed))
        self.items: List[Tuple[int, Any]] = []

    def offer(self, item: Any) -> None:
        if self.budget <= 0:
            return
        if len(self.items) < self.budget:
            self.items.append((self.seen, item))
        else:
            j = int(self.rng.integers(0, self.seen + 1))
            if j < self.budget:
                self.items[j] = (self.seen, item)
        self.seen += 1

    def sample(self) -> List[Any]:
        return [item for _pos, item in sorted(self.items, key=lambda kv: kv[0])]


def ingest(
    paths: Sequence[Path],
    burn_ins: Sequence[int],
    max_lag: int,
    min_n: int,
    mode: str,
    verify_blocks: int,
    verify_seed: int = 4242,
) -> Tuple[CellStore, SlotStore, Dict[str, Any]]:
    """Stream every sidecar into per-segment and per-slot statistics.

    The mirror-check budget is spent on a seeded uniform subsample of the
    (run, matrix, group, segment, burn-in) blocks
    (:class:`DeterministicSampler`) rather than the first N of them, and the
    checked slot index walks instead of sitting at ``n_slots // 2`` every
    time.  Both matter: taking the first 64 blocks put every check on one
    run, two matrices, raw length 50 and slot 16 -- a BULK slot, so the
    kernel that produced every headline `top` number was never checked on a
    `top` slot or on any short segment.  ``diagnostics.mirror_check`` reports
    which slot positions and raw lengths the sample covered, so the coverage
    is a number in the output rather than a claim in this docstring.
    """
    store = CellStore(max_lag)
    slots = SlotStore()
    meta: Dict[str, Any] = {
        "batch_sizes": set(),
        "lrs": set(),
        "matrices": set(),
        "mirror_check": {"n_checked": 0, "n_slot_checked": 0},
        "n_directions": 0,
        "n_observations": 0,
        "n_refresh_without_reset": 0,
        "n_refreshes": 0,
        "n_segment_start_parity_odd": 0,
        "n_segment_starts": 0,
        "n_segments_dropped_short": 0,
        "n_segments_used": 0,
        "n_slots_used": 0,
        "runs": [],
        "seeds": set(),
        "skipped": [],
        "slot_shapes": {},
    }
    shape_ids: Dict[Tuple[int, ...], int] = {}
    seg_sampler = DeterministicSampler(verify_blocks, verify_seed)
    slot_sampler = DeterministicSampler(
        max(int(verify_blocks) // 8, 1) if verify_blocks else 0, verify_seed + 1
    )

    usable: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        info = run_metadata(path)
        if info["problem"] is not None:
            meta["skipped"].append(
                {"reason": info["problem"], "sidecar": info["sidecar"]}
            )
            continue
        usable.append((path, info))

    for run_idx, (path, info) in enumerate(usable):
        log = load_instrumentation(path)
        run, lr, batch = info["run"], info["lr"], info["batch_size"]
        meta["runs"].append(
            {"batch_size": batch, "lr": lr, "run": run, "seed": info["seed"]}
        )
        meta["lrs"].add(lr)
        meta["batch_sizes"].add(batch)
        meta["seeds"].add(info["seed"])
        for mat_idx, name in enumerate(sorted(log.get("matrices", {}))):
            mat = log["matrices"][name]
            dirs = mat.get("directions", [])
            if not dirs:
                continue
            meta["matrices"].add(name)
            meta["n_directions"] += len(dirs)
            steps = np.asarray(mat["steps"], dtype=np.int64)
            parity_full = np.where(steps % 2 == 0, 1.0, -1.0)
            series = np.asarray([d["s"] for d in dirs], dtype=np.float64)
            meta["n_observations"] += int(series.size)
            kinds = np.asarray([str(d.get("kind", "unknown")) for d in dirs])
            n_refresh = max(len(mat.get("refresh_steps", [])) - 1, 0)
            meta["n_refreshes"] += n_refresh * len(dirs)
            meta["n_refresh_without_reset"] += sum(
                n_refresh - len(d.get("reset_steps", [])) for d in dirs
            )
            for grp_idx, (bounds, members) in enumerate(
                direction_groups(mat, steps, mode)
            ):
                idx = np.asarray(members, dtype=np.int64)
                sub_kinds = kinds[idx]
                for a, _b in bounds:
                    meta["n_segment_starts"] += 1
                    meta["n_segment_start_parity_odd"] += int(steps[a] % 2)
                for burn_in in burn_ins:
                    acc = SlotAccumulator(idx.size, max_lag)
                    shape: List[int] = []
                    kept_raw: List[np.ndarray] = []
                    kept_parity: List[np.ndarray] = []
                    for seg_idx, (a, b) in enumerate(bounds):
                        raw = series[idx, a:b]
                        parity = parity_full[a:b]
                        if raw.shape[1] - burn_in < min_n:
                            meta["n_segments_dropped_short"] += idx.size
                            continue
                        block = segment_block(raw, parity, burn_in, max_lag)
                        meta["n_segments_used"] += idx.size
                        acc.add(block)
                        shape.append(int(raw.shape[1]))
                        kept_raw.append(raw)
                        kept_parity.append(parity)
                        seg_sampler.offer(
                            (raw, parity, burn_in,
                             (run_idx + mat_idx + seg_idx + burn_in) % idx.size, idx)
                        )
                        rank = (
                            ((run_idx * 64 + mat_idx) * 4096 + idx) * 4096 + seg_idx
                        ).astype(np.float64)
                        for kind in KINDS:
                            sel = sub_kinds == kind
                            store.add(
                                (kind, lr, batch, burn_in), run, block, sel, rank
                            )
                    pooled = acc.finish()
                    if pooled is None:
                        continue
                    meta["n_slots_used"] += idx.size
                    shape_key = "x".join(str(v) for v in shape)
                    meta["slot_shapes"][shape_key] = (
                        meta["slot_shapes"].get(shape_key, 0) + idx.size
                    )
                    shape_id = shape_ids.setdefault(tuple(shape), len(shape_ids))
                    for kind in KINDS:
                        sel = sub_kinds == kind
                        slots.add(
                            (kind, lr, batch, burn_in), run, pooled, sel, shape_id
                        )
                    slot_sampler.offer(
                        (kept_raw, kept_parity, burn_in,
                         (run_idx + mat_idx + grp_idx) % idx.size)
                    )

    deviations: Dict[str, List[float]] = {}
    rows_checked: set = set()
    lengths_checked: set = set()
    for raw, parity, burn_in, row, idx in seg_sampler.sample():
        for field, value in mirror_deviation(
            raw, parity, burn_in, max_lag, row
        ).items():
            deviations.setdefault(field, []).append(value)
        meta["mirror_check"]["n_checked"] += 1
        rows_checked.add(int(idx[row]))
        lengths_checked.add(int(raw.shape[1]))
    for kept_raw, kept_parity, burn_in, row in slot_sampler.sample():
        for field, value in slot_mirror_deviation(
            kept_raw, kept_parity, burn_in, max_lag, row
        ).items():
            deviations.setdefault(field, []).append(value)
        meta["mirror_check"]["n_slot_checked"] += 1

    for field, values in sorted(deviations.items()):
        meta["mirror_check"][f"max_abs_dev_{field}"] = float(np.nanmax(values))
    meta["mirror_check"]["segment_lengths_checked"] = sorted(lengths_checked)
    meta["mirror_check"]["slot_positions_checked"] = sorted(rows_checked)
    for field in ("batch_sizes", "lrs", "matrices", "seeds"):
        meta[field] = sorted(x for x in meta[field] if x is not None)
    meta["slot_shapes"] = dict(sorted(meta["slot_shapes"].items()))
    meta["slot_shape_table"] = [
        list(shape) for shape, _i in sorted(shape_ids.items(), key=lambda kv: kv[1])
    ]
    return store, slots, meta


# ------------------------------------------------------------- cell summary


def _median(values: np.ndarray) -> Optional[float]:
    v = values[np.isfinite(values)]
    return float(np.median(v)) if v.size else None


def _frac_at_least(values: np.ndarray, threshold: float) -> Optional[float]:
    v = values[np.isfinite(values)]
    return float(np.mean(v >= threshold)) if v.size else None


def _null_seed(
    base: int, phi: float, n: int, burn_in: int, n_segments: int = 1
) -> int:
    """Deterministic per-null seed from the null's own parameters.

    No timestamps, no global RNG state, and no dependence on iteration order:
    two cells that request the same (phi, n, burn_in, n_segments) null get the
    same stream, which is also what makes the null cache safe.
    """
    mix = (
        int(base) * 1000003
        + int(round(phi * 1000.0)) * 10007
        + int(n) * 101
        + int(burn_in)
        + int(n_segments) * 1000033
    )
    return int(abs(mix) % (2**31 - 1))


class NullBank:
    """Cache of AR(1) surrogate nulls, per-segment and per-slot.

    ``get`` serves :func:`src.stats.spectral.ar1_surrogate_null` keyed by
    (phi, n, burn_in); ``get_slot`` serves
    :func:`src.stats.spectral.ar1_segmented_null` keyed by
    (phi, seg_len, n_segments, burn_in).  Both are drawn at the SAME phi as
    the cell they calibrate, so the segment-level and slot-level ratios are
    on a common footing and their quotient is the growth factor of report
    section 7.

    Two properties of this calibration are disclosed rather than corrected,
    because correcting either would change a registered estimator (prereg
    §5.5/§5.6) and both are smaller than the Monte-Carlo floor of the reps
    actually used:

    * **Plug-in, not indirect inference.**  The cell's measured ``rho_1``
      (already bias-corrected, but still carrying the ~+0.014 residual of
      the spectral module docstring) is used as the null process's TRUE phi.
      The surrogate therefore does not reproduce the statistic it is matched
      on: this ladder estimator run on the null's own streams returns about
      ``phi + 0.014``.  Redrawing at the bias-inverted phi moves the null
      median |t| by -0.4% (dc) / +0.8% (alt) at the `top` ladder.
    * **AR(1), not the full ladder.**  Matching only rho_1 leaves rho_2..8
      to the AR(1) extension.  A Yule-Walker AR(8) fitted to the pooled
      ladder raises the null median |t| by 0.4-4.4% depending on cell and
      channel, i.e. the reported ratios are overstated by up to ~4.5% and by
      a channel-dependent amount.  The direction is anti-conservative, which
      is why it is stated in the report body and not only here.
    """

    def __init__(self, reps: int, seed: int, max_lag: int) -> None:
        self.reps, self.seed, self.max_lag = int(reps), int(seed), int(max_lag)
        self._cache: Dict[Tuple[int, int, int], Optional[Dict[str, Any]]] = {}
        self._slot_cache: Dict[Tuple[Any, ...], Optional[Dict[str, Any]]] = {}

    def get(self, phi: float, n: int, burn_in: int) -> Optional[Dict[str, Any]]:
        if phi is None or not np.isfinite(phi) or n - burn_in < 2:
            return None
        phi = float(np.clip(phi, -PHI_CLAMP, PHI_CLAMP))
        key = (int(round(phi * 1000.0)), int(n), int(burn_in))
        if key not in self._cache:
            self._cache[key] = ar1_surrogate_null(
                key[0] / 1000.0,
                n,
                self.reps,
                _null_seed(self.seed, phi, n, burn_in),
                burn_in=burn_in,
                max_lag=self.max_lag,
            )
        return self._cache[key]

    def get_slot(
        self, phi: float, seg_len, n_segments: int, burn_in: int
    ) -> Optional[Dict[str, Any]]:
        """``seg_len`` is one length or the slot's exact length tuple.

        The tuple form is the one the report uses: a slot whose last refresh
        window is truncated has a shape like (50, 50, 50, 42), and matching
        the null to its first (or modal, or median) length instead of its
        actual shape is the same defect the per-segment mixture repairs.
        """
        if phi is None or not np.isfinite(phi):
            return None
        lengths = (
            (int(seg_len),) * int(n_segments)
            if np.ndim(seg_len) == 0
            else tuple(int(v) for v in seg_len)
        )
        if not lengths or min(lengths) - int(burn_in) < 2:
            return None
        phi = float(np.clip(phi, -PHI_CLAMP, PHI_CLAMP))
        key = (int(round(phi * 1000.0)), lengths, int(burn_in))
        if key not in self._slot_cache:
            self._slot_cache[key] = ar1_segmented_null(
                key[0] / 1000.0,
                list(lengths),
                len(lengths),
                self.reps,
                _null_seed(
                    self.seed, phi, sum(lengths), burn_in, len(lengths)
                ),
                burn_in=burn_in,
                max_lag=self.max_lag,
            )
        return self._slot_cache[key]


def _bootstrap(
    values: np.ndarray,
    block: int,
    reps: int,
    seed: int,
    max_elems: int,
) -> Optional[Dict[str, Any]]:
    """Block-bootstrap CI on the median at the FULL requested rep count.

    ``max_elems`` bounds the (reps x n) resample index array in memory only:
    :func:`src.stats.spectral.block_bootstrap_ci` draws the reps in chunks
    from one generator and the result is bit-identical to the unchunked
    computation.  An earlier version of this function reduced ``reps``
    instead, which bottomed out at 50 on exactly the largest and most-quoted
    pooled cells -- a 95% interval from 50 draws is the 2nd order statistic
    at each end, seed-unstable at ~20% of its own width and ~5% too narrow.
    The rep count is still written into every interval so the reader never
    has to trust this comment.
    """
    v = values[np.isfinite(values)]
    if v.size < 2:
        return None
    return block_bootstrap_ci(
        v, int(min(block, v.size)), int(reps), int(seed), level=95.0,
        max_elems=int(max_elems),
    )


def _length_weights(n_raw: np.ndarray) -> List[Tuple[int, int]]:
    """(raw segment length, count) pairs of a cell, ascending by length."""
    lengths, counts = np.unique(
        np.asarray(n_raw, dtype=np.float64).astype(np.int64), return_counts=True
    )
    return [(int(a), int(b)) for a, b in zip(lengths, counts)]


def _null_mixture(
    nulls: NullBank,
    phi: float,
    weights: Sequence[Tuple[int, int]],
    burn_in: int,
    ch: str,
) -> Optional[Dict[str, Any]]:
    """The cell's null as a MIXTURE over its own raw segment lengths.

    A cell is not one series length.  Drawing a single null at the cell's
    MEDIAN raw length is how every b04000 cell of an earlier version got a
    null at n_raw = 48 -- a length that does not occur in that cell at all,
    whose segments are 46 and 50 in equal numbers -- and how the registered
    n = 42 -> 37 null was never drawn.

    Here each distinct length gets its own matched null and contributes
    ``round(reps * count / total)`` of its i.i.d. draws (at least one) to a
    pooled sample.  Because the per-length draws are already i.i.d., the
    concatenation is an i.i.d. sample of the correct mixture, so its median,
    its exceedance fractions and its bootstrap interval are all read off it
    directly.  ``by_segment_len`` records what went in.
    """
    total = float(sum(c for _n, c in weights))
    if total <= 0:
        return None
    pooled: List[np.ndarray] = []
    pooled_ess: List[np.ndarray] = []
    detail: Dict[str, Any] = {}
    reps_target = nulls.reps
    for n_raw, count in weights:
        null = nulls.get(phi, n_raw, burn_in)
        if null is None:
            continue
        abs_t = np.abs(np.asarray(null[ch]["samples"]["t_nw"], dtype=np.float64))
        ess = np.asarray(null[ch]["samples"]["ess"], dtype=np.float64)
        take = max(1, int(round(reps_target * count / total)))
        take = min(take, abs_t.size)
        pooled.append(abs_t[:take])
        pooled_ess.append(ess[:take] / max(n_raw - int(burn_in), 1))
        detail[str(n_raw)] = {
            "count": count,
            "median_abs_t": float(np.median(abs_t)),
            "n_kept": int(max(n_raw - int(burn_in), 0)),
            "reps_used": int(take),
            "seed": null["seed"],
            "weight": count / total,
        }
    if not pooled:
        return None
    return {
        "abs_t": np.concatenate(pooled),
        "by_segment_len": detail,
        "ess_over_n": np.concatenate(pooled_ess),
        "phi": float(np.clip(phi, -PHI_CLAMP, PHI_CLAMP)),
    }


def _ratio_with_error(
    num: Optional[Dict[str, Any]], den: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """``ratio`` plus the interval it has never been printed with.

    The numerator is a block-bootstrap median over segments; the denominator
    is a Monte-Carlo median over ``--null-reps`` draws and carries ~2.8%
    (dc) / ~3.1% (alt) sd of its own at reps = 2000.  On the pooled cells the
    denominator's error alone EXCEEDS the numerator's CI half-width, so a
    bare `ratio` printed to three decimals overstates its own resolution.

    Both bootstraps report an ``se``; the two are independent by construction
    (different data, different generators), so the delta-method combination
    on the log scale is exact to first order:

        se(ratio)/ratio = sqrt( (se_num/num)^2 + (se_den/den)^2 ) .
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
        "ci95": [ratio - 1.959964 * se_ratio, ratio + 1.959964 * se_ratio],
        "denominator_rel_se": float(den.get("se", float("nan"))) / b
        if b > 0.0 else None,
        "numerator_rel_se": float(num.get("se", float("nan"))) / a
        if a > 0.0 else None,
        "ratio": ratio,
        "se": se_ratio,
    }


def summarize_cell(
    data: Dict[str, np.ndarray],
    key: Tuple[Any, ...],
    n_runs: int,
    nulls: NullBank,
    max_lag: int,
    boot: Dict[str, int],
) -> Dict[str, Any]:
    """All reported quantities for one (kind, lr, batch, burn_in) cell.

    The two channels are bootstrapped with the SAME seed and therefore the
    same resample index sets.  That is deliberate, not an oversight: the
    dc-vs-alt contrast is the point of the report, and common random numbers
    remove the resampling noise from their difference.  It also means the
    two intervals are not independent of each other and must not be combined
    as if they were.
    """
    kind, lr, batch, burn_in = key
    rho = data["rho"]
    weights = _length_weights(data["n_raw"])
    n_raw_median = int(round(float(np.median(data["n_raw"]))))
    rho_median = [_median(rho[:, k]) for k in range(max_lag)]
    phi_hat = rho_median[0]
    n_kept_median = _median(data["n_kept"])
    order = np.argsort(data["slot_rank"], kind="stable")
    out: Dict[str, Any] = {
        "batch_size": batch,
        "burn_in": int(burn_in),
        "channels": {},
        "kind": kind,
        "lr": lr,
        "lag_truncation_at_median_n": newey_west_bandwidth(
            int(round(n_kept_median)) if n_kept_median else 0, max_lag
        ),
        "n_kept_median": n_kept_median,
        "n_obs_kept": int(np.nansum(data["n_kept"])),
        "n_runs": int(n_runs),
        "n_segments": int(rho.shape[0]),
        "phi_hat": phi_hat,
        "rho_median": rho_median,
        "rho_raw_median": [_median(data["rho_raw"][:, k]) for k in range(max_lag)],
        "segment_len_raw_counts": {str(n): c for n, c in weights},
        "segment_len_raw_median": n_raw_median,
        "tau": {},
    }
    out["rho_1_ci95"] = _bootstrap(
        rho[:, 0], boot["block"], boot["reps"], boot["seed"], boot["max_elems"]
    )
    out["rho_1_ci95_direction_block"] = _bootstrap(
        rho[order, 0], boot["block_alt"], boot["reps"], boot["seed"],
        boot["max_elems"],
    )
    for k in TAU_LAGS:
        if k <= max_lag:
            terms = [r for r in rho_median[:k] if r is not None]
            out["tau"][str(k)] = 1.0 + 2.0 * float(np.sum(terms)) if terms else None
    # The integrated time the Newey-West estimator ACTUALLY uses: Bartlett
    # weights at the in-force bandwidth L, not a flat lag-K sum.  Its
    # reciprocal is the null's ESS/n, which is the internal check.
    L = out["lag_truncation_at_median_n"]
    terms = [
        (1.0 - j / (L + 1.0)) * rho_median[j - 1]
        for j in range(1, L + 1)
        if j <= max_lag and rho_median[j - 1] is not None
    ]
    out["tau_nw"] = 1.0 + 2.0 * float(np.sum(terms)) if L >= 1 else 1.0

    for ch in sorted(CHANNELS):
        t = data[f"t_nw_{ch}"]
        abs_t = np.abs(t)
        ci = _bootstrap(
            abs_t, boot["block"], boot["reps"], boot["seed"] + 1, boot["max_elems"]
        )
        entry: Dict[str, Any] = {
            "ci95_median_abs_t": ci,
            "ci95_median_abs_t_direction_block": _bootstrap(
                abs_t[order], boot["block_alt"], boot["reps"], boot["seed"] + 1,
                boot["max_elems"],
            ),
            "frac_abs_t_ge": {
                f"{thr:g}": _frac_at_least(abs_t, thr) for thr in T_THRESHOLDS
            },
            "median_abs_t": _median(abs_t),
            "median_abs_t_naive": _median(np.abs(data[f"t_naive_{ch}"])),
            "median_ess_over_n": _median(data[f"ess_{ch}"] / data["n_kept"]),
            "median_t": _median(t),
            "null": None,
            "nw_floored_frac": float(np.mean(data[f"nw_floored_{ch}"])),
            "ratio": None,
            "ratio_ci95": None,
        }
        mix = _null_mixture(nulls, phi_hat, weights, int(burn_in), ch)
        if mix is not None:
            null_abs = mix["abs_t"]
            null_ci = _bootstrap(
                null_abs, 1, boot["reps"], boot["seed"] + 2, boot["max_elems"]
            )
            entry["null"] = {
                "by_segment_len": mix["by_segment_len"],
                "ci95_median_abs_t": null_ci,
                "frac_abs_t_ge": {
                    f"{thr:g}": _frac_at_least(null_abs, thr) for thr in T_THRESHOLDS
                },
                # 1/reps: frac_abs_t_ge cannot resolve anything below this,
                # and the registered reps (prereg 5.6) is 200000, not 2000.
                "frac_abs_t_ge_floor": 1.0 / max(null_abs.size, 1),
                "median_abs_t": float(np.median(null_abs)),
                "median_ess_over_n": float(np.median(mix["ess_over_n"])),
                "n_draws": int(null_abs.size),
                "phi": mix["phi"],
                "reps_per_length": nulls.reps,
                "seed": nulls.seed,
            }
            detail = _ratio_with_error(ci, null_ci)
            if detail is not None:
                entry["ratio"] = detail["ratio"]
                entry["ratio_ci95"] = detail
        out["channels"][ch] = entry
    return out


def summarize_slot_cell(
    data: Dict[str, np.ndarray],
    segment_cell: Dict[str, Any],
    nulls: NullBank,
    burn_in: int,
    boot: Dict[str, int],
    shape_table: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """The prereg §5.1 slot-level read of a cell, and its growth factor.

    ``segment_cell`` is the same cell's per-segment summary; the reported
    ``growth_calibrated`` is (slot ratio) / (segment ratio), i.e. how much
    the null-calibrated |t| grows when the window is lengthened from one
    segment to a whole slot.  The two hypotheses the per-segment ``ratio``
    cannot separate predict different values:

    * a mean that survives a subspace refresh -> growth ~ sqrt(k);
    * zero-mean low-frequency power inside segments -> growth ~ 1, because
      the Newey-West bandwidth grows with N and starts absorbing it.

    ``sqrt_n_segments`` is printed next to it as the reference.  Nulls are
    mixed over the slot SHAPES actually present (the exact tuple of raw
    segment lengths), never a modal or median shape.
    """
    phi_hat = segment_cell["phi_hat"]
    n_seg = data["n_segments"].astype(np.int64)
    ids, counts = np.unique(data["shape_id"].astype(np.int64), return_counts=True)
    shapes = {
        tuple(int(v) for v in shape_table[int(i)]): int(c)
        for i, c in zip(ids, counts)
    }
    out: Dict[str, Any] = {
        "burn_in": int(burn_in),
        "channels": {},
        "median_lag_truncation": _median(data["lag_truncation"]),
        "median_n": _median(data["n"]),
        "median_n_segments": _median(data["n_segments"]),
        "n_slots": int(n_seg.size),
        "phi_hat": phi_hat,
        "shapes": {
            "x".join(str(v) for v in shape): c
            for shape, c in sorted(shapes.items())
        },
        # Median over SLOTS of sqrt(k), not sqrt(median k): the cell mixes
        # slot shapes (k = 2, 4, 8, 16 all occur), so the growth reference is
        # a mixture reference and taking the root of a pooled k would quote a
        # sharper prediction than the data supports.
        "sqrt_n_segments": _median(np.sqrt(np.maximum(data["n_segments"], 1.0))),
    }
    total = float(sum(shapes.values()))
    for ch in sorted(CHANNELS):
        abs_t = np.abs(data[f"t_nw_{ch}"])
        ci = _bootstrap(
            abs_t, boot["block"], boot["reps"], boot["seed"] + 3, boot["max_elems"]
        )
        entry: Dict[str, Any] = {
            "ci95_median_abs_t": ci,
            "growth_calibrated": None,
            "median_abs_t": _median(abs_t),
            "median_abs_t_naive": _median(np.abs(data[f"t_naive_{ch}"])),
            "median_ess_over_n": _median(data[f"ess_{ch}"] / data["n"]),
            "median_segment_mean_acf1": _median(data[f"acf1_{ch}"]),
            "median_segment_mean_coherence": _median(data[f"coherence_{ch}"]),
            "null": None,
            "ratio": None,
            "ratio_ci95": None,
        }
        pooled: List[np.ndarray] = []
        coh: List[np.ndarray] = []
        detail: Dict[str, Any] = {}
        for shape, count in sorted(shapes.items()):
            null = nulls.get_slot(phi_hat, shape, len(shape), int(burn_in))
            if null is None:
                continue
            samples = np.abs(
                np.asarray(null[ch]["samples"]["t_nw"], dtype=np.float64)
            )
            take = min(max(1, int(round(nulls.reps * count / total))), samples.size)
            pooled.append(samples[:take])
            coh.append(
                np.asarray(
                    null[ch]["samples"]["segment_mean_coherence"], dtype=np.float64
                )[:take]
            )
            detail["x".join(str(v) for v in shape)] = {
                "count": count,
                "median_abs_t": float(np.median(samples)),
                "n_kept": null["n"],
                "reps_used": int(take),
                "seed": null["seed"],
            }
        if pooled:
            null_abs = np.concatenate(pooled)
            null_ci = _bootstrap(
                null_abs, 1, boot["reps"], boot["seed"] + 4, boot["max_elems"]
            )
            entry["null"] = {
                "by_shape": detail,
                "ci95_median_abs_t": null_ci,
                "median_abs_t": float(np.median(null_abs)),
                "median_segment_mean_coherence": float(
                    np.median(np.concatenate(coh))
                ),
                "n_draws": int(null_abs.size),
            }
            detail_ratio = _ratio_with_error(ci, null_ci)
            if detail_ratio is not None:
                entry["ratio"] = detail_ratio["ratio"]
                entry["ratio_ci95"] = detail_ratio
                seg_ratio = segment_cell["channels"][ch]["ratio"]
                if seg_ratio:
                    entry["growth_calibrated"] = detail_ratio["ratio"] / seg_ratio
        out["channels"][ch] = entry
    return out


def pool(
    cells: Dict[Tuple[Any, ...], Dict[str, np.ndarray]],
    keys: Iterable[Tuple[Any, ...]],
) -> Dict[str, np.ndarray]:
    """Concatenate several cells' per-segment rows, preserving ingest order."""
    keys = [k for k in keys if k in cells]
    fields = sorted(cells[keys[0]])
    return {f: np.concatenate([cells[k][f] for k in keys]) for f in fields}


# ------------------------------------------------------------------ report


def cell_key_str(kind: str, lr: Any, batch: Any) -> str:
    """Sortable, numeric-order-preserving cell label.

    ``kind/lrLL.LL/bBBBBB`` with zero padding so lexicographic key order is
    numeric order, and with ``/`` rather than ``|`` so the label can be
    dropped into a markdown table cell unescaped.  ``None`` means "pooled
    over this axis" and sorts after every concrete value.
    """
    lr_s = "lrALL" if lr is None else f"lr{float(lr):05.2f}"
    b_s = "bALL" if batch is None else f"b{int(batch):05d}"
    return f"{kind}/{lr_s}/{b_s}"


def phase_a_ledger(
    strata: Dict[str, Dict[str, Dict[str, Any]]], primary: int
) -> Dict[str, Any]:
    """Reproduction status of the peeked prereg §2 anchors A1-A5.

    Prereg §2 registers, verbatim: "Before any Phase B run,
    scripts/analyze_channel_audit.py ... must reproduce A1-A5
    deterministically ... Any disagreement between the numbers quoted above
    and the reproduced ones is reported as an amendment to this file."

    This function does not decide anything (CLAUDE.md ground rule 1).  It
    puts each anchor next to what this script actually produces and marks the
    obligation's status, so a reader cannot mistake "the script ran" for "the
    launch precondition is met".  The status is OPEN whenever any anchor is
    NOT_REPRODUCIBLE or DISAGREES, and that is expected here: this script's
    unit is the refresh segment (N ~ 45) and the registered unit is the slot
    (N ~ 720), so magnitudes are not comparable rung-for-rung, and A5 needs
    the beta = 0.9 / 0.99 tier contrast, which this script does not compute at
    all.
    """
    rows: List[Dict[str, Any]] = []
    by_lr = strata["by_kind_lr"]
    by_kind = strata["by_kind"]

    def cell(stratum, label, burn_in):
        return (stratum.get(label) or {}).get(str(burn_in))

    # A1 -- phi_hat across lr, top tier.
    top_lr = sorted(
        (c["lr"], c["phi_hat"])
        for lab, byb in by_lr.items()
        for c in [byb.get(str(primary))]
        if c is not None and c["kind"] == "top" and c["phi_hat"] is not None
    )
    a1: Dict[str, Any] = {"reproduced": None, "status": "NOT_REPRODUCIBLE"}
    if top_lr:
        lo, hi = min(v for _l, v in top_lr), max(v for _l, v in top_lr)
        pooled = cell(by_kind, "top/lrALL/bALL", primary)
        a1 = {
            "reproduced": {
                "phi_hat_pooled_top": None if pooled is None else pooled["phi_hat"],
                "phi_hat_top_max": hi,
                "phi_hat_top_min": lo,
                "phi_hat_top_span": hi - lo,
            },
            "status": "DISAGREES" if (hi - lo) > 0.10 else "AGREES",
            "note": (
                "phi_hat is NOT lr-invariant: over the lr ladder it runs "
                f"{lo:.3f} to {hi:.3f}, a span of {hi - lo:.3f}, against A1's "
                "single '-0.34, LR-invariant'. The pooled value may still "
                "land near -0.34; the invariance clause is what disagrees, "
                "and it needs an amendment. Separately, this column is an "
                "ESTIMATOR value: the +1/n ladder correction leaves a "
                "measured +0.014 residual at phi = -0.34, n = 45, so a "
                "reproduced -0.343 corresponds to a true phi of about -0.358 "
                "(src/stats/spectral.py module docstring has the measured "
                "map)."
            ),
        }
    rows.append(dict(PHASE_A_ANCHORS[0], **a1))

    # A2 -- rho_2 at burn-in 0 against burn-in 5.
    a2: Dict[str, Any] = {"reproduced": None, "status": "NOT_REPRODUCIBLE"}
    zero, five = cell(by_kind, "top/lrALL/bALL", 0), cell(by_kind, "top/lrALL/bALL", 5)
    if zero is not None and five is not None:
        r2_0, r2_5 = zero["rho_median"][1], five["rho_median"][1]
        ar1_pred = five["phi_hat"] ** 2 if five["phi_hat"] is not None else None
        a2 = {
            "reproduced": {
                "ar1_prediction_at_burn_in_5": ar1_pred,
                "rho_2_burn_in_0": r2_0,
                "rho_2_burn_in_5": r2_5,
            },
            "status": "AGREES" if (r2_0 is not None and r2_0 < 0.5 * (ar1_pred or 1.0))
            else "DISAGREES",
            "note": (
                "burn-in 0 is carried in the sweep for exactly this anchor; "
                "the registered sweep {5, 15, 25} cannot exhibit a transient "
                "confined to the head of a segment."
            ),
        }
    rows.append(dict(PHASE_A_ANCHORS[1], **a2))

    # A3 -- alternating channel flat at 0.75-0.85.
    alt = [
        c["channels"]["alt"]["median_abs_t"]
        for byb in by_lr.values()
        for b, c in byb.items()
        if int(b) in REGISTERED_BURN_INS
        and c["channels"]["alt"]["median_abs_t"] is not None
    ]
    a3: Dict[str, Any] = {"reproduced": None, "status": "NOT_REPRODUCIBLE"}
    if alt:
        a3 = {
            "reproduced": {
                "alt_median_abs_t_max": max(alt),
                "alt_median_abs_t_min": min(alt),
                "n_cells": len(alt),
            },
            "status": "AGREES" if 0.70 <= min(alt) and max(alt) <= 0.95
            else "DISAGREES",
            "note": (
                "range over every kind x lr cell at the REGISTERED burn-ins "
                f"{list(REGISTERED_BURN_INS)}; burn-in 0 is excluded because "
                "A3 is stated on burn-in-cleaned segments."
            ),
        }
    rows.append(dict(PHASE_A_ANCHORS[2], **a3))

    # A4 -- DC excess peak in lr, and its burn-in 25 value.
    a4: Dict[str, Any] = {"reproduced": None, "status": "NOT_REPRODUCIBLE"}
    dc_by_lr = [
        (c["lr"], c["channels"]["dc"]["median_abs_t"])
        for byb in by_lr.values()
        for c in [byb.get(str(primary))]
        if c is not None and c["kind"] == "top"
        and c["channels"]["dc"]["median_abs_t"] is not None
    ]
    if dc_by_lr:
        peak_lr, peak = max(dc_by_lr, key=lambda kv: kv[1])
        at_096 = [v for lr, v in dc_by_lr if lr is not None and abs(lr - 0.96) < 1e-9]
        b25 = (by_lr.get("top/lr00.96/bALL") or {}).get("25")
        a4 = {
            "reproduced": {
                "dc_median_abs_t_at_lr_0.96": at_096[0] if at_096 else None,
                "dc_median_abs_t_at_lr_0.96_burn_in_25": None if b25 is None
                else b25["channels"]["dc"]["median_abs_t"],
                "dc_median_abs_t_peak": peak,
                "dc_median_abs_t_peak_lr": peak_lr,
            },
            "status": "AGREES" if at_096 and abs(at_096[0] - 3.77) < 0.5
            else "DISAGREES",
            "note": (
                "A4's 3.77 is a median |t_dc|, not a ratio; compare it "
                "against the median |t| column, not the null-calibrated one. "
                "The lr of the peak is a separate claim from its height and "
                "both are listed."
            ),
        }
    rows.append(dict(PHASE_A_ANCHORS[3], **a4))

    # A5 -- not computable here at all.
    rows.append(dict(
        PHASE_A_ANCHORS[4],
        reproduced=None,
        status="NOT_REPRODUCIBLE",
        note=(
            "the beta = 0.9 vs 0.99 tier contrast is a frozen-probe-tier "
            "quantity; this script reads tracked-direction sidecars only and "
            "has no producer for it. Prereg 6b assigns it to "
            "scripts/analyze_channel_audit_frozen.py, which does not exist."
        ),
    ))

    open_rows = [r for r in rows if r["status"] != "AGREES"]
    return {
        "anchors": rows,
        "obligation": "prereg 2 Phase A reproducibility obligation",
        "status": "OPEN" if open_rows else "DISCHARGED",
        "unreproduced": sorted(r["anchor"] for r in open_rows),
        "why_open": (
            "This script's unit of analysis is the refresh segment (N ~ 45); "
            "prereg 5.1 registers the slot (N ~ 720). |t| scales like "
            "sqrt(N), so A3/A4 magnitudes are not rung-for-rung comparable "
            "with the registered unit even when they numerically agree, and "
            "A5 has no producer. The obligation is therefore NOT discharged "
            "by running this script, and the disagreements listed above are "
            "amendments owed to reports/channel-audit-preregistration.md 2."
        ) if open_rows else "",
    }


def build_report(
    store: CellStore,
    slots: SlotStore,
    meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Assemble every table of the audit into one sorted-key dict."""
    cells = store.finalize()
    slot_cells = slots.finalize()
    max_lag = store.max_lag
    nulls = NullBank(args.null_reps, args.null_seed, max_lag)
    boot = {
        "block": args.bootstrap_block,
        "block_alt": args.bootstrap_block_alt,
        "max_elems": args.bootstrap_max_elems,
        "reps": args.bootstrap_reps,
        "seed": args.bootstrap_seed,
    }
    runs_by_key = store.runs

    strata: Dict[str, Dict[str, Dict[str, Any]]] = {
        "by_cell": {},
        "by_kind": {},
        "by_kind_batch": {},
        "by_kind_lr": {},
    }
    slot_strata: Dict[str, Dict[str, Dict[str, Any]]] = {
        "by_kind": {},
        "by_kind_batch": {},
        "by_kind_lr": {},
    }

    def emit(stratum: str, kind: str, lr, batch, burn_in: int, keys) -> None:
        """Summarize the union of ``keys`` into ``strata[stratum]``."""
        keys = [k for k in keys if k in cells]
        if not keys:
            return
        run_set: set = set()
        for k in keys:
            run_set |= runs_by_key.get(k, set())
        label = cell_key_str(kind, lr, batch)
        cell = summarize_cell(
            pool(cells, keys), (kind, lr, batch, burn_in), len(run_set),
            nulls, max_lag, boot,
        )
        strata[stratum].setdefault(label, {})[str(burn_in)] = cell
        if stratum == "by_cell" or args.no_slot_level:
            return
        if stratum != "by_kind" and burn_in != args.primary_burn_in:
            return
        skeys = [k for k in keys if k in slot_cells]
        if not skeys:
            return
        slot_strata[stratum].setdefault(label, {})[str(burn_in)] = (
            summarize_slot_cell(
                pool(slot_cells, skeys), cell, nulls, burn_in, boot,
                meta.get("slot_shape_table", []),
            )
        )

    lrs, batches = meta["lrs"], meta["batch_sizes"]
    burn_ins = sorted({k[3] for k in cells})
    n_populated = 0
    for kind in KINDS:
        for burn_in in burn_ins:
            for lr in lrs:
                for batch in batches:
                    if (kind, lr, batch, burn_in) in cells:
                        n_populated += burn_in == args.primary_burn_in
                    emit("by_cell", kind, lr, batch, burn_in,
                         [(kind, lr, batch, burn_in)])
                emit("by_kind_lr", kind, lr, None, burn_in,
                     [(kind, lr, b, burn_in) for b in batches])
            for batch in batches:
                emit("by_kind_batch", kind, None, batch, burn_in,
                     [(kind, lr, batch, burn_in) for lr in lrs])
            emit("by_kind", kind, None, None, burn_in,
                 [(kind, lr, b, burn_in) for lr in lrs for b in batches])

    diagnostics = {
        "mirror_check": meta["mirror_check"],
        "n_directions": meta["n_directions"],
        "n_observations": meta["n_observations"],
        "n_refresh_without_reset": meta["n_refresh_without_reset"],
        "n_refreshes": meta["n_refreshes"],
        "n_segment_stats_dropped_short": meta["n_segments_dropped_short"],
        "n_segment_stats_used": meta["n_segments_used"],
        "n_slot_stats_used": meta["n_slots_used"],
        "segment_start_parity": {
            "n_odd": meta["n_segment_start_parity_odd"],
            "n_starts": meta["n_segment_starts"],
            # A mixed-parity segment set would make the alternating channel's
            # absolute-step demodulation flip sign between segments of one
            # slot, so the pooled alt mean would cancel a real signal.
            "uniform": meta["n_segment_start_parity_odd"]
            in (0, meta["n_segment_starts"]),
        },
        "skipped_sidecars": meta["skipped"],
        "slot_shapes": meta["slot_shapes"],
        "slot_shape_table": meta.get("slot_shape_table", []),
    }
    if meta["n_refreshes"]:
        diagnostics["frac_refresh_without_reset"] = (
            meta["n_refresh_without_reset"] / meta["n_refreshes"]
        )
    if args.self_test_reps > 0:
        seg_len = int(
            np.median([c["segment_len_raw_median"]
                       for by_b in strata["by_kind"].values()
                       for c in by_b.values()])
        )
        n_seg_slot = int(max(
            np.median([
                c["median_n_segments"] or 1
                for by_b in slot_strata["by_kind"].values()
                for c in by_b.values()
            ]) if slot_strata["by_kind"] else 4,
            2,
        ))
        diagnostics["self_test"] = self_test(
            args.self_test_reps, args.primary_burn_in, max_lag, seg_len,
            n_seg_slot, nulls,
        )
    return {
        "config": {
            "bootstrap_block": args.bootstrap_block,
            "bootstrap_block_alt": args.bootstrap_block_alt,
            "bootstrap_max_elems": args.bootstrap_max_elems,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "burn_ins": list(map(int, burn_ins)),
            "max_lag": max_lag,
            "min_n": args.min_n,
            "null_reps": args.null_reps,
            "null_seed": args.null_seed,
            "primary_burn_in": args.primary_burn_in,
            "registered_burn_ins": list(REGISTERED_BURN_INS),
            "segment_at": args.segment_at,
            "self_test_reps": args.self_test_reps,
            "slot_level": not args.no_slot_level,
            "t_thresholds": list(T_THRESHOLDS),
            "unit": "segment",
        },
        "diagnostics": diagnostics,
        "inputs": {
            "batch_sizes": meta["batch_sizes"],
            "cells_populated": n_populated,
            "cells_possible": len(KINDS) * len(lrs) * len(batches),
            "lrs": meta["lrs"],
            "matrices": meta["matrices"],
            "n_runs": len(meta["runs"]),
            "n_sidecars_selected": len(meta["runs"]) + len(meta["skipped"]),
            "runs": sorted(meta["runs"], key=lambda r: r["run"]),
            "seeds": meta["seeds"],
            "selection": meta.get("selection", {}),
        },
        "phase_a_reproduction": phase_a_ledger(strata, args.primary_burn_in),
        "slot_strata": slot_strata,
        "strata": strata,
        "tier": "tracked",
    }


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    if isinstance(x, float) and not np.isfinite(x):
        return "n/a"
    return f"{float(x):.{nd}f}"


def _ci(entry: Optional[Dict[str, Any]], nd: int = 3) -> str:
    if not entry:
        return "n/a"
    return f"[{_fmt(entry['ci_lo'], nd)}, {_fmt(entry['ci_hi'], nd)}]"


def _pair(values: Optional[Sequence[float]], nd: int = 3) -> str:
    if not values:
        return "n/a"
    return f"[{_fmt(values[0], nd)}, {_fmt(values[1], nd)}]"


def _ladder_rows(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
    out = [
        "| group | n_runs | n_seg | seg len | n_kept | L | rho_1 | rho_2 "
        "| rho_3 | rho_4 | rho_1 raw | rho_1 CI95 | rho_1 CI95 (dir block) "
        "| tau_nw(L) | tau(4) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- | --- | --- |",
    ]
    for label, c in rows:
        r = c["rho_median"] + [None] * 4
        raw = c["rho_raw_median"] + [None] * 4
        out.append(
            f"| {label} | {c['n_runs']} | {c['n_segments']} | "
            f"{c['segment_len_raw_median']} | {_fmt(c['n_kept_median'], 1)} | "
            f"{c['lag_truncation_at_median_n']} | "
            f"{_fmt(r[0])} | {_fmt(r[1])} | {_fmt(r[2])} | {_fmt(r[3])} | "
            f"{_fmt(raw[0])} | {_ci(c['rho_1_ci95'])} | "
            f"{_ci(c.get('rho_1_ci95_direction_block'))} | "
            f"{_fmt(c.get('tau_nw'))} | {_fmt(c['tau'].get('4'))} |"
        )
    return out


def _channel_rows(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
    out = [
        "| group | n_runs | n_seg | ch | median \\|t\\| | CI95 "
        "| null median \\|t\\| | null CI95 | ratio | ratio CI95 "
        "| frac \\|t\\|>=2 | null frac>=2 | frac \\|t\\|>=4 "
        "| null frac>=4 | ESS/n | null ESS/n |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- | --- | --- | --- |",
    ]
    for label, c in rows:
        for ch in sorted(CHANNELS):
            e = c["channels"][ch]
            nl = e["null"] or {}
            nf = nl.get("frac_abs_t_ge") or {}
            floor = nl.get("frac_abs_t_ge_floor")
            rc = e.get("ratio_ci95") or {}
            f4 = nf.get("4")
            f4_s = _fmt(f4, 5)
            if f4 is not None and floor is not None and f4 <= floor:
                f4_s = f"<{floor:.5f}"
            out.append(
                f"| {label} | {c['n_runs']} | {c['n_segments']} | {ch} | "
                f"{_fmt(e['median_abs_t'])} | "
                f"{_ci(e['ci95_median_abs_t'])} | {_fmt(nl.get('median_abs_t'))} | "
                f"{_ci(nl.get('ci95_median_abs_t'))} | "
                f"{_fmt(e['ratio'])} | "
                f"{_pair(rc.get('ci95'))} | "
                f"{_fmt(e['frac_abs_t_ge']['2'])} | "
                f"{_fmt(nf.get('2'))} | "
                f"{_fmt(e['frac_abs_t_ge']['4'])} | {f4_s} | "
                f"{_fmt(e['median_ess_over_n'], 2)} | "
                f"{_fmt(nl.get('median_ess_over_n'), 2)} |"
            )
    return out


def _slot_rows(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
    out = [
        "| group | n_slots | k (median) | N | L | ch | slot median \\|t\\| "
        "| slot null \\|t\\| | slot ratio | growth (cal.) | sqrt(k) "
        "| seg-mean coherence | null coherence | seg-mean acf1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- | --- |",
    ]
    for label, c in rows:
        for ch in sorted(CHANNELS):
            e = c["channels"][ch]
            nl = e["null"] or {}
            out.append(
                f"| {label} | {c['n_slots']} | {_fmt(c['median_n_segments'], 1)} | "
                f"{_fmt(c['median_n'], 0)} | "
                f"{_fmt(c['median_lag_truncation'], 0)} | {ch} | "
                f"{_fmt(e['median_abs_t'])} | {_fmt(nl.get('median_abs_t'))} | "
                f"{_fmt(e['ratio'])} | {_fmt(e['growth_calibrated'])} | "
                f"{_fmt(c['sqrt_n_segments'])} | "
                f"{_fmt(e['median_segment_mean_coherence'])} | "
                f"{_fmt(nl.get('median_segment_mean_coherence'))} | "
                f"{_fmt(e['median_segment_mean_acf1'])} |"
            )
    return out


def _rows_at(stratum: Dict[str, Any], burn_in: int) -> List[Tuple[str, Dict[str, Any]]]:
    key = str(burn_in)
    return [
        (label, by_b[key]) for label, by_b in sorted(stratum.items()) if key in by_b
    ]


def to_markdown(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    inp = report["inputs"]
    diag = report["diagnostics"]
    primary = cfg["primary_burn_in"]
    strata = report["strata"]
    slot_strata = report.get("slot_strata", {})
    ledger = report.get("phase_a_reproduction", {})
    sel = inp.get("selection") or {}
    lines = [
        "# Per-direction channel audit (tracked tier) — EXPLORATORY, PHASE A",
        "",
        "**EXPLORATORY. These numbers are a prior, not evidence.** This report",
        "is offline analysis of sidecars that have ALREADY BEEN PEEKED: the",
        "lag-ladder and channel readings recorded as A1–A5 in",
        "`reports/channel-audit-preregistration.md` §2 were produced by ad-hoc",
        "in-session analysis over exactly this data, before any criterion",
        "existed. Nothing here is unblinded, controlled for multiplicity, or",
        "separable from the tracked tier's period-50 refresh cadence — whose",
        "25th harmonic lands exactly on the 0.5 cyc/step frequency the",
        "alternating channel reads. The confirmatory surface is the",
        "frozen-probe tier re-run on GPU (prereg §3), which does not exist on",
        "disk.",
        "",
        "**Phase A output, at the Phase A path.** Prereg §6a registers this",
        "file as `reports/channel-audit-phase-a.{md,json}` and §6b reserves",
        "`reports/channel-audit.{md,json}` for the CONFIRMATORY Phase B",
        "producer, which is a different script. Writing to the reserved names",
        "is refused by the argument parser.",
        "",
        "Descriptive output of `scripts/analyze_channel_audit.py`: quantities",
        "only. No gate is evaluated and no pass/fail is stated; gate decisions",
        "are human-only (CLAUDE.md ground rule 1).",
        "",
        "## 0. Inputs and estimator",
        "",
        f"- sidecars discovered: {sel.get('n_discovered', 'n/a')}; selected: "
        f"{inp['n_sidecars_selected']}; runs used: {inp['n_runs']}; "
        f"skipped: {len(diag['skipped_sidecars'])}",
        f"- selection filters: run prefix `{sel.get('run_prefix')}`, min seed "
        f"{sel.get('min_seed')}, `results/INVALID_RUNS.json` tombstones — "
        f"excluded {sel.get('n_excluded', 0)} "
        + "(" + ", ".join(
            f"{k}: {len(v)}"
            for k, v in sorted((sel.get("excluded") or {}).items())
        ) + ")",
        f"- observations read: {diag['n_observations']}; direction slots: "
        f"{diag['n_directions']}; matrices: {len(inp['matrices'])}",
        f"- lr grid: {inp['lrs']}",
        f"- batch grid: {inp['batch_sizes']}; seeds: {inp['seeds']}",
        f"- **the grid is not a full factorial**: cells populated "
        f"{inp.get('cells_populated')} of {inp.get('cells_possible')} "
        "(kind × lr × batch); the empty combinations are absent from §4, not "
        "zero",
        f"- unit of analysis: **one refresh segment** (`--segment-at "
        f"{cfg['segment_at']}`), burn-in {cfg['burn_ins']} (registered sweep "
        f"{cfg['registered_burn_ins']}; 0 added for the A2 transient anchor), "
        f"primary {primary}; lag ladder to {cfg['max_lag']}; segments with "
        f"fewer than {cfg['min_n']} post-burn-in observations dropped",
        f"- segment statistics used: {diag['n_segment_stats_used']} "
        f"(dropped short: {diag['n_segment_stats_dropped_short']}), summed "
        "over the burn-in sweep; slot statistics used: "
        f"{diag['n_slot_stats_used']}",
        f"- AR(1) surrogate null: {cfg['null_reps']} reps per "
        "(phi_hat, **each distinct raw segment length in the cell**, "
        "burn-in), mixed at the cell's own length frequencies, seeded from "
        f"those parameters and base seed {cfg['null_seed']}",
        f"- block bootstrap: block {cfg['bootstrap_block']} "
        "(= one run × matrix × segment cluster of direction slots), "
        f"{cfg['bootstrap_reps']} reps at every pool size, seed "
        f"{cfg['bootstrap_seed']}; the registered direction-block "
        f"({cfg['bootstrap_block_alt']} segments of one direction, prereg "
        "§5.8) is reported beside it in §1",
        "",
        "### What `ratio` does not mean",
        "",
        "`ratio` = observed median |t_nw| / null median |t_nw| is a **scale**",
        "correction, not an identification. Its numerator is the",
        "zero-frequency content of a 45-observation window; its denominator",
        "integrates lags 1..L with Bartlett weights and L = 3 at n = 45. Any",
        "zero-mean component slower than ~3 steps is therefore absent from",
        "the denominator and fully present in the numerator, and an AR(1)",
        "null fitted to ρ₁ has a correlation time under one step by",
        "construction. A **zero-mean** stream (fast AR(1) φ = −0.40 plus slow",
        "AR(1) φ = 0.97 at variance 0.15) run through this exact pipeline",
        "returns ρ₁ = −0.33, dc median |t| = 1.78, dc ratio = 2.71, alt ratio",
        "= 1.04 — numerically the pooled `top` cell below, with no mean",
        "anywhere in it. It is control **K3** in the estimator-control table",
        "of §8, and §7 is the discriminator on the real data.",
        "",
        "### Deviations from the DRAFT pre-registration, stated up front",
        "",
        "1. **Unit.** Prereg §5.1 registers the SLOT-level estimator (a",
        "   slot's post-burn-in segments concatenated, N ≈ 720 at burn-in 5).",
        "   Sections 1–5 use the single segment (N ≈ 45). |t| scales like",
        "   √N, so those magnitudes are not comparable rung-for-rung with a",
        "   slot-level reading; the null calibration is matched to the same",
        "   short window, so the *ratios* are. §7 restores the registered",
        "   slot-level unit as a diagnostic.",
        f"2. **`--null-reps` = {cfg['null_reps']}, not the registered 200000**",
        "   (§5.6). The registration states why: exceedance rates of order",
        "   1e-4 cannot be resolved by 2000 draws. `null frac>=4` is printed",
        "   as `<floor` when it sits at or below its own 1/reps resolution",
        "   floor, and the null median's Monte-Carlo error is carried into",
        "   every `ratio CI95`.",
        f"3. **`--bootstrap-block` = {cfg['bootstrap_block']} is not the",
        "   registered block** (§5.8 registers \"the 4 segments of one",
        "   direction\" for tracked per-segment statistics). The two are",
        "   orthogonal clusterings of the same table; both are computed and",
        "   §1 prints them side by side.",
        "4. **Nulls are mixed over each cell's actual raw segment lengths**,",
        "   not drawn once at its median length (§5.6 asks for every distinct",
        "   length in the design). This is a repair: the median put every",
        "   b04000 cell's null at raw length 48, which does not occur in that",
        "   cell at all.",
        f"5. **Burn-in 0 is carried** alongside the registered",
        f"   {cfg['registered_burn_ins']} (§5.3), because it is the only value",
        "   that can exhibit the A2 transient. Every criterion is still read",
        f"   at {primary}.",
        "6. **Batched kernel.** The per-segment and per-slot statistics are",
        "   computed by batched NumPy kernels and checked against",
        "   `src.stats.spectral` on a seeded uniform sample whose coverage",
        "   (slot positions, raw segment lengths) is printed in §6.",
        "",
    ]

    # ---- Phase A reproduction ledger -------------------------------------
    if ledger:
        lines += [
            f"## 0b. Phase A reproduction obligation — **{ledger['status']}**",
            "",
            "Prereg §2 registers, verbatim: *\"Before any Phase B run,",
            "`scripts/analyze_channel_audit.py` … must reproduce A1–A5",
            "deterministically … Any disagreement between the numbers quoted",
            "above and the reproduced ones is reported as an amendment to this",
            "file.\"* This section is that reproduction. It states quantities",
            "and an obligation status; it evaluates no gate.",
            "",
            "| anchor | registered claim | reproduced | status |",
            "| --- | --- | --- | --- |",
        ]
        for row in ledger["anchors"]:
            got = row.get("reproduced")
            got_s = "—" if not got else "; ".join(
                f"{k} = {_fmt(v, 3) if isinstance(v, float) else v}"
                for k, v in sorted(got.items())
            )
            lines.append(
                f"| {row['anchor']} | {row['claim']} | {got_s} "
                f"| **{row['status']}** |"
            )
        lines += [""]
        for row in ledger["anchors"]:
            if row.get("note"):
                lines.append(f"- **{row['anchor']}**: {row['note']}")
        if ledger["status"] != "DISCHARGED":
            lines += [
                "",
                f"**The §2 obligation is OPEN** (unreproduced: "
                f"{', '.join(ledger['unreproduced'])}). {ledger['why_open']}",
            ]
        lines += [""]

    lines += [
        f"## 1. Lag ladder, kind × lr (burn-in {primary})",
        "",
        "Bias-corrected `rho_k = c_k/c_0 + 1/n` per segment, median over",
        "segments; `rho_1 raw` is the same median without the +1/n correction",
        "(computed from the raw ladder, so variance-floored rows read 0 and",
        "not −1/n).",
        "",
        "`L` is the Newey-West bandwidth actually in force at this window",
        "(`min(max_lag, floor(4(n/100)^(2/9)), n−2)`), and **`tau_nw(L)` is",
        "the integrated autocorrelation time that bandwidth implies**:",
        "`1 + 2*sum_{j<=L} (1 − j/(L+1)) rho_j`, Bartlett-weighted. Its",
        "reciprocal is the inflation factor that bandwidth applies (1.889 on",
        "pooled `top`, against the 1.854 the null's own ESS/n column",
        "measures); the two are not algebraically equal, because ESS/n is a",
        "median of per-segment ratios and this is a ratio of medians, so read",
        "the agreement as a sanity check and not as an identity. `tau(4)` is",
        "the flat, unweighted",
        "lag-4 sum, kept for continuity with the pre-registration's τ(K)",
        "family — it is **not** what the estimator uses, and at the tracked",
        "window the bandwidth is L = 3, not 4 (L = 4 is the frozen tier's, at",
        "n ∈ {187, 195}). On pooled `top` the two read 0.529 and 0.405, 30%",
        "apart.",
        "",
        "`rho_1` is an estimator value, not φ: the +1/n correction leaves a",
        "measured +0.014 residual at φ = −0.34, n = 45 (2× the printed CI",
        "half-width), so −0.343 corresponds to a true φ of about −0.358. See",
        "the `src/stats/spectral.py` module docstring for the measured map.",
        "",
    ]
    lines += _ladder_rows(_rows_at(strata["by_kind_lr"], primary))
    lines += [
        "",
        f"## 2. Channels, kind × lr (burn-in {primary})",
        "",
        "`ratio` is the null-calibrated statistic: observed median |t_nw|",
        "divided by the median |t_nw| of an AR(1) surrogate null matched to",
        "that cell's own `phi_hat` and mixed over its own raw segment",
        "lengths, pushed through the identical estimator. Under a correct",
        "null `ratio ≈ 1` — but read `ratio CI95`, not `ratio`: the",
        "denominator is a Monte-Carlo median over `--null-reps` draws and",
        "carries ~3% error of its own, which on the pooled cells exceeds the",
        "numerator's CI half-width. A ratio whose CI95 contains 1.0 is not a",
        "measurement of anything.",
        "",
        "Two further limits of the denominator, neither of them in that CI:",
        "",
        "- **The null is matched on ρ₁ only.** Fitting a Yule-Walker AR(8) to",
        "  a pooled ladder and re-drawing (20k zero-mean streams through the",
        "  identical estimator) raises the null median |t| by 0.4–4.4%",
        "  depending on cell and channel, so these ratios are *overstated* by",
        "  up to ~4.5%, by a channel-dependent amount (measured at this",
        "  report's own pooled ladders: top dc 4.4%, top alt 4.1%, bulk dc",
        "  0.4%, bulk alt 3.9%). No contrast changes sign, but `ratio ≈ 1` is",
        "  not attainable to better than a few percent.",
        "- **`phi_hat` is plugged in, not inverted.** The null is drawn with",
        "  the cell's measured `rho_1` as its TRUE φ, so running this ladder",
        "  estimator on the null's own streams returns about `rho_1 + 0.014`",
        "  rather than `rho_1` — the surrogate does not reproduce the",
        "  statistic it is matched on. Redrawing at the bias-inverted φ moves",
        "  the null median |t| by −0.4% (dc) and +0.8% (alt) at the `top`",
        "  ladder, i.e. inside the Monte-Carlo floor above.",
        "",
    ]
    lines += _channel_rows(_rows_at(strata["by_kind_lr"], primary))
    lines += [
        "",
        f"## 3. Channels, kind × batch (burn-in {primary})",
        "",
    ]
    lines += _channel_rows(_rows_at(strata["by_kind_batch"], primary))
    lines += [
        "",
        f"## 4. Full cells, kind × lr × batch (burn-in {primary})",
        "",
        f"{inp.get('cells_populated')} of {inp.get('cells_possible')} "
        "kind × lr × batch cells exist on disk; the rest are absent, not zero.",
        "",
    ]
    lines += _channel_rows(_rows_at(strata["by_cell"], primary))
    lines += [
        "",
        "## 5. Burn-in sensitivity (pooled over lr and batch)",
        "",
        "Every quantity above is read at burn-in 5; this table is the same",
        f"pooled read at {cfg['burn_ins']}. Burn-in 0 is the A2 row: the",
        "re-anchoring transient at the head of a segment contaminates the",
        "ladder and does not average out over segments, and 0 is the only",
        "value that can show it — the {5, 15, 25} rows are nested windows of",
        "the SAME segments (n_kept 45/35/25), so their |t| shrinks mechanically",
        "like sqrt(n_kept) whether or not a transient exists.",
        "",
        "| kind | burn-in | n_seg | n_kept | rho_1 | rho_2 | rho_3 | rho_4 "
        "| tau_nw(L) | dc median \\|t\\| | dc ratio | alt median \\|t\\| "
        "| alt ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- |",
    ]
    for label, by_b in sorted(strata["by_kind"].items()):
        for burn_in in cfg["burn_ins"]:
            c = by_b.get(str(burn_in))
            if c is None:
                continue
            r = c["rho_median"] + [None] * 4
            dc, alt = c["channels"]["dc"], c["channels"]["alt"]
            lines.append(
                f"| {label} | {burn_in} | {c['n_segments']} | "
                f"{_fmt(c['n_kept_median'], 1)} | {_fmt(r[0])} | "
                f"{_fmt(r[1])} | {_fmt(r[2])} | {_fmt(r[3])} | "
                f"{_fmt(c.get('tau_nw'))} | {_fmt(dc['median_abs_t'])} | "
                f"{_fmt(dc['ratio'])} | {_fmt(alt['median_abs_t'])} | "
                f"{_fmt(alt['ratio'])} |"
            )
    mc = diag["mirror_check"]
    lines += [
        "",
        "## 6. Diagnostics",
        "",
        f"- mirror check against `src.stats.spectral` on "
        f"{mc.get('n_checked', 0)} seeded-uniform segments and "
        f"{mc.get('n_slot_checked', 0)} slots — coverage: slot positions "
        f"{mc.get('slot_positions_checked')}, raw segment lengths "
        f"{mc.get('segment_lengths_checked')}: "
        + ", ".join(
            f"{k[len('max_abs_dev_'):]} {mc[k]:.3e}"
            for k in sorted(mc)
            if k.startswith("max_abs_dev_")
        )
        + (" (batched kernel vs canonical scalar functions)"
           if mc.get("n_checked") else "not run (--verify-blocks 0)"),
        f"- segment-start parity: {diag['segment_start_parity']['n_odd']} of "
        f"{diag['segment_start_parity']['n_starts']} starts are odd — uniform: "
        f"{diag['segment_start_parity']['uniform']}. Mixed parity would flip "
        "the alternating channel's sign between segments of one slot and "
        "cancel a real signal in §7; the pooling refuses to run on a "
        "mixed-parity set.",
        f"- slot shapes (raw segment lengths per slot, ×slots): "
        f"{diag.get('slot_shapes')}",
        f"- refreshes that did not reset the direction (alignment >= "
        f"align_min): {diag['n_refresh_without_reset']} of "
        f"{diag['n_refreshes']} "
        f"({_fmt(diag.get('frac_refresh_without_reset'))}); "
        "with `--segment-at refresh` those are still cut, with "
        "`--segment-at reset` they are not",
        "- Newey-West flooring, per channel (pooled, burn-in "
        f"{primary}):",
    ]
    for label, by_b in sorted(strata["by_kind"].items()):
        c = by_b.get(str(primary))
        if c is None:
            continue
        lines.append(
            f"  - {label}: dc {_fmt(c['channels']['dc']['nw_floored_frac'], 4)}, "
            f"alt {_fmt(c['channels']['alt']['nw_floored_frac'], 4)}"
        )
    lines.append(
        "- the two channels are bootstrapped with the same seed and therefore "
        "the same resample index sets (common random numbers): deliberate, "
        "because the dc-vs-alt contrast is the point, but it means the two "
        "intervals are not independent of each other."
    )

    # ---- slot-level section ---------------------------------------------
    if slot_strata.get("by_kind"):
        lines += [
            "",
            "## 7. Slot-level read (prereg §5.1 unit) — the discriminator",
            "",
            "Sections 1–6 cannot distinguish a persistent per-direction mean",
            "from zero-mean low-frequency power inside a segment; §0's K3",
            "control shows a zero-mean stream reproducing the whole `top` dc",
            "cell. This section pools each slot's segments into the",
            "pre-registered slot-level statistic (lag products never crossing",
            "a segment boundary) and reports the **calibrated growth factor**",
            "= slot ratio / segment ratio. The two hypotheses predict",
            "different values:",
            "",
            "- a mean that survives a subspace refresh → growth ≈ √k;",
            "- zero-mean low-frequency power inside segments → growth ≈ 1,",
            "  because the Newey-West bandwidth grows with N (L = 3 at n = 45,",
            "  L = 4 at N = 180, L = 5 at N ≈ 360) and begins absorbing that",
            "  power.",
            "",
            "`seg-mean coherence` is the fraction of a slot's segment means",
            "sharing the modal sign (null reference in the next column);",
            "`seg-mean acf1` is their lag-1 autocorrelation about zero, whose",
            "ceiling is (k−1)/k, not 1. Nulls are drawn at the exact slot",
            "shapes present, never a modal or median shape.",
            "",
            f"### 7a. Pooled, per burn-in",
            "",
        ]
        for burn_in in cfg["burn_ins"]:
            rows = _rows_at(slot_strata["by_kind"], burn_in)
            if not rows:
                continue
            lines += [f"burn-in {burn_in}:", ""]
            lines += _slot_rows(rows)
            lines += [""]
        for name, title in (
            ("by_kind_lr", f"7b. kind × lr (burn-in {primary})"),
            ("by_kind_batch", f"7c. kind × batch (burn-in {primary})"),
        ):
            rows = _rows_at(slot_strata.get(name, {}), primary)
            if not rows:
                continue
            lines += [f"### {title}", ""]
            lines += _slot_rows(rows)
            lines += [""]

    st = diag.get("self_test")
    if st:
        lines += [
            "",
            "## 8. Estimator control (synthetic, seeded)",
            "",
            f"{st['reps']} synthetic direction slots of {st['n_segments']} "
            f"segments × {st['segment_length']} steps at burn-in "
            f"{st['burn_in']}, through the identical `segment_block` / "
            "`SlotAccumulator` / surrogate-null path. On a zero-mean stream "
            "there is nothing to detect, so both `ratio` columns read 1 "
            "whatever the autocorrelation (K1). K2a plants an alternating "
            "mean and must move `alt` only; K2b plants a per-slot DC mean and "
            "must move `dc` **and** drive the growth factor to "
            f"√k = {st['sqrt_n_segments']:.3f}. **K3 has no mean at all** and "
            "still produces a large `dc ratio` — its growth factor is what "
            "separates it from K2b, and it is the reason §7 exists.",
            "",
            "| control | rho_1 | tau(4) | dc \\|t\\| | dc ratio | dc growth "
            "| alt \\|t\\| | alt ratio | alt growth |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for label, _phi, _a, _d, _sp, _sv in SELF_TEST_CASES:
            e = st.get(label)
            if not e:
                continue
            lines.append(
                f"| {label} | {_fmt(e['rho_1_hat'])} | {_fmt(e['tau_4'])} | "
                f"{_fmt(e['dc']['median_abs_t'])} | {_fmt(e['dc']['ratio'])} | "
                f"{_fmt(e['dc']['slot_growth_calibrated'])} | "
                f"{_fmt(e['alt']['median_abs_t'])} | {_fmt(e['alt']['ratio'])} | "
                f"{_fmt(e['alt']['slot_growth_calibrated'])} |"
            )
    if diag["skipped_sidecars"]:
        lines += ["", "Skipped sidecars:", ""]
        for s in diag["skipped_sidecars"]:
            lines.append(f"- `{s['sidecar']}`: {s['reason']}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so a test can assert its defaults.

    Prereg §6b requires a producer's defaults to be asserted against the
    registration rather than trusted; this script is the Phase A producer and
    three of its defaults deliberately differ from §5 (see the module
    docstring's deviations block).  ``tests/test_analyze_channel_audit.py``
    reads the parser and pins each one, so a silent drift in either direction
    fails CI.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sidecars", type=Path, default=Path("results"),
        help="directory of *.instrumentation.json tracked-tier sidecars",
    )
    ap.add_argument(
        "--out-md", type=Path, default=Path("reports/channel-audit-phase-a.md"),
        help="prereg 6a registers the -phase-a name; 6b reserves "
             "reports/channel-audit.md for the Phase B producer",
    )
    ap.add_argument(
        "--out-json", type=Path,
        default=Path("reports/channel-audit-phase-a.json"),
    )
    ap.add_argument(
        "--allow-phase-b-path", action="store_true",
        help="permit writing to the Phase B reserved filenames (refused by "
             "default; this script is the Phase A producer)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="use an evenly spaced subset of N sidecars (smoke test)",
    )
    ap.add_argument(
        "--stride", type=int, default=1,
        help="use every K-th sidecar of the sorted list",
    )
    ap.add_argument(
        "--run-prefix", type=str, default="airbench_instrumented_",
        help="only ingest sidecars whose run name starts with this "
             "(empty string disables); keeps Phase B sidecars written into "
             "the same results/ directory out of the Phase A pool",
    )
    ap.add_argument(
        "--min-seed", type=int, default=1000,
        help="refuse seeds below this (CLAUDE.md ground rule 2: 0-99 are "
             "evaluation seeds and never enter development analysis)",
    )
    ap.add_argument(
        "--segment-at", choices=("refresh", "reset"), default="refresh",
        help="cut segments at the matrix refresh cadence (default) or at each "
             "direction's own reset_steps (DRAFT prereg 5.1)",
    )
    ap.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    ap.add_argument(
        "--burn-ins", type=str, default=",".join(str(b) for b in BURN_INS),
        help="comma-separated burn-in sweep (registered: "
             + ",".join(str(b) for b in REGISTERED_BURN_INS)
             + "; 0 is added for the A2 transient anchor)",
    )
    ap.add_argument("--primary-burn-in", type=int, default=PRIMARY_BURN_IN)
    ap.add_argument(
        "--min-n", type=int, default=10,
        help="drop segments with fewer post-burn-in observations than this",
    )
    ap.add_argument(
        "--null-reps", type=int, default=2000,
        help="AR(1) surrogate draws per (phi, length, burn-in); the "
             "registered value is 200000 (prereg 5.6) and the shortfall is "
             "reported as frac_abs_t_ge_floor and in every ratio CI95",
    )
    ap.add_argument("--null-seed", type=int, default=4242)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument(
        "--bootstrap-block", type=int, default=16,
        help="primary block: one run x matrix x segment cluster of slots",
    )
    ap.add_argument(
        "--bootstrap-block-alt", type=int, default=4,
        help="the registered block (prereg 5.8): consecutive segments of one "
             "direction, reported alongside the primary one",
    )
    ap.add_argument("--bootstrap-seed", type=int, default=4242)
    ap.add_argument(
        "--bootstrap-max-elems", type=int, default=2_000_000,
        help="memory bound on the resample index array; the rep count is "
             "never reduced (the draws are chunked and bit-identical)",
    )
    ap.add_argument(
        "--self-test-reps", type=int, default=4000,
        help="synthetic slots per estimator control (0 disables)",
    )
    ap.add_argument(
        "--no-slot-level", action="store_true",
        help="skip report section 7, the slot-level (prereg 5.1) read; it is "
             "the only section that can tell a persistent mean from "
             "zero-mean low-frequency power, so this is a speed switch for "
             "smoke runs, not a reporting choice",
    )
    ap.add_argument(
        "--verify-blocks", type=int, default=64,
        help="segments re-checked against src.stats.spectral (mirror contract)",
    )
    return ap


def _check_out_path(path: Optional[Path], allow: bool) -> None:
    """Refuse the Phase B reserved filenames unless explicitly allowed.

    ``Path.write_text`` has no overwrite guard, and the confirmatory Phase B
    producer writes ``reports/channel-audit.{md,json}``.  A Phase A run that
    lands there silently replaces (or is replaced by) a different tier's
    numbers at a path no reader can tell apart -- exactly the peeked-vs-
    confirmatory confusion the EXPLORATORY framing exists to prevent.
    """
    if path is None or allow:
        return
    if path.name in PHASE_B_RESERVED:
        raise SystemExit(
            f"refusing to write {path}: reports/channel-audit-preregistration"
            f".md 6b reserves {list(PHASE_B_RESERVED)} for the CONFIRMATORY "
            "Phase B producer (scripts/analyze_channel_audit_frozen.py). "
            "This is the Phase A producer; write "
            "reports/channel-audit-phase-a.{md,json} (the default), or pass "
            "--allow-phase-b-path if you really mean to."
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    burn_ins = tuple(sorted({int(b) for b in args.burn_ins.split(",") if b.strip()}))
    if not burn_ins:
        raise SystemExit("--burn-ins must name at least one value")
    if args.primary_burn_in not in burn_ins:
        raise SystemExit(
            f"--primary-burn-in {args.primary_burn_in} is not in "
            f"--burn-ins {list(burn_ins)}"
        )
    for path in (args.out_md, args.out_json):
        _check_out_path(path, args.allow_phase_b_path)

    paths, selection = select_sidecars(
        args.sidecars, args.limit, args.stride,
        run_prefix=args.run_prefix or None,
        min_seed=None if args.min_seed is None or args.min_seed < 0 else args.min_seed,
    )
    store, slots, meta = ingest(
        paths, burn_ins, int(args.max_lag), int(args.min_n),
        args.segment_at, int(args.verify_blocks), int(args.bootstrap_seed),
    )
    meta["selection"] = selection
    if not store.rows:
        raise SystemExit(
            "no usable tracked-direction segments in these sidecars -- check "
            "--segment-at, --burn-ins and --min-n against the run length"
        )
    parity = meta["n_segment_start_parity_odd"]
    if parity not in (0, meta["n_segment_starts"]):
        raise SystemExit(
            f"segment starts have MIXED parity ({parity} odd of "
            f"{meta['n_segment_starts']}): the alternating channel is tied to "
            "the absolute step, so pooling segments of one slot would flip "
            "its sign between them and cancel a real signal. Refusing to "
            "report section 7 semantics on this input."
        )
    report = build_report(store, slots, meta, args)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(report))

    primary = str(args.primary_burn_in)
    for label, by_b in sorted(report["strata"]["by_kind"].items()):
        c = by_b.get(primary)
        if c is None:
            continue
        dc, alt = c["channels"]["dc"], c["channels"]["alt"]
        slot = ((report["slot_strata"]["by_kind"].get(label) or {}).get(primary)
                or {}).get("channels", {})
        growth = (slot.get("dc") or {}).get("growth_calibrated")
        print(
            f"{label} b={primary}: n_seg={c['n_segments']} "
            f"phi_hat={_fmt(c['phi_hat'])} tau_nw={_fmt(c.get('tau_nw'))} "
            f"dc |t|={_fmt(dc['median_abs_t'])} ratio={_fmt(dc['ratio'])} "
            f"dc slot growth={_fmt(growth)} "
            f"alt |t|={_fmt(alt['median_abs_t'])} ratio={_fmt(alt['ratio'])}"
        )
    led = report["phase_a_reproduction"]
    print(
        f"phase A reproduction obligation (prereg 2): {led['status']}"
        + (f" -- unreproduced {','.join(led['unreproduced'])}"
           if led["unreproduced"] else "")
    )
    mc = report["diagnostics"]["mirror_check"]
    devs = [mc[k] for k in mc if k.startswith("max_abs_dev_")]
    print(
        f"mirror check: {mc.get('n_checked', 0)} segments + "
        f"{mc.get('n_slot_checked', 0)} slots, "
        f"max abs deviation {max(devs) if devs else float('nan'):.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
