#!/usr/bin/env python
"""WP0.2 analysis: our nanogpt port vs the pinned record's published trace.

DESCRIPTIVE ONLY. This script reports overlays, deviations, and distributions;
it never emits a pass/fail. The reproduction-quality judgment is the human's
at the WP0.2 checkpoint, against `criteria/nanogpt_tolerance.yaml`.

Usage::

    uv run python scripts/analyze_nanogpt.py --results results/nanogpt_seed*.json
    uv run python scripts/analyze_nanogpt.py --results ... \
        --out-md reports/wp02-nanogpt-repro.md --out-png reports/figures/wp02-overlay.png

What it produces:

1. **Ensemble overlay** of our val-loss-vs-tokens curve(s) against ALL n=20
   record validation logs: per-step record mean and sd, and our deviation in
   both loss units and record-sigma units, at the 125-step eval cadence.
   Comparing against a *single* record log — as this script used to — makes
   the record's own between-run seed noise look like our port's deviation;
   that noise is 5x larger at step 125 than at step 1750, so a single-log
   "max deviation" lands early and measures the record, not us. The single-log
   number is still computed, clearly labelled, and is NOT the headline.
2. **Phase decomposition**: loss removed over the stable phase (step 0 to the
   LR cooldown onset, derived from `cooldown_frac`) vs over the cooldown
   phase, ours against the record ensemble, with the deficit in absolute loss
   and as a percentage of the loss that phase removes; plus the per-eval-
   segment drop ratio within cooldown and a sign test over those segments.
   This is what localises *where* in training a deviation accumulates.
3. **Steps-to-target** (val loss <= 3.28) — reported only with its censoring
   disclosure, because 6 of the record's own 20 runs never reach 3.28.
4. **Our seed variance vs the record's n=20 distribution**, with a Welch
   t-test as a descriptive statistic (not a gate).
5. **Power note**: the n our variance implies for a 1% and a 3%
   steps-to-target effect at 80% power — the number WP3.x needs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nanogpt.config import RECORD_TOKENS_PER_STEP  # noqa: E402
from src.nanogpt.record_log import (  # noqa: E402
    Censoring,
    RecordTrace,
    censoring,
    deviation_at_checkpoints,
    ensemble_deviation,
    phase_decomposition,
    record_validation_traces,
    steps_to_target,
)


def load_runs(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data.get("experiment") != "nanogpt":
            continue
        m = data["metrics"]
        ncfg = m.get("nanogpt_config", {}) or {}
        runs.append(
            {
                "path": Path(path),
                "seed": data["seed"],
                "gpu_type": data["gpu_type"],
                "git_sha": data["git_sha"][:8],
                "cost_usd": data.get("cost_usd"),
                "wall_time_s": data["wall_time_s"],
                "train_time_s": m.get("train_time_s"),
                "steps": [int(p["step"]) for p in m["val_curve"]],
                "tokens": [int(p["tokens"]) for p in m["val_curve"]],
                "losses": [float(p["val_loss"]) for p in m["val_curve"]],
                "final_val_loss": m.get("final_val_loss"),
                "steps_to_target": m.get("steps_to_target_loss"),
                "target": m.get("target_val_loss", 3.28),
                "record_faithful": m.get("record_faithful"),
                "deviations": m.get("deviations", {}),
                "device_count": m.get("device_count"),
                "accumulation_factor": m.get("accumulation_factor"),
                "tokens_per_step": m.get("tokens_per_step"),
                "precision_mode": m.get("precision_mode"),
                "num_iterations": ncfg.get("num_iterations", m.get("num_iterations")),
                "cooldown_frac": ncfg.get("cooldown_frac"),
                "record_world_size": m.get("record_world_size"),
                "fp32_embed_grad_accum": m.get("fp32_embed_grad_accum"),
            }
        )
    return runs


def project_costs(results_dir: Path) -> Dict[str, float]:
    """Sum ``cost_usd`` over every results JSON — the true spend to date.

    Cost is a human-filled provenance field (CLAUDE.md rule 5); this only adds
    up what is actually recorded, so a run whose cost was never filled in is
    silently absent. ``n_missing`` reports how many those are.
    """
    total = 0.0
    nanogpt = 0.0
    n_costed = 0
    n_missing = 0
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict) or "experiment" not in data:
            continue
        cost = data.get("cost_usd")
        if cost is None:
            n_missing += 1
            continue
        n_costed += 1
        total += float(cost)
        if data.get("experiment") == "nanogpt":
            nanogpt += float(cost)
    return {
        "total_usd": total, "nanogpt_usd": nanogpt,
        "n_costed": n_costed, "n_missing": n_missing,
    }


def censoring_note(cens: Censoring) -> List[str]:
    """The mandatory disclosure that must accompany any steps-to-target number."""
    out = [
        f"\n> **CENSORING DISCLOSURE — steps-to-{cens.target:g} is not a sound "
        f"primary endpoint here.**",
    ]
    if not cens.is_censored:
        out.append(
            f"> All {cens.n_total} record runs reach {cens.target:g}; the statistic "
            "is uncensored in this ensemble."
        )
        return out
    finals = ", ".join(f"{v:.4f}" for v in cens.unreached_finals)
    out.append(
        f"> **{cens.n_unreached} of the record's own {cens.n_total} runs never reach "
        f"{cens.target:g}** (their finals: {finals}). The record's "
        f"\"steps-to-{cens.target:g} mean "
        f"{cens.survivor_mean:.1f}, std {cens.survivor_sd:.1f}\" is therefore computed "
        f"over the **n={cens.n_reached} survivors only** — the runs that happened to "
        "clear the bar — and is a **censored** statistic, not the ensemble's "
        "steps-to-target."
    )
    out.append(
        "> It is also **biased between arms**: the target sits inside the record's "
        "own final-loss distribution, so an arm straddling 3.28 drops its slow runs "
        "from the average and is flattered relative to an arm that clears it "
        "outright. The direction of that bias depends on where each arm's "
        "distribution sits, so it does not cancel. **Unsuitable as a primary "
        "endpoint** at this target; final val loss at fixed steps is the "
        "uncensored alternative."
    )
    return out


def welch_t(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    if va + vb == 0:
        return None
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)


def n_for_effect(sd: float, effect: float, power_z: float = 2.802) -> Optional[int]:
    """Per-arm n for a two-sided 0.05 test at 80% power (z_{a/2}+z_b = 2.802)."""
    if effect <= 0 or sd <= 0:
        return None
    return math.ceil(2 * (power_z * sd / effect) ** 2)


def build_report(
    runs: List[Dict[str, Any]],
    record: List[RecordTrace],
    costs: Optional[Dict[str, float]] = None,
) -> str:
    ref: RecordTrace = record[0]
    rec_finals = [t.final_val_loss for t in record]
    target = runs[0]["target"] if runs else 3.28
    cens = censoring(record, target)

    out: List[str] = []
    w = out.append
    w("# WP0.2 — nanogpt port vs pinned record (descriptive)\n")
    w(f"Record: **2025-07-12_BosAlign**, n={len(record)} same-script validation runs "
      f"in `{ref.path.parent.relative_to(REPO_ROOT)}` (script md5 `{ref.script_md5[:8]}`).")
    w(f"Record final val loss: mean {st.mean(rec_finals):.4f}, "
      f"std {st.stdev(rec_finals):.4f}, min {min(rec_finals):.4f}, "
      f"max {max(rec_finals):.4f}, n={len(rec_finals)}.")
    w("\nAll comparisons below are against the **ensemble** of those "
      f"{len(record)} logs, not against any single one. A single record log "
      "carries the record's own between-run seed noise, which is 5x larger at "
      "step 125 than at step 1750; measuring against it attributes that noise "
      "to our port.")

    if not runs:
        w("\n**No local runs supplied** — record-side summary only. "
          "Pass `--results results/nanogpt_*.json` once cloud runs have synced.\n")
        return "\n".join(out)

    # ---- provenance / deviation audit ------------------------------------
    w("\n## Runs\n")
    w("| seed | GPU | git | device_count x accum | tokens/step | precision | final val | steps→3.28 | train s | $ |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s2t = "—" if r["steps_to_target"] is None else f"{r['steps_to_target']:.1f}"
        w(f"| {r['seed']} | {r['gpu_type']} | {r['git_sha']} | "
          f"{r['device_count']} x {r['accumulation_factor']} | {r['tokens_per_step']:,} | "
          f"{r['precision_mode']} | {r['final_val_loss']:.4f} | {s2t} | "
          f"{r['train_time_s']:.0f} | {r['cost_usd'] if r['cost_usd'] is not None else '—'} |")

    # `record_faithful` as recomputed by the CURRENT predicate. Results JSONs
    # are append-only, so a run written before the D != 8 fix carries the old
    # (over-permissive) flag; the corrected reading is here, in the report.
    stale = [
        r for r in runs
        if r["record_faithful"] and r["record_world_size"] is not None
        and r["device_count"] != r["record_world_size"]
    ]
    if stale:
        w("\n**Correction to the stored `record_faithful` flag.** "
          + ", ".join(f"seed {r['seed']}" for r in stale)
          + f": the results JSON records `record_faithful: true`, but the run uses "
            "gradient accumulation (`device_count` != the record's world size) and "
            "its own `deviations` dict lists `grad_accumulation`. That flag was "
            "computed by a predicate that did not inspect `device_count` — a code "
            "defect, since accumulation changes the gradient *reduction order* and "
            "at D<8 the bf16 embedding grads make that a genuine precision "
            "difference (docs/nanogpt-port.md §2). The predicate is fixed in "
            "`src/nanogpt/config.py` (D != 8 is never record-faithful) and pinned "
            "by a regression test. `results/` is append-only, so the JSON keeps the "
            "pre-fix value; **the correct reading is that these runs are NOT "
            "record-faithful**, and this report is where that correction lives.")
        for r in stale:
            r["record_faithful"] = False

    unfaithful = [r for r in runs if not r["record_faithful"]]
    if unfaithful:
        w("\n**Deviation flags active** (these runs are NOT record-faithful):\n")
        for r in unfaithful:
            for k, v in r["deviations"].items():
                w(f"- seed {r['seed']} — `{k}`: {v}")
    tps = {r["tokens_per_step"] for r in runs}
    if tps != {RECORD_TOKENS_PER_STEP}:
        w(f"\n**WARNING: tokens/step {sorted(tps)} != record {RECORD_TOKENS_PER_STEP}** — "
          "the runs are not on the record's token batch; the overlay is not comparable.")

    # ---- ensemble overlay -------------------------------------------------
    w("\n## Overlay vs the record ensemble (headline)\n")
    w(f"Deviation = our val loss − the mean of the n={len(record)} record logs at "
      f"the same step (token checkpoints of {ref.tokens_per_step:,} tokens/step). "
      "`sigma` divides that deviation by the record's **between-run sd at that "
      "step**.\n")
    w("Read `sigma` with care: it is *the record's* sigma, not ours. We have "
      "n=1 and therefore **no estimate of our own harness's seed variance**, so "
      "these are not z-scores for our run. The record's sd also shrinks by ~5x "
      "over training, so a constant absolute deviation grows in sigma by "
      "yardstick shrinkage alone.\n")

    for r in runs:
        rows = ensemble_deviation((r["steps"], r["losses"]), record)
        if not rows:
            w(f"**seed {r['seed']}: no shared checkpoints with the record trace.**")
            continue
        w(f"### seed {r['seed']} — deviation trajectory\n")
        w("| step | ours | record mean | record sd | dev (loss) | dev (sigma) |")
        w("|---|---|---|---|---|---|")
        for row in rows:
            sig = "—" if row.sigma is None else f"{row.sigma:+.1f}"
            w(f"| {row.step} | {row.our_loss:.4f} | {row.record_mean:.4f} | "
              f"{row.record_sd:.5f} | {row.deviation:+.4f} | {sig} |")
        final = rows[-1]
        w(f"\nAt the final step our loss is **{final.deviation:+.4f}** from the "
          f"record ensemble mean and **{final.our_loss - final.record_max:+.4f}** "
          f"from the record's observed MAXIMUM ({final.record_max:.4f}) — "
          + ("**outside the observed support of the record's n=%d distribution**."
             % final.n if final.our_loss > final.record_max else
             "inside the record's observed range."))

        single = deviation_at_checkpoints((r["steps"], r["losses"]), ref)
        if single:
            worst = max(single, key=lambda row: abs(row[3]))
            by_step = {row.step: row for row in rows}
            at_worst = by_step.get(worst[0])
            w(f"\n*(Single-log statistic, non-headline: against `{ref.path.name[:8]}` "
              f"alone the max |dev| is {abs(worst[3]):.4f} at step {worst[0]}.")
            if at_worst is not None and at_worst.sigma is not None:
                w(f"  At that step the record's between-run sd is "
                  f"{at_worst.record_sd:.4f} and our deviation from the ensemble "
                  f"mean is only {at_worst.deviation:+.4f} "
                  f"({at_worst.sigma:+.1f} sigma) — i.e. the single-log number is "
                  "mostly the record run's own seed noise, not our port's. That is "
                  "why it was dropped as the headline.)*")
            else:
                w("  It is dominated by which record run was picked, and is "
                  "reported only to show why it was dropped as a headline.)*")

    # ---- phase decomposition ---------------------------------------------
    w("\n## Where the deviation accumulates (phase decomposition)\n")
    w("The record's LR cooldown begins at `num_iterations * (1 - cooldown_frac)` "
      "(RECORD:670-684). Splitting training there separates a deviation that "
      "accrues during the stable-LR phase from one that accrues while the LR "
      "anneals. The deficit is stated in absolute loss AND as a fraction of the "
      "loss that phase actually removes — the second is the meaningful one, "
      "since the two phases remove very different amounts.\n")
    for r in runs:
        if r["cooldown_frac"] is None or r["num_iterations"] is None:
            w(f"seed {r['seed']}: schedule not recorded in the results JSON; skipped.")
            continue
        pd = phase_decomposition(
            (r["steps"], r["losses"]), record, r["num_iterations"], r["cooldown_frac"]
        )
        if pd is None:
            w(f"seed {r['seed']}: too few shared checkpoints; skipped.")
            continue
        w(f"### seed {r['seed']}\n")
        w(f"Cooldown onset: step {pd.cooldown_start_exact:.1f} exactly "
          f"(= {r['num_iterations']} x (1 − {r['cooldown_frac']})), snapped to the "
          f"nearest validation step **{pd.cooldown_start_step}**.\n")
        w("| phase | steps | our drop | record drop | deficit | deficit % of phase drop |")
        w("|---|---|---|---|---|---|")
        for ph in (pd.stable, pd.cooldown):
            frac = "—" if ph.deficit_frac is None else f"{100*ph.deficit_frac:.2f}%"
            w(f"| {ph.name} | {ph.start_step}→{ph.end_step} | {ph.our_drop:.4f} | "
              f"{ph.record_drop:.4f} | {ph.deficit:+.4f} | {frac} |")
        sr, cr = pd.stable.deficit_frac, pd.cooldown.deficit_frac
        if sr and cr:
            w(f"\nPer unit of loss removed, the deficit is **{cr/sr:.1f}x denser in "
              "the cooldown phase** than in the stable phase.")
        w("\n**Per-eval-segment drops within cooldown** (ratio < 1 = we remove less "
          "loss than the record over that interval):\n")
        w("| segment | our drop | record drop | ratio |")
        w("|---|---|---|---|")
        for s in pd.cooldown_segments:
            ratio = "—" if s.ratio is None else f"{s.ratio:.4f}"
            w(f"| {s.start_step}→{s.end_step} | {s.our_drop:.4f} | "
              f"{s.record_drop:.4f} | {ratio} |")
        n_seg = len(pd.cooldown_segments)
        p = pd.sign_test_p
        w(f"\nSign test: we remove less loss than the record in "
          f"**{pd.n_segments_below_record} of {n_seg}** cooldown segments"
          + (f", two-sided p = {p:.3f}." if p is not None else ".")
          + " Consecutive segments share one curve, so treat this as a "
            "consistency measure, not an inferential test.")

    # ---- distributions ---------------------------------------------------
    finals = [r["final_val_loss"] for r in runs if r["final_val_loss"] is not None]
    ours_s2t = [r["steps_to_target"] for r in runs if r["steps_to_target"] is not None]
    w("\n## Distributions\n")
    w(f"Ours: final val loss mean {st.mean(finals):.4f}"
      + (f", std {st.stdev(finals):.4f}" if len(finals) > 1 else " (n=1, no std)")
      + f", n={len(finals)}.")
    w(f"Record: mean {st.mean(rec_finals):.4f}, std {st.stdev(rec_finals):.4f}, n={len(rec_finals)}.")
    t = welch_t(finals, rec_finals)
    if t is not None:
        w(f"Welch t (ours vs record, descriptive only): t = {t:+.2f}.")
    w(f"\n### Steps-to-{target:g}\n")
    if cens.survivor_mean is not None:
        w(f"Record steps-to-{target:g} (interpolated from its own 125-step trace): "
          f"mean {cens.survivor_mean:.1f}, std {cens.survivor_sd:.1f} of "
          f"{ref.total_steps} total — **over n={cens.n_reached}, not "
          f"n={cens.n_total}**.")
    for line in censoring_note(cens):
        w(line)
    if ours_s2t:
        w(f"\nOurs: steps-to-{target:g} mean {st.mean(ours_s2t):.1f}"
          + (f", std {st.stdev(ours_s2t):.1f}" if len(ours_s2t) > 1 else " (n=1)") + ".")
        if len(ours_s2t) > 1 and st.mean(ours_s2t) > 0:
            sd = st.stdev(ours_s2t)
            mean = st.mean(ours_s2t)
            w("\n### Power (per-arm n, two-sided 0.05, 80% power)\n")
            w("| effect on steps-to-target | absolute steps | required n/arm |")
            w("|---|---|---|")
            for pct in (0.01, 0.03, 0.05):
                eff = pct * mean
                w(f"| {pct*100:.0f}% | {eff:.1f} | {n_for_effect(sd, eff)} |")
            w("\nBasis: our own seed std above (small n — treat as an order-of-magnitude "
              "input to WP3.x, not a precise number).")
    else:
        w(f"\n**None of our runs reached {target:g}** — our steps-to-target is "
          "undefined and is NOT extrapolated. Note that our run is censored by "
          "the same mechanism as the 6 record runs above, which is precisely why "
          "the metric cannot carry this comparison.")

    # ---- reading of the numbers (descriptive) -----------------------------
    single_run = [r for r in runs if r["final_val_loss"] is not None]
    if len(single_run) == 1:
        r = single_run[0]
        rows = ensemble_deviation((r["steps"], r["losses"]), record)
        if rows:
            f = rows[-1]
            w("\n## Reading of these numbers (descriptive)\n")
            w(f"1. **Our one run sits {f.deviation:+.4f} above the record's n="
              f"{f.n} mean ({f.record_mean:.4f}), and {f.our_loss - f.record_max:+.4f} "
              f"above the record's observed maximum ({f.record_max:.4f})** — that is, "
              "outside the observed support of the record's distribution, not merely "
              "in its upper tail.")
            w(f"2. **The \"{f.sigma:.1f} sigma\" is {f.sigma:.1f}x the RECORD's sigma, "
              "and is partly a yardstick artefact.** We have n=1 and therefore no "
              "estimate of our own harness's seed variance; the record's sd shrinks "
              f"from {rows[1].record_sd:.4f} (step {rows[1].step}) to "
              f"{f.record_sd:.5f} (step {f.step}) while our absolute deviation is "
              "roughly flat after step 875, so most of the growth in the sigma "
              "column is the denominator shrinking, not our run drifting.")
            w("3. **The deviation is cooldown-concentrated** — see the phase table: "
              "per unit of loss removed, the deficit is an order of magnitude denser "
              "during the LR cooldown, and every cooldown segment underperforms.")
            w("4. **Leading suspect: bf16 embedding-gradient accumulation at D<8.** "
              "At `device_count: 1` the port sums 8 chunk gradients sequentially into "
              "bf16 `p.grad` (embeddings are bf16, RECORD:628-630) where the record "
              "does an 8-way `ReduceOp.AVG` across ranks. docs/nanogpt-port.md §2 "
              "already names this \"the least-controlled numeric deviation in the "
              "port\". It is not the only candidate — torch-version/kernel drift vs "
              "the record's 2025 nightly is unmeasured — but it is the one we can "
              "test with a single one-variable run.")
            w("\nNo pass/fail is drawn from any of this.")

            w("\n## PRE-REGISTERED next diagnostic (written BEFORE the run)\n")
            w("**This section was written before the probe run was launched and "
              "must not be revised after seeing its result.**\n")
            w("Probe: `configs/wp02_nanogpt_fp32embed.yaml` — the port doc's §6.1 "
              "diagnostic. fp32 master-buffer accumulation of embedding gradients "
              "across the 8 micro-batches (cast back to bf16 once per step) at "
              "`device_count: 1`, **seed 1701, everything else identical** to "
              "`configs/wp02_nanogpt_repro.yaml`. One variable changes.\n")
            w("Read, fixed in advance:\n")
            w("| final deficit vs record mean | conclusion |")
            w("|---|---|")
            w("| **<= +0.006** | bf16-accumulation suspect **confirmed** |")
            w(f"| **unchanged at ~{f.deviation:+.3f}** | suspect **excluded**; residual "
              "is torch-version / kernel / hardware |")
            w("\nAn intermediate outcome (between +0.006 and +0.011) is partial "
              "attribution and is to be reported as partial, not rounded to either "
              "verdict. The probe run is NOT record-faithful (two deviation flags: "
              "`grad_accumulation` and `fp32_embed_grad_accum`) and never enters a "
              "reproduction table.")

    if costs:
        w("\n## Cost reconciliation\n")
        w(f"Summed from `cost_usd` across `results/`: "
          f"**${costs['total_usd']:.2f} project total**, of which "
          f"**${costs['nanogpt_usd']:.2f} is WP0.2 nanogpt** "
          f"({costs['n_costed']} costed run(s); {costs['n_missing']} run(s) carry "
          "no `cost_usd` and are excluded).")

    w("\n---\nDescriptive only; no pass/fail. Reproduction quality is judged by the "
      "human against `criteria/nanogpt_tolerance.yaml`.")
    return "\n".join(out)


def plot(runs, record: List[RecordTrace], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for t in record:
        ax.plot([x / 1e6 for x in t.tokens], t.val_losses, color="0.7", lw=0.8,
                label="record (n=%d)" % len(record) if t is record[0] else None)
    for r in runs:
        ax.plot([x / 1e6 for x in r["tokens"]], r["losses"], lw=1.5, label=f"ours seed {r['seed']}")
    ax.axhline(3.28, color="crimson", ls="--", lw=0.8, label="target 3.28")
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel("val loss")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("loss vs tokens")

    ax = axes[1]
    for r in runs:
        rows = ensemble_deviation((r["steps"], r["losses"]), record)
        rows = [row for row in rows if row.record_sd > 0]
        if not rows:
            continue
        steps = [row.step for row in rows]
        ax.plot(steps, [row.deviation for row in rows], marker="o", ms=3,
                label=f"seed {r['seed']} − ensemble mean")
        # The record's own +/-1 and +/-2 sd envelope, so the reader can see the
        # yardstick shrinking underneath a roughly flat absolute deviation.
        for k, alpha in ((1, 0.30), (2, 0.15)):
            ax.fill_between(steps, [-k * row.record_sd for row in rows],
                            [k * row.record_sd for row in rows],
                            color="0.4", alpha=alpha, lw=0,
                            label=f"record ±{k}sd (n={rows[0].n})")
        if r["cooldown_frac"] is not None and r["num_iterations"] is not None:
            pd = phase_decomposition((r["steps"], r["losses"]), record,
                                     r["num_iterations"], r["cooldown_frac"])
            if pd is not None:
                ax.axvline(pd.cooldown_start_exact, color="darkorange", ls=":",
                           lw=1.0, label="cooldown onset")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("ours − record ensemble mean (val loss)")
    ax.set_title("deviation vs the record ensemble")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, nargs="*", default=[], help="nanogpt results JSONs")
    ap.add_argument("--record-dir", type=Path, default=None, help="record log directory")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                    help="directory summed for the cost reconciliation")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    args = ap.parse_args(argv)

    record = record_validation_traces(args.record_dir) if args.record_dir else record_validation_traces()
    runs = load_runs(args.results)
    costs = project_costs(args.results_dir) if args.results_dir.is_dir() else None
    report = build_report(runs, record, costs)
    print(report)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(report + "\n")
        print(f"\nwrote {args.out_md}")
    if args.out_png:
        plot(runs, record, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
