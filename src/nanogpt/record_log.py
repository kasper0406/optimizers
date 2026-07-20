"""Parse the published trace out of a modded-nanogpt record log file.

Each record ``.txt`` is the training script followed by the run's stdout. The
validation lines have the shape (RECORD:748)::

    step:125/1750 val_loss:4.2887 train_time:12345ms step_avg:98.76ms

and the per-step training lines (RECORD:779)::

    step:126/1750 train_time:12432ms step_avg:98.67ms

Only the validation lines carry a loss, so the record's published loss-vs-
tokens trace is sampled every ``val_loss_every`` (=125) steps, plus the final
step. Tokens are ``step * tokens_per_step`` with the record's 393,216.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.nanogpt.config import (
    RECORD_DIR,
    RECORD_LOG,
    RECORD_SCRIPT_LINES,
    RECORD_TOKENS_PER_STEP,
)

VAL_LINE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+) "
    r"train_time:(?P<time>\d+)ms"
)
SEP = "=" * 100


@dataclass
class RecordTrace:
    """A record run's published loss-vs-tokens trace."""

    path: Path
    steps: List[int]
    val_losses: List[float]
    train_time_ms: List[float]
    total_steps: int
    tokens_per_step: int
    script_md5: str

    @property
    def tokens(self) -> List[int]:
        return [s * self.tokens_per_step for s in self.steps]

    @property
    def final_val_loss(self) -> float:
        return self.val_losses[-1]

    @property
    def final_train_time_s(self) -> float:
        return self.train_time_ms[-1] / 1000.0

    def loss_at_step(self, step: int) -> Optional[float]:
        try:
            return self.val_losses[self.steps.index(step)]
        except ValueError:
            return None

    def steps_to_target(self, target: float) -> Optional[float]:
        return steps_to_target(self.steps, self.val_losses, target)


def script_md5(path: Path, script_lines: int = RECORD_SCRIPT_LINES) -> str:
    """MD5 of the embedded training script (the lines before the log body).

    Used to tell the 20 validation runs (identical script) apart from the
    07/13 retiming run, whose script differs in ML — see
    ``src/nanogpt/__init__.py``.
    """
    lines = Path(path).read_text(errors="replace").splitlines(keepends=True)
    end = script_lines
    for i, line in enumerate(lines):
        if line.strip() == SEP:
            end = i
            break
    return hashlib.md5("".join(lines[:end]).encode()).hexdigest()


def parse_record_log(path: Path = RECORD_LOG, tokens_per_step: int = RECORD_TOKENS_PER_STEP) -> RecordTrace:
    """Extract the validation trace from a record log file."""
    path = Path(path)
    steps: List[int] = []
    losses: List[float] = []
    times: List[float] = []
    total = 0
    for line in path.read_text(errors="replace").splitlines():
        m = VAL_LINE.match(line.strip())
        if m is None:
            continue
        steps.append(int(m["step"]))
        losses.append(float(m["loss"]))
        times.append(float(m["time"]))
        total = int(m["total"])
    if not steps:
        raise ValueError(f"no `step:N/M val_loss:` lines found in {path}")
    return RecordTrace(
        path=path,
        steps=steps,
        val_losses=losses,
        train_time_ms=times,
        total_steps=total,
        tokens_per_step=tokens_per_step,
        script_md5=script_md5(path),
    )


def collect_record_runs(directory: Path = RECORD_DIR) -> Dict[str, List[RecordTrace]]:
    """Group every log in a record directory by embedded-script MD5.

    The largest group is the record's validation set (n=20 for BosAlign); any
    other group is a differently-scripted run (e.g. the 07/13 retiming).
    """
    groups: Dict[str, List[RecordTrace]] = {}
    for path in sorted(Path(directory).glob("*.txt")):
        trace = parse_record_log(path)
        groups.setdefault(trace.script_md5, []).append(trace)
    return groups


def record_validation_traces(directory: Path = RECORD_DIR) -> List[RecordTrace]:
    """The n=20 validation runs: the largest same-script group in the dir."""
    groups = collect_record_runs(directory)
    return max(groups.values(), key=len)


def steps_to_target(
    steps: Sequence[int], losses: Sequence[float], target: float
) -> Optional[float]:
    """First step at which val loss reaches ``target``, linearly interpolated.

    Returns ``None`` if the curve never reaches the target (the honest answer
    — never extrapolate a steps-to-target from a run that did not get there).
    The record's own curve is sampled every 125 steps, so the interpolation
    error is bounded by that spacing; it is the same for every arm, so it does
    not bias a comparison.
    """
    prev_s: Optional[int] = None
    prev_l: Optional[float] = None
    for s, loss in zip(steps, losses):
        if loss <= target:
            if prev_s is None or prev_l is None or prev_l == loss:
                return float(s)
            frac = (prev_l - target) / (prev_l - loss)
            return float(prev_s) + frac * (s - prev_s)
        prev_s, prev_l = s, loss
    return None


def deviation_at_checkpoints(
    ours: Tuple[Sequence[int], Sequence[float]],
    record: RecordTrace,
) -> List[Tuple[int, float, float, float]]:
    """(step, our loss, record loss, our-record) at every shared step.

    SINGLE-LOG statistic. Deprecated as a headline: a deviation against one
    record run confounds our port's offset with the record's own between-run
    seed noise, which is large early (sd 0.0075 at step 125) and small late
    (sd 0.0013 at step 1750). Use :func:`ensemble_deviation` instead; this
    remains only for the plot's per-log reference curves.
    """
    our_steps, our_losses = ours
    ours_map = dict(zip(our_steps, our_losses))
    rows = []
    for step, rl in zip(record.steps, record.val_losses):
        if step in ours_map:
            rows.append((step, ours_map[step], rl, ours_map[step] - rl))
    return rows


# ---------------------------------------------------------------------------
# Ensemble statistics: our run vs the record's n=20 distribution
# ---------------------------------------------------------------------------


@dataclass
class EnsembleDeviation:
    """Our loss vs the record ensemble at one shared validation step."""

    step: int
    our_loss: float
    record_mean: float
    record_sd: float
    record_min: float
    record_max: float
    n: int

    @property
    def deviation(self) -> float:
        """Ours - record ensemble mean, in loss units."""
        return self.our_loss - self.record_mean

    @property
    def sigma(self) -> Optional[float]:
        """Deviation in units of the RECORD's between-run sd at this step.

        Not a z-score for our own run: we have n=1 and therefore no estimate
        of our harness's own seed variance. The record's sd shrinks over
        training, so a flat absolute deviation grows in sigma by yardstick
        shrinkage alone.
        """
        if self.record_sd <= 0:
            return None
        return self.deviation / self.record_sd


def ensemble_at_step(traces: Sequence[RecordTrace], step: int) -> Optional[Tuple[float, float, float, float, int]]:
    """(mean, sd, min, max, n) of the record ensemble's loss at ``step``."""
    vals = [v for v in (t.loss_at_step(step) for t in traces) if v is not None]
    if len(vals) < 2:
        return None
    return (st.mean(vals), st.stdev(vals), min(vals), max(vals), len(vals))


def ensemble_deviation(
    ours: Tuple[Sequence[int], Sequence[float]],
    traces: Sequence[RecordTrace],
) -> List[EnsembleDeviation]:
    """Our trace against the record ENSEMBLE at every shared validation step.

    This is the headline overlay statistic for WP0.2: it separates our port's
    systematic offset from the record's between-run noise, which a comparison
    against any single record log cannot do.
    """
    our_steps, our_losses = ours
    ours_map = dict(zip(our_steps, our_losses))
    rows: List[EnsembleDeviation] = []
    for step in sorted(ours_map):
        stats = ensemble_at_step(traces, step)
        if stats is None:
            continue
        mean, sd, lo, hi, n = stats
        rows.append(
            EnsembleDeviation(
                step=step, our_loss=ours_map[step], record_mean=mean,
                record_sd=sd, record_min=lo, record_max=hi, n=n,
            )
        )
    return rows


@dataclass
class Censoring:
    """How many record runs never reach a target — see :func:`censoring`."""

    target: float
    n_total: int
    n_reached: int
    unreached_finals: List[float]
    survivor_mean: Optional[float]
    survivor_sd: Optional[float]

    @property
    def n_unreached(self) -> int:
        return self.n_total - self.n_reached

    @property
    def is_censored(self) -> bool:
        return self.n_unreached > 0


def censoring(traces: Sequence[RecordTrace], target: float) -> Censoring:
    """Censoring of a steps-to-``target`` statistic over the record ensemble.

    Runs that never reach the target contribute no steps-to-target value, so
    the surviving mean is computed over the *fastest* subset only. That makes
    steps-to-target a censored statistic here, and a between-arm-biased one:
    an arm whose loss distribution straddles the target loses its slow runs
    from the average, flattering it relative to an arm that clears the target
    outright. It is therefore unsuitable as a primary endpoint at this target.
    """
    values = [t.steps_to_target(target) for t in traces]
    reached = [v for v in values if v is not None]
    unreached = sorted(t.final_val_loss for t, v in zip(traces, values) if v is None)
    return Censoring(
        target=target,
        n_total=len(values),
        n_reached=len(reached),
        unreached_finals=unreached,
        survivor_mean=st.mean(reached) if reached else None,
        survivor_sd=st.stdev(reached) if len(reached) > 1 else None,
    )


@dataclass
class PhaseDrop:
    """Loss removed over one training phase, ours vs the record ensemble."""

    name: str
    start_step: int
    end_step: int
    our_drop: float
    record_drop: float

    @property
    def deficit(self) -> float:
        """Loss we failed to remove relative to the record, in loss units."""
        return self.record_drop - self.our_drop

    @property
    def deficit_frac(self) -> Optional[float]:
        """Deficit as a fraction of the loss the record removes in the phase."""
        if self.record_drop == 0:
            return None
        return self.deficit / self.record_drop


@dataclass
class Segment:
    """One eval-to-eval interval's loss drop, ours vs the record ensemble."""

    start_step: int
    end_step: int
    our_drop: float
    record_drop: float

    @property
    def ratio(self) -> Optional[float]:
        if self.record_drop == 0:
            return None
        return self.our_drop / self.record_drop


@dataclass
class PhaseDecomposition:
    """Where in training our deviation from the record accumulates."""

    cooldown_start_exact: float
    cooldown_start_step: int
    stable: PhaseDrop
    cooldown: PhaseDrop
    cooldown_segments: List[Segment]

    @property
    def n_segments_below_record(self) -> int:
        return sum(1 for s in self.cooldown_segments if s.our_drop < s.record_drop)

    @property
    def sign_test_p(self) -> Optional[float]:
        """Two-sided sign-test p over the cooldown segments (H0: p=1/2).

        Segments share a curve, so they are not independent — read this as a
        descriptive consistency measure, not an inferential p-value.
        """
        n = len(self.cooldown_segments)
        if n == 0:
            return None
        k = self.n_segments_below_record
        k = max(k, n - k)
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
        return min(1.0, 2 * tail)


def cooldown_start(num_iterations: int, cooldown_frac: float) -> float:
    """Exact step at which the record's LR cooldown begins (RECORD:670-684).

    The record's schedule is ``w = min((1 - x) / cooldown_frac, 1.0)`` with
    ``x = step / num_iterations``, so ``w < 1`` exactly when
    ``step > num_iterations * (1 - cooldown_frac)``.
    """
    return num_iterations * (1.0 - cooldown_frac)


def phase_decomposition(
    ours: Tuple[Sequence[int], Sequence[float]],
    traces: Sequence[RecordTrace],
    num_iterations: int,
    cooldown_frac: float,
) -> Optional[PhaseDecomposition]:
    """Split the loss drop into stable and cooldown phases, ours vs record.

    The phase boundary is snapped to the nearest shared validation step, since
    the loss is only observed on the eval grid; the exact boundary is reported
    alongside it.
    """
    rows = ensemble_deviation(ours, traces)
    if len(rows) < 3:
        return None
    by_step = {r.step: r for r in rows}
    steps = sorted(by_step)
    exact = cooldown_start(num_iterations, cooldown_frac)
    boundary = min(steps[1:-1], key=lambda s: abs(s - exact))

    def drop(name: str, a: int, b: int) -> PhaseDrop:
        return PhaseDrop(
            name=name, start_step=a, end_step=b,
            our_drop=by_step[a].our_loss - by_step[b].our_loss,
            record_drop=by_step[a].record_mean - by_step[b].record_mean,
        )

    cooldown_steps = [s for s in steps if s >= boundary]
    segments = [
        Segment(
            start_step=a, end_step=b,
            our_drop=by_step[a].our_loss - by_step[b].our_loss,
            record_drop=by_step[a].record_mean - by_step[b].record_mean,
        )
        for a, b in zip(cooldown_steps, cooldown_steps[1:])
    ]
    return PhaseDecomposition(
        cooldown_start_exact=exact,
        cooldown_start_step=boundary,
        stable=drop("stable", steps[0], boundary),
        cooldown=drop("cooldown", boundary, steps[-1]),
        cooldown_segments=segments,
    )
