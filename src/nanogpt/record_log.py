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
import re
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
    """(step, our loss, record loss, our-record) at every shared step."""
    our_steps, our_losses = ours
    ours_map = dict(zip(our_steps, our_losses))
    rows = []
    for step, rl in zip(record.steps, record.val_losses):
        if step in ours_map:
            rows.append((step, ours_map[step], rl, ours_map[step] - rl))
    return rows
