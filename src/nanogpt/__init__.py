"""WP0.2 — port of the pinned modded-nanogpt record to a 1-2 GPU testbed.

Pinned record (human decision, `reports/wp02-record-candidates.md` DECISION):
**2025-07-12_BosAlign**.

RECORD SOURCE FILE (the file every ``RECORD:`` line reference in this package
points at)::

    vendor/modded-nanogpt/records/track_1_short/2025-07-12_BosAlign/
        0c5449cc-0b01-4ecc-bec3-f46a09741d60.txt

Each record ``.txt`` is self-contained: the full training script (lines
1-783) followed by that run's log.

**Which of the 21 logs in the record directory is authoritative — and why it
is NOT the one the candidates report cites.**  The directory holds 21 logs.
Twenty of them embed a byte-identical script (md5 of lines 1..783 =
5ffba04fdf6f977dab0f868fa48ae459) and their 20 final val losses are an exact
multiset match for the ``accs`` list in the record's own README (mean 3.2791,
std 0.0013) — i.e. **those 20 runs are the n=20 distribution that is our power
baseline**.  The 21st log, ``c1fd8a38-bb9f-45c4-8af0-d37f70c993f3.txt`` (the
log cited as "primary" in `reports/wp02-record-candidates.md`), embeds a
*different, refactored* script: it is the 07/13/25 retiming run that the
record README describes as done "with a refactored version of the code", and
it is **not** one of the 20 validation runs.  The refactored script differs in
ML, not just in plumbing:

- LR schedule floor: retiming script ``w*1.0 + (1-w)*0.1`` vs validation
  script ``w*1.0 + (1-w)*0.05``.  The record README explicitly lists
  "decreased minimum lr schedule factor from 0.1 to 0.05" as one of the three
  changes that *make* this record, so **0.05 is the record's ML** and the
  retiming script has 0.1.
- the retiming script clamps ``assert 0 <= x < 1`` and branches on the
  cooldown; the validation script uses ``w = min((1-x)/cooldown_frac, 1.0)``.
- ``assert world_size == 8`` is present in the retiming script, absent in the
  validation script.

This port therefore reproduces the **validation script** (the n=20 one), so
that our loss-vs-tokens overlay and our steps-to-3.28 numbers are comparable
to the n=20 distribution we power against.  See `docs/nanogpt-port.md`.

Modules:

- ``config``   YAML -> :class:`NanoGPTConfig`; the accumulation arithmetic.
- ``model``    architecture, verbatim from the record except where marked.
- ``optim``    ``Muon`` and ``DistAdam``, verbatim from the record.
- ``data``     FineWeb loader; emulates the record's 8-rank BOS-aligned batching.
- ``train``    training loop with gradient accumulation + the results metrics.
- ``record_log``  parser for the loss-vs-tokens trace inside a record log.
"""

from src.nanogpt.config import (  # noqa: F401
    RECORD_LOG,
    RECORD_TOKENS_PER_STEP,
    NanoGPTConfig,
    accumulation_factor,
    tokens_per_step,
)

__all__ = [
    "NanoGPTConfig",
    "RECORD_LOG",
    "RECORD_TOKENS_PER_STEP",
    "accumulation_factor",
    "tokens_per_step",
]
