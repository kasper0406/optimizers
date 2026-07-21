"""FineWeb data pipeline: the record's loader, with 8-rank batching emulated.

Record source (see src/nanogpt/__init__.py): the 2025-07-12_BosAlign
validation script, ``0c5449cc-....txt`` lines 509-565.

WHY THIS FILE IS NOT VERBATIM
-----------------------------
The record's BOS-aligned training loader (RECORD:522 ``find_batch_starts``)
picks **``world_size`` start points jointly** per optimizer step: consecutive
BOS-token positions at least ``local_batch_size`` apart, and then advances
``pos`` by the span those 8 chunks covered.  The chunk boundaries are
therefore a function of the *number of ranks*, not of any per-rank state — so
running the record loader at ``world_size=1`` would produce a **different
token stream**, not merely a differently-distributed one.

This port keeps the record's chunking exactly: every optimizer step it
computes the same ``record_world_size`` (8) BOS-aligned chunks the record
would have computed from the same file position, and hands each device its
``accum_factor`` of them.  The chunk→(device, micro-step) assignment is
``chunk = micro * device_count + rank``; since the optimizer step consumes the
sum over all 8 chunks (see the accumulation math in ``config.py``), the
assignment does not affect the resulting gradient.

Validation (RECORD:737, ``align_to_bos=False``) is a plain sequential scan, so
the port simply distributes the record's 40 fixed 262,144-token chunks over
the available devices — the same tokens, in the same order, evaluated in the
same fixed-size pieces.

``_load_data_shard`` is verbatim (RECORD:509-519) except that ``pin_memory``
is only requested when CUDA is available (it raises on a CPU-only box).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

BOS_TOKEN = 50256  # RECORD:523 / RECORD:431 — GPT-2 <|endoftext|>


def _load_data_shard(file: Path) -> Tensor:
    """RECORD:509-519, verbatim except the pin_memory guard (PORT)."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        # PORT: pin_memory=torch.cuda.is_available() — pinned allocation raises
        # on a CPU-only box; identical behaviour wherever the record ran.
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=torch.cuda.is_available()) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def find_batch_starts(
    tokens: Tensor, pos: int, world_size: int, local_batch_size: int, max_batch_span: int
) -> Tuple[List[int], int]:
    """RECORD:522-539, verbatim.

    ``world_size`` here is always the RECORD's world size (8), never the port's
    device count — that is the entire point of the emulation.
    """
    boundary_mask = tokens[pos:pos+max_batch_span] == BOS_TOKEN
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    batch_end = None
    for i in range(len(boundary_positions) - 1):
        end = boundary_positions[i + 1].item()
        if end - start >= local_batch_size:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == world_size:
                batch_end = end
                break
            start = end
    assert batch_end is not None # increase max_batch_span if necessary
    batch_span = batch_end - pos
    return starts, batch_span


class RecordDataGenerator:
    """The record's data stream, sliced for ``device_count`` devices.

    One ``next_step()`` yields this rank's ``accum`` micro-batches for one
    optimizer step; concatenated across ranks and micro-steps they are exactly
    the ``record_world_size`` chunks the record consumed on that step.
    """

    def __init__(
        self,
        filename_pattern: str,
        *,
        local_batch_size: int,
        record_world_size: int,
        device_count: int,
        rank: int,
        align_to_bos: bool,
        device: Optional[torch.device] = None,
    ):
        self.files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(
                f"no data shards match {filename_pattern!r}; run "
                "`uv run python scripts/fetch_fineweb.py --config <config.yaml>` first"
            )
        if record_world_size % device_count != 0:
            raise ValueError("device_count must divide record_world_size")
        self.local_batch_size = local_batch_size
        self.record_world_size = record_world_size
        self.device_count = device_count
        self.accum = record_world_size // device_count
        self.rank = rank
        self.align_to_bos = align_to_bos
        self.device = device or torch.device("cpu")
        self.batch_size = record_world_size * local_batch_size
        # RECORD:547: buffer to handle samples up to length local_batch_size.
        # The BOS-boundary search window is floored at the RECORD's window
        # (2 x 8 x local_batch_size): at the program-#7 small token batches
        # (record_world_size here < 8) a proportional window can fail to
        # contain enough qualifying document boundaries (observed: assertion
        # in find_batch_starts at chunks_per_step=2). A wider window never
        # changes which chunks are selected when the search succeeds — the
        # greedy scan is order-deterministic — it only prevents the overflow
        # (and moves the end-of-shard advance guard correspondingly earlier
        # for those non-record geometries). Record geometry (8 chunks) is
        # bit-identical: its own window already equals the floor.
        span_floor = 2 * 8 * local_batch_size
        self.max_batch_span = (
            max(2 * self.batch_size, span_floor) if align_to_bos else self.batch_size
        )
        self.file_index = 0
        self.tokens = _load_data_shard(self.files[0])
        self.pos = 0

    # -------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, int]:
        return {"file_index": self.file_index, "pos": self.pos}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        if state["file_index"] != self.file_index:
            self.file_index = int(state["file_index"])
            self.tokens = _load_data_shard(self.files[self.file_index])
        self.pos = int(state["pos"])

    # -------------------------------------------------------------- stream
    def _advance_shard(self) -> None:
        self.file_index += 1
        if self.file_index >= len(self.files):
            raise RuntimeError(
                f"ran out of data shards after {len(self.files)} files "
                f"({self.files[-1].name}); fetch more with scripts/fetch_fineweb.py"
            )
        self.tokens = _load_data_shard(self.files[self.file_index])
        self.pos = 0

    def _chunk_starts(self) -> List[int]:
        """The record_world_size chunk starts for one optimizer step (RECORD:541-565)."""
        if self.pos + self.max_batch_span + 1 >= len(self.tokens):
            self._advance_shard()
        if self.align_to_bos:
            starts, batch_span = find_batch_starts(
                self.tokens, self.pos, self.record_world_size, self.local_batch_size, self.max_batch_span
            )
        else:
            batch_span = self.batch_size
            starts = [self.pos + i * self.local_batch_size for i in range(self.record_world_size)]
        self.pos += batch_span
        return starts

    def _materialize(self, start_idx: int) -> Tuple[Tensor, Tensor]:
        buf = self.tokens[start_idx:][: self.local_batch_size + 1]
        non_blocking = self.device.type == "cuda"
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=non_blocking)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=non_blocking)
        return inputs, targets

    def next_step(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Yield this rank's ``accum`` micro-batches for one optimizer step."""
        starts = self._chunk_starts()
        for micro in range(self.accum):
            # chunk = micro * device_count + rank (see module docstring).
            yield self._materialize(starts[micro * self.device_count + self.rank])


def data_footprint_gb(num_shards: int) -> float:
    """FineWeb10B GPT-2 shards are 100M uint16 tokens + a 1 KiB header each."""
    return num_shards * (100_000_000 * 2 + 256 * 4) / 1e9
