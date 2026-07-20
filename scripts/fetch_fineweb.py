#!/usr/bin/env python
"""Fetch exactly the FineWeb10B shards a nanogpt config needs (WP0.2).

The record trains on the GPT-2-tokenized FineWeb10B shards published as the
HuggingFace dataset ``kjj0/fineweb10B-gpt2`` — the same source the vendored
downloader uses (``vendor/modded-nanogpt/data/cached_fineweb10B.py``). That
downloader always fetches whole 100M-token chunks and needs
``huggingface_hub`` (not a project dependency); this script fetches only the
shards the configured token budget requires, over plain HTTPS, and is
resumable and verified.

Usage::

    uv run python scripts/fetch_fineweb.py --config configs/wp02_nanogpt_repro.yaml
    uv run python scripts/fetch_fineweb.py --config <cfg> --dry-run   # print plan
    uv run python scripts/fetch_fineweb.py --shards 9                 # explicit

Verification (every shard, on every run — cheap, no re-download):

1. the 256-int32 header: magic ``20240520``, version ``1``, ``num_tokens``;
2. file size == ``1024 + 2 * num_tokens`` bytes (uint16 tokens);
3. sha256 recorded into ``<data-dir>/shard_manifest.json`` on first successful
   download and re-checked on later runs, so a silently corrupted shard is
   caught before it corrupts a benchmark run.

Resumability: partial files are downloaded to ``<name>.part`` and continued
with an HTTP Range request; only a fully verified file is renamed into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.nanogpt.config import NanoGPTConfig  # noqa: E402
from src.nanogpt.data import data_footprint_gb  # noqa: E402

BASE_URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"
TOKENS_PER_SHARD = 100_000_000  # each published chunk is 100M GPT-2 tokens
HEADER_BYTES = 256 * 4
MAGIC = 20240520

# BOS-aligned batching consumes slightly more than batch_size tokens per step
# (the span between BOS boundaries), and abandons up to max_batch_span tokens
# at the tail of each shard. 1.15 covers both with room to spare; the record's
# own README recommends 9 chunks for a 1770-step run, which this reproduces.
ALIGNMENT_OVERHEAD = 1.15


def shards_needed(cfg: NanoGPTConfig) -> int:
    """Train shards required for the configured token budget (inc. warmup)."""
    steps = cfg.num_iterations if cfg.max_steps is None else min(cfg.num_iterations, cfg.max_steps)
    tokens = (steps + cfg.warmup_steps) * cfg.tokens_per_step
    raw = tokens * ALIGNMENT_OVERHEAD / TOKENS_PER_SHARD
    return int(raw) + 1 + 1  # ceil + one shard of headroom


def shard_names(num_train_shards: int) -> List[str]:
    """Validation shard first (RECORD:571 val_files), then train shards."""
    names = ["fineweb_val_%06d.bin" % 0]
    names += ["fineweb_train_%06d.bin" % i for i in range(1, num_train_shards + 1)]
    return names


# ------------------------------------------------------------- verification


def read_header(path: Path) -> Dict[str, int]:
    with path.open("rb") as fh:
        head = fh.read(12)
    if len(head) < 12:
        raise ValueError(f"{path.name}: file shorter than its header")
    magic, version, num_tokens = struct.unpack("<iii", head)
    if magic != MAGIC:
        raise ValueError(f"{path.name}: magic {magic} != {MAGIC}")
    if version != 1:
        raise ValueError(f"{path.name}: unsupported version {version}")
    return {"magic": magic, "version": version, "num_tokens": num_tokens}


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify(path: Path, manifest: Dict[str, Dict[str, object]], check_hash: bool) -> Dict[str, object]:
    header = read_header(path)
    expected = HEADER_BYTES + 2 * header["num_tokens"]
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path.name}: size {actual} != header-implied {expected} "
            "(truncated or corrupt; delete it and re-run)"
        )
    entry = {"num_tokens": header["num_tokens"], "bytes": actual}
    known = manifest.get(path.name)
    if check_hash or (known and "sha256" in known):
        digest = sha256_of(path)
        if known and known.get("sha256") and known["sha256"] != digest:
            raise ValueError(
                f"{path.name}: sha256 {digest} != recorded {known['sha256']} — "
                "the shard changed on disk; refusing to use it"
            )
        entry["sha256"] = digest
    return entry


# ---------------------------------------------------------------- download


def remote_size(url: str) -> Optional[int]:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req) as resp:
            size = resp.headers.get("Content-Length")
            return int(size) if size else None
    except urllib.error.URLError:
        return None


def download(url: str, dest: Path, quiet: bool = False) -> None:
    """Resumable download to ``dest`` via a ``.part`` file + Range requests."""
    part = dest.with_suffix(dest.suffix + ".part")
    total = remote_size(url)
    have = part.stat().st_size if part.exists() else 0
    if total is not None and have == total:
        part.rename(dest)
        return
    headers = {}
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        if not quiet:
            print(f"  resuming at {have / 1e9:.2f} GB")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, part.open(mode) as fh:
        if have and resp.status != 206:
            # Server ignored the Range header: restart cleanly.
            fh.close()
            part.unlink()
            return download(url, dest, quiet=quiet)
        written = have
        while True:
            block = resp.read(1 << 22)
            if not block:
                break
            fh.write(block)
            written += len(block)
            if not quiet and total:
                pct = 100 * written / (total if resp.status != 206 else total)
                print(f"\r  {written / 1e9:6.2f} / {total / 1e9:.2f} GB ({pct:5.1f}%)", end="")
    if not quiet:
        print()
    part.rename(dest)


# -------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="nanogpt experiment YAML (derives the shard count)")
    parser.add_argument("--shards", type=int, default=None, help="explicit number of train shards")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "fineweb10B")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--verify-hashes", action="store_true", help="sha256 every present shard")
    args = parser.parse_args(argv)

    if args.shards is None:
        if args.config is None:
            parser.error("pass --config <experiment.yaml> or --shards N")
        with open(args.config) as fh:
            raw = yaml.safe_load(fh)
        cfg = NanoGPTConfig.from_config(raw)
        n_train = shards_needed(cfg)
        budget = (cfg.num_iterations + cfg.warmup_steps) * cfg.tokens_per_step
        print(
            f"config {args.config}: {cfg.num_iterations} steps x {cfg.tokens_per_step:,} "
            f"tokens/step (+{cfg.warmup_steps} warmup) = {budget / 1e6:.0f}M tokens"
        )
    else:
        n_train = args.shards

    names = shard_names(n_train)
    total_gb = data_footprint_gb(len(names))
    print(
        f"plan: {len(names)} shards ({n_train} train + 1 val), "
        f"{n_train * TOKENS_PER_SHARD / 1e9:.1f}B train tokens available, "
        f"{total_gb:.1f} GB on disk"
    )
    if args.dry_run:
        return 0

    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "shard_manifest.json"
    manifest: Dict[str, Dict[str, object]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    for name in names:
        dest = args.data_dir / name
        if dest.exists():
            entry = verify(dest, manifest, check_hash=args.verify_hashes)
            manifest[name] = {**manifest.get(name, {}), **entry}
            print(f"ok      {name} ({entry['bytes'] / 1e9:.2f} GB)")
            continue
        print(f"fetch   {name}")
        download(f"{BASE_URL}/{name}", dest)
        entry = verify(dest, manifest, check_hash=True)
        manifest[name] = entry
        print(f"ok      {name} ({entry['bytes'] / 1e9:.2f} GB, sha256 {entry['sha256'][:16]}...)")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
