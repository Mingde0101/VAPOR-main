#!/usr/bin/env python
"""Build strict-valid Bridge train/val TFRecord indexes from an audit JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    return parser.parse_args()


def is_valid(row: dict[str, Any]) -> bool:
    stages = row.get("filter_stages", {})
    return all(
        stages.get(stage, {}).get("status") == "pass"
        for stage in ("source_policy", "semantic_presence", "semantic_validity")
    )


def load_manifest(
    path: Path,
    splits: set[str],
) -> tuple[dict[str, dict[int, dict[int, tuple[int, int]]]], dict[str, int]]:
    selected = {split: {} for split in splits}
    totals = {split: 0 for split in splits}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            split = str(row.get("split", ""))
            if split not in splits:
                continue
            totals[split] += 1
            if not is_valid(row):
                continue
            shard = int(row["shard_number"])
            record = int(row["record_index"])
            records = selected[split].setdefault(shard, {})
            if record in records:
                raise ValueError(f"Duplicate {split}/{shard}/{record} at line {line_number}.")
            records[record] = (int(row["num_steps"]), int(row["episode_id"]))
    return selected, totals


def find_shards(data_dir: Path, split: str) -> dict[int, Path]:
    result = {}
    for path in sorted(data_dir.glob(f"*-{split}.tfrecord-*-of-*")):
        match = re.search(r"\.tfrecord-(\d+)-of-\d+$", path.name)
        if match:
            result[int(match.group(1))] = path
    if not result:
        raise FileNotFoundError(f"No {split} TFRecord shards in {data_dir}.")
    return result


def scan_shard(path: Path) -> list[tuple[int, int]]:
    offsets = []
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            header = f.read(12)
            if not header:
                return offsets
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header in {path}.")
            length = int(struct.unpack("<Q", header[:8])[0])
            f.seek(length, os.SEEK_CUR)
            if len(f.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord payload/footer in {path}.")
            offsets.append((offset, length))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(
    split: str,
    selected: dict[int, dict[int, tuple[int, int]]],
    manifest_total: int,
    *,
    manifest: Path,
    manifest_sha256: str,
    data_dir: Path,
    output_dir: Path,
) -> None:
    shard_paths = find_shards(data_dir, split)
    if set(selected) - set(shard_paths):
        raise ValueError(f"Manifest references missing {split} shards.")

    entries = []
    shard_info = []
    total_records = 0
    for shard, path in sorted(shard_paths.items()):
        offsets = scan_shard(path)
        total_records += len(offsets)
        shard_info.append((shard, path.name, path.stat().st_size, len(offsets)))
        for record, (num_steps, episode_id) in sorted(selected.get(shard, {}).items()):
            if record >= len(offsets):
                raise ValueError(f"Missing TFRecord entry {split}/{shard}/{record}.")
            offset, length = offsets[record]
            entries.append((shard, record, offset, length, num_steps, episode_id))
    if total_records != manifest_total:
        raise ValueError(
            f"Manifest/TFRecord count mismatch for {split}: {manifest_total} != {total_records}."
        )

    values = np.asarray(entries, dtype=np.int64)
    shard_numbers = np.asarray([row[0] for row in shard_info], dtype=np.int32)
    shard_names = np.asarray([row[1] for row in shard_info])
    shard_sizes = np.asarray([row[2] for row in shard_info], dtype=np.int64)
    shard_counts = np.asarray([row[3] for row in shard_info], dtype=np.int32)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bridge_1_0_0_{split}_semantic_valid_v1"
    index_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        index_path,
        schema_version=np.asarray(1, dtype=np.int32),
        split=np.asarray(split),
        shard_number=values[:, 0].astype(np.int32),
        record_index=values[:, 1].astype(np.int32),
        byte_offset=values[:, 2],
        record_length=values[:, 3],
        num_steps=values[:, 4].astype(np.int32),
        episode_id=values[:, 5],
        indexed_shard_number=shard_numbers,
        indexed_shard_filename=shard_names,
        indexed_shard_file_size=shard_sizes,
        indexed_shard_record_count=shard_counts,
    )
    fingerprint_rows = [
        f"{number}:{name}:{size}:{count}"
        for number, name, size, count in shard_info
    ]
    metadata = {
        "schema_version": 1,
        "dataset_version": "1.0.0",
        "split": split,
        "selection": {
            "source_policy": "pass",
            "semantic_presence": "pass",
            "semantic_validity": "pass",
        },
        "num_manifest_trajectories": manifest_total,
        "num_trajectories": int(len(values)),
        "num_timesteps": int(values[:, 4].sum()),
        "num_shards": int(len(shard_info)),
        "manifest_path": str(manifest),
        "audit_manifest_sha256": manifest_sha256,
        "dataset_dir": str(data_dir),
        "dataset_fingerprint": hashlib.sha256("\n".join(fingerprint_rows).encode()).hexdigest(),
        "index_file": index_path.name,
        "index_size_bytes": index_path.stat().st_size,
    }
    metadata_path = output_dir / f"{stem}.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"index": str(index_path), **metadata}), flush=True)


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    splits = {str(split) for split in args.splits}
    selected, totals = load_manifest(manifest, splits)
    manifest_sha256 = file_sha256(manifest)
    for split in args.splits:
        build_split(
            split,
            selected[split],
            totals[split],
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            data_dir=data_dir,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
