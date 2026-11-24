#!/usr/bin/env python
"""Compute policy state/action normalization stats for Bridge RLDS data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lang_slot.paths import resources
from lang_slot.data.bridge_rlds import (
    _example_message_class, _feature_float_array, _flatten_tfrecord_actions,
    _iter_tfrecord_records, _reshape_flat_feature,
)

DEFAULT_BRIDGE_TFDS_DIR = str(resources()["BRIDGE_DATASET_DIR"])


def _load_make_policy_actions():
    module_path = REPO_ROOT / "lang_slot" / "policy_actions.py"
    spec = importlib.util.spec_from_file_location("_lang_slot_policy_actions_light", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load policy action helper from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_policy_actions


make_policy_actions = _load_make_policy_actions()


@dataclass
class RunningStats:
    dim: int
    count: int = 0
    total: Optional[np.ndarray] = None
    total_sq: Optional[np.ndarray] = None
    min_value: Optional[np.ndarray] = None
    max_value: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.total = np.zeros((self.dim,), dtype=np.float64)
        self.total_sq = np.zeros((self.dim,), dtype=np.float64)
        self.min_value = np.full((self.dim,), np.inf, dtype=np.float64)
        self.max_value = np.full((self.dim,), -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"Expected values [N,{self.dim}], got {values.shape}.")
        if values.shape[0] == 0:
            return
        self.count += int(values.shape[0])
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.min_value = np.minimum(self.min_value, values.min(axis=0))
        self.max_value = np.maximum(self.max_value, values.max(axis=0))

    def to_json(self, *, eps: float) -> Dict[str, Any]:
        if self.count <= 0:
            raise ValueError("Cannot serialize empty stats.")
        mean = self.total / float(self.count)
        var = self.total_sq / float(self.count) - np.square(mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        std_safe = np.where(np.abs(std) < eps, 1.0, std)
        clipped = np.where(np.abs(std) < eps)[0].astype(int).tolist()
        return {
            "type": "zscore",
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "std_safe": std_safe.astype(float).tolist(),
            "min": self.min_value.astype(float).tolist(),
            "max": self.max_value.astype(float).tolist(),
            "num_dims": int(self.dim),
            "num_samples": int(self.count),
            "clipped_std_dims": clipped,
        }


def _materialize_steps(episode: Dict[str, Any]) -> Optional[Sequence[Dict[str, Any]]]:
    steps = episode.get("steps")
    if steps is None:
        return None
    if isinstance(steps, dict):
        first_value = next(iter(steps.values()))
        length = len(first_value)
        return [{key: value[i] for key, value in steps.items()} for i in range(length)]
    return list(steps)


def _get_nested(data: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("/".join(path))
        cur = cur[key]
    return cur


def _get_step_sequence(steps: Sequence[Dict[str, Any]], path: Sequence[str]) -> np.ndarray:
    if len(path) < 2 or path[0] != "steps":
        raise ValueError(f"Expected a step path beginning with 'steps', got {path}.")
    values = [_get_nested(step, path[1:]) for step in steps]
    try:
        return np.stack([np.asarray(value) for value in values], axis=0)
    except Exception:
        return np.asarray(values)


def _flatten_bridge_actions(
    steps: Sequence[Dict[str, Any]],
    *,
    include_terminate: bool = False,
) -> np.ndarray:
    try:
        action = np.asarray(_get_step_sequence(steps, ("steps", "action")), dtype=np.float32)
    except KeyError:
        action = None
    if action is not None:
        if action.ndim == 1:
            action = action[:, None]
        if action.ndim != 2 or action.shape[-1] < 7:
            raise ValueError(f"Expected tensor action [T,>=7], got {action.shape}.")
        if include_terminate:
            try:
                terminate = np.asarray(
                    _get_step_sequence(steps, ("steps", "action", "terminate_episode")),
                    dtype=np.float32,
                )
            except KeyError:
                terminate = None
            if terminate is not None:
                if terminate.ndim == 1:
                    terminate = terminate[:, None]
                action = np.concatenate([action[:, :7], terminate], axis=-1)
        return action[:, : 8 if include_terminate and action.shape[-1] > 7 else 7]

    paths = [
        ("steps", "action", "world_vector"),
        ("steps", "action", "rotation_delta"),
        ("steps", "action", "open_gripper"),
    ]
    if include_terminate:
        paths.append(("steps", "action", "terminate_episode"))

    parts = []
    for path in paths:
        array = np.asarray(_get_step_sequence(steps, path), dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError(f"Expected action component [T,D], got {array.shape} for {'/'.join(path)}.")
        parts.append(array)
    lengths = {part.shape[0] for part in parts}
    if len(lengths) != 1:
        raise ValueError(f"Action components have inconsistent lengths: {sorted(lengths)}.")
    return np.concatenate(parts, axis=-1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_BRIDGE_TFDS_DIR, help="Bridge TFDS root directory.")
    parser.add_argument("--split", default="train", help="TFDS split to scan.")
    parser.add_argument(
        "--output",
        default=str(resources()["POLICY_OUTPUT_DIR"] / "bridge_train_norm_stats.json"),
        help="Output stats JSON path.",
    )
    parser.add_argument("--shard-indices", default="", help="Comma-separated physical TFDS shard ids.")
    parser.add_argument("--max-episodes", type=int, default=0, help="Debug limit; 0 scans all selected episodes.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print one JSON progress row every N episodes.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Std clipping epsilon.")
    parser.add_argument(
        "--reader",
        choices=("auto", "tfrecord", "tfds"),
        default="auto",
        help="Reader backend. 'tfrecord' parses only state/action and is best for large Bridge 1.0.0 scans.",
    )
    parser.add_argument(
        "--raw-action",
        action="store_true",
        help="Use raw Bridge world_vector/rotation_delta/open_gripper instead of state-delta policy actions.",
    )
    parser.add_argument(
        "--include-terminate-action",
        action="store_true",
        help="Append terminate_episode to raw actions before relabeling.",
    )
    return parser


def parse_shard_indices(value: str) -> Optional[tuple[int, ...]]:
    text = str(value).strip()
    if not text:
        return None
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(part))
    return tuple(indices)


def import_tfds():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    import tensorflow_datasets as tfds

    return tfds


def image_skip_decoders(builder: Any, tfds: Any) -> Dict[str, Any]:
    """Skip image decoding while preserving state/action tensors for stats."""
    try:
        obs_features = builder.info.features["steps"].feature["observation"]
        image_keys = [
            key
            for key in obs_features.keys()
            if str(key).startswith("image")
        ]
    except Exception:
        image_keys = []
    if not image_keys:
        return {}
    return {
        "steps": {
            "observation": {
                key: tfds.decode.SkipDecoding()
                for key in image_keys
            }
        }
    }


def iter_episodes(data_dir: str, split: str, shard_indices: Optional[Sequence[int]]) -> Iterable[tuple[int, Any]]:
    tfds = import_tfds()
    print(json.dumps({"event": "tfds_imported"}), flush=True)
    builder = tfds.builder_from_directory(str(Path(data_dir).expanduser()))
    decoders = image_skip_decoders(builder, tfds)
    if split not in builder.info.splits:
        raise ValueError(f"Unknown split {split!r}; available splits are {list(builder.info.splits)}.")
    num_shards = int(builder.info.splits[split].num_shards)
    shards = list(range(num_shards)) if shard_indices is None else [int(i) for i in shard_indices]
    invalid = [i for i in shards if i < 0 or i >= num_shards]
    if invalid:
        raise ValueError(f"Shard indices {invalid} exceed split {split!r} with {num_shards} shards.")

    for shard_index in shards:
        print(json.dumps({"event": "open_shard", "shard_index": int(shard_index)}), flush=True)
        dataset = builder.as_dataset(
            split=f"{split}[{shard_index}shard]",
            shuffle_files=False,
            decoders=decoders,
        )
        for episode in tfds.as_numpy(dataset):
            yield shard_index, episode


def import_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    tf.get_logger().setLevel("ERROR")
    return tf


def _tfrecord_shards(data_dir: str, split: str, shard_indices: Optional[Sequence[int]]) -> list[tuple[int, Path]]:
    data_path = Path(data_dir).expanduser()
    pattern = f"*-{split}.tfrecord-*-of-*"
    matches = sorted(data_path.glob(pattern))
    if not matches:
        return []

    shards = []
    for path in matches:
        match = re.search(r"\.tfrecord-(\d+)-of-\d+$", path.name)
        if match is None:
            continue
        shards.append((int(match.group(1)), path))
    if shard_indices is not None:
        wanted = {int(index) for index in shard_indices}
        shards = [(index, path) for index, path in shards if index in wanted]
        missing = sorted(wanted - {index for index, _ in shards})
        if missing:
            raise ValueError(f"Shard indices not found for split {split!r}: {missing}")
    return shards


def iter_tfrecord_arrays(
    data_dir: str,
    split: str,
    shard_indices: Optional[Sequence[int]],
    *,
    include_terminate: bool,
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    example_class = _example_message_class()
    shards = _tfrecord_shards(data_dir, split, shard_indices)
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards matching '*-{split}.tfrecord-*-of-*' in {data_dir}.")
    print(json.dumps({"event": "tfrecord_reader", "num_shards": len(shards)}), flush=True)
    for shard_index, path in shards:
        print(json.dumps({"event": "open_shard", "shard_index": int(shard_index), "path": str(path)}), flush=True)
        for raw in _iter_tfrecord_records(path):
            example = example_class.FromString(raw)
            state_values = _feature_float_array(example, "steps/observation/state")
            if state_values is None:
                raise KeyError("steps/observation/state")
            states = _reshape_flat_feature(state_values, 7, "steps/observation/state")
            raw_actions = _flatten_tfrecord_actions(
                example,
                length=int(states.shape[0]),
                include_terminate=include_terminate,
            )
            yield shard_index, states, raw_actions


def split_action_stats(policy_actions: np.ndarray, *, state_delta_action: bool) -> np.ndarray:
    if state_delta_action:
        return policy_actions[:-1, :6]
    return policy_actions[:, :6]


def main() -> None:
    args = build_argparser().parse_args()
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")

    state_delta_action = not bool(args.raw_action)
    state_stats = RunningStats(dim=7)
    action_stats = RunningStats(dim=6)
    gripper_stats = RunningStats(dim=1)
    processed_episodes = 0
    processed_frames = 0
    processed_action_samples = 0

    shard_indices = parse_shard_indices(args.shard_indices) if args.shard_indices else None
    use_tfrecord = args.reader == "tfrecord" or (
        args.reader == "auto" and bool(_tfrecord_shards(args.data_dir, args.split, shard_indices))
    )
    if use_tfrecord:
        arrays_iter = iter_tfrecord_arrays(
            args.data_dir,
            args.split,
            shard_indices,
            include_terminate=bool(args.include_terminate_action),
        )
    else:
        def _tfds_arrays():
            for shard_index, episode in iter_episodes(args.data_dir, args.split, shard_indices):
                steps = _materialize_steps(episode)
                if steps is None:
                    raise ValueError("Expected RLDS episode['steps'] to be an iterable step dataset.")
                states = np.asarray(_get_step_sequence(steps, ("steps", "observation", "state")), dtype=np.float32)
                raw_actions = _flatten_bridge_actions(steps, include_terminate=bool(args.include_terminate_action))
                yield shard_index, states, raw_actions
        arrays_iter = _tfds_arrays()

    for shard_index, states, raw_actions in arrays_iter:
        policy_actions = make_policy_actions(
            states,
            raw_actions,
            state_delta_action=state_delta_action,
        )
        state_stats.update(states[:, :7])
        action_values = split_action_stats(policy_actions, state_delta_action=state_delta_action)
        action_stats.update(action_values)
        gripper_values = policy_actions[:-1, 6:7] if state_delta_action else policy_actions[:, 6:7]
        gripper_stats.update(gripper_values)

        processed_episodes += 1
        processed_frames += int(states.shape[0])
        processed_action_samples += int(action_values.shape[0])
        if args.progress_interval > 0 and processed_episodes % int(args.progress_interval) == 0:
            print(
                json.dumps(
                    {
                        "processed_episodes": processed_episodes,
                        "processed_frames": processed_frames,
                        "processed_action_samples": processed_action_samples,
                        "last_shard_index": int(shard_index),
                    }
                ),
                flush=True,
            )
        if args.max_episodes > 0 and processed_episodes >= args.max_episodes:
            break

    state_json = state_stats.to_json(eps=float(args.eps))
    action_json = action_stats.to_json(eps=float(args.eps))
    gripper_json = gripper_stats.to_json(eps=float(args.eps))
    action_mean = np.asarray(action_json["mean"], dtype=np.float64)
    action_std = np.asarray(action_json["std"], dtype=np.float64)
    action_std_safe = np.asarray(action_json["std_safe"], dtype=np.float64)
    action_min = np.asarray(action_json["min"], dtype=np.float64)
    action_max = np.asarray(action_json["max"], dtype=np.float64)
    action_key = "steps/action/state_delta" if state_delta_action else "steps/action/raw_action"
    action_mode = "state_delta" if state_delta_action else "raw_action"
    action_note = (
        "For state_delta_action=True, action stats exclude the terminal all-zero action."
        if state_delta_action
        else "For state_delta_action=False, action stats use raw Bridge action rows."
    )
    action_description = (
        "Action dimensions 0:6 are z-score normalized. With state_delta_action=True they are "
        "state[t+1,:6] - state[t,:6]. The gripper dimension is passed through unchanged."
        if state_delta_action
        else "Action dimensions 0:6 are z-score normalized from raw Bridge action fields. "
        "The gripper dimension is passed through unchanged."
    )

    output = {
        "description": "Bridge policy normalization stats generated for lang_slot flow-matching policy training.",
        "source": {
            "dataset": "BridgeData V2",
            "data_dir": str(Path(args.data_dir).expanduser()),
            "split": args.split,
            "reader": "tfrecord" if use_tfrecord else "tfds",
            "state_delta_action": bool(state_delta_action),
            "policy_action_key": action_key,
            "policy_action_mode": action_mode,
            "processed_records": int(processed_episodes),
            "num_state_samples": int(processed_frames),
            "num_action_samples": int(processed_action_samples),
            "shard_indices": None if shard_indices is None else [int(i) for i in shard_indices],
            "note": action_note,
        },
        "normalization": {
            "steps/observation/state": state_json,
            action_key: action_json,
            "steps/action/world_vector": {
                **action_json,
                "mean": action_mean[:3].astype(float).tolist(),
                "std": action_std[:3].astype(float).tolist(),
                "std_safe": action_std_safe[:3].astype(float).tolist(),
                "min": action_min[:3].astype(float).tolist(),
                "max": action_max[:3].astype(float).tolist(),
                "num_dims": 3,
            },
            "steps/action/rotation_delta": {
                **action_json,
                "mean": action_mean[3:6].astype(float).tolist(),
                "std": action_std[3:6].astype(float).tolist(),
                "std_safe": action_std_safe[3:6].astype(float).tolist(),
                "min": action_min[3:6].astype(float).tolist(),
                "max": action_max[3:6].astype(float).tolist(),
                "num_dims": 3,
            },
            "steps/action/open_gripper": gripper_json,
        },
        "policy_notes": {
            "state": "All 7 state dimensions are z-score normalized.",
            "action": action_description,
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"output": str(output_path), "processed_records": processed_episodes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
