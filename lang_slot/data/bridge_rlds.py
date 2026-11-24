"""Bridge RLDS/TFDS continuous clip reader.

The dataset intentionally keeps TensorFlow imports local to iteration time so
the rest of ``lang_slot`` remains PyTorch-native.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import re
import struct
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from lang_slot.paths import resources, resolve_path_string, work_path
from lang_slot.policy_actions import make_policy_actions


DEFAULT_BRIDGE_TFDS_DIR = str(resources()["BRIDGE_DATASET_DIR"])
DEFAULT_BRIDGE_TRAIN_NORM_STATS = str(
    work_path(
        "configs",
        "bridge_1_0_0_train_state_delta_action_norm_stats.json",
    )
)
PathSpec = Sequence[str]


@dataclass
class BridgeClipSample:
    """One continuous video clip and aligned semantic slot queries."""

    images: torch.Tensor
    semantic_queries: torch.Tensor
    contrastive_loss_mask: torch.Tensor
    instruction: str
    start_index: int
    flow_target: Optional[torch.Tensor] = None


@dataclass
class BridgeTrajectorySample:
    """One full Bridge trajectory for rolling-window video slot encoding."""

    images: torch.Tensor
    instruction: str
    trajectory_index: int
    flow_target: Optional[torch.Tensor] = None


@dataclass
class PolicyWindowSample:
    """One policy sample centered at a trajectory timestep.

    Every padding mask uses ``True`` to mark a padded position.
    """

    hist_images: torch.Tensor
    hist_states: torch.Tensor
    hist_actions: torch.Tensor
    future_images: torch.Tensor
    future_states: torch.Tensor
    future_actions: torch.Tensor
    hist_image_padding_mask: torch.Tensor
    hist_state_padding_mask: torch.Tensor
    hist_action_padding_mask: torch.Tensor
    future_frame_padding_mask: torch.Tensor
    future_state_padding_mask: torch.Tensor
    future_action_padding_mask: torch.Tensor
    future_video_padding_mask: torch.Tensor
    instruction: str
    timestep: int
    trajectory_index: int
    shard_index: int
    # Export-side provenance and filter labels. Generic Bridge datasets leave
    # this empty; filtered subsets populate it from export_manifest.jsonl.
    trajectory_metadata: Dict[str, Any] = field(default_factory=dict)


def _get_nested(data: Dict[str, Any], path: PathSpec) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("/".join(path))
        cur = cur[key]
    return cur


def _try_get_nested(data: Dict[str, Any], paths: Iterable[PathSpec]) -> Any:
    for path in paths:
        try:
            return _get_nested(data, path)
        except KeyError:
            continue
    return None


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        return item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
    if len(arr) == 0:
        return ""
    item = arr[0]
    return item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)


def _select_time(value: Any, start: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] > start:
        return arr[start]
    return arr


def _is_step_path(path: PathSpec) -> bool:
    return len(path) > 0 and path[0] == "steps"


def _materialize_steps(episode: Dict[str, Any]) -> Optional[Sequence[Dict[str, Any]]]:
    steps = episode.get("steps")
    if steps is None:
        return None
    if isinstance(steps, dict):
        first_value = next(iter(steps.values()))
        length = len(first_value)
        return [{key: value[i] for key, value in steps.items()} for i in range(length)]
    return list(steps)


def _get_step_sequence(steps: Sequence[Dict[str, Any]], path: PathSpec) -> np.ndarray:
    values = [_get_nested(step, path[1:]) for step in steps]
    try:
        return np.stack([np.asarray(value) for value in values], axis=0)
    except Exception:
        return np.asarray(values)


def _try_get_step_sequence(
    steps: Sequence[Dict[str, Any]],
    paths: Iterable[PathSpec],
) -> Any:
    for path in paths:
        if not _is_step_path(path):
            continue
        try:
            return _get_step_sequence(steps, path)
        except KeyError:
            continue
    return None


def _flatten_bridge_actions(
    steps: Sequence[Dict[str, Any]],
    *,
    include_terminate: bool = False,
) -> np.ndarray:
    """Flatten Bridge actions to [T,7], optionally appending termination."""
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


def _safe_std(std: np.ndarray, eps: float) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32)
    return np.where(np.abs(std) < eps, 1.0, std).astype(np.float32)


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_trajectory_metadata_manifest(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Index an exported Bridge subset manifest by output file and record.

    The exporter writes one JSON object per trajectory.  The pair
    ``(output_shard, output_record_index)`` is stable and maps directly to the
    physical TFRecord reader used by :class:`PolicyWindowDataset`.
    """
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Trajectory metadata manifest not found: {manifest_path}")

    records: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest_path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {manifest_path} at line {line_number}."
                )
            missing = [key for key in ("output_shard", "output_record_index") if key not in row]
            if missing:
                raise ValueError(
                    f"Manifest line {line_number} is missing required fields: {missing}."
                )
            key = (str(row["output_shard"]), int(row["output_record_index"]))
            if key in records:
                raise ValueError(f"Duplicate trajectory location {key!r} in {manifest_path}.")
            records[key] = dict(row)
    if not records:
        raise ValueError(f"Trajectory metadata manifest is empty: {manifest_path}")
    return records


@lru_cache(maxsize=1)
def _example_message_class() -> Any:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "tensorflow/core/example/example.proto"
    file_desc.package = "tensorflow"
    file_desc.syntax = "proto3"

    def add_message(name: str) -> Any:
        message = file_desc.message_type.add()
        message.name = name
        return message

    def add_field(
        message: Any,
        name: str,
        number: int,
        label: int,
        field_type: int,
        type_name: str = "",
    ) -> Any:
        field = message.field.add()
        field.name = name
        field.number = number
        field.label = label
        field.type = field_type
        if type_name:
            field.type_name = type_name
        return field

    label_optional = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    label_repeated = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    type_bytes = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    type_float = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    type_int64 = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    type_message = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    type_string = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    bytes_list = add_message("BytesList")
    add_field(bytes_list, "value", 1, label_repeated, type_bytes)

    float_list = add_message("FloatList")
    float_value = add_field(float_list, "value", 1, label_repeated, type_float)
    float_value.options.packed = True

    int64_list = add_message("Int64List")
    int64_value = add_field(int64_list, "value", 1, label_repeated, type_int64)
    int64_value.options.packed = True

    feature = add_message("Feature")
    feature.oneof_decl.add().name = "kind"
    for name, number, type_name in (
        ("bytes_list", 1, ".tensorflow.BytesList"),
        ("float_list", 2, ".tensorflow.FloatList"),
        ("int64_list", 3, ".tensorflow.Int64List"),
    ):
        field = add_field(feature, name, number, label_optional, type_message, type_name)
        field.oneof_index = 0

    features = add_message("Features")
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    add_field(entry, "key", 1, label_optional, type_string)
    add_field(entry, "value", 2, label_optional, type_message, ".tensorflow.Feature")
    add_field(features, "feature", 1, label_repeated, type_message, ".tensorflow.Features.FeatureEntry")

    example = add_message("Example")
    add_field(example, "features", 1, label_optional, type_message, ".tensorflow.Features")

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_desc)
    descriptor = pool.FindMessageTypeByName("tensorflow.Example")
    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(descriptor)
    return message_factory.MessageFactory(pool).GetPrototype(descriptor)


def _iter_tfrecord_records(path: Path) -> Iterator[bytes]:
    with Path(path).open("rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header in {path}.")
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            if len(data) != length:
                raise ValueError(f"Truncated TFRecord payload in {path}.")
            footer = f.read(4)
            if len(footer) != 4:
                raise ValueError(f"Truncated TFRecord footer in {path}.")
            yield data


def _iter_selected_tfrecord_records(
    path: Path,
    records: Sequence[Tuple[int, int, int]],
) -> Iterator[Tuple[int, bytes]]:
    """Read selected TFRecord payloads by (record_index, byte_offset, length)."""
    with Path(path).open("rb") as f:
        for record_index, byte_offset, expected_length in records:
            f.seek(int(byte_offset))
            header = f.read(12)
            if len(header) != 12:
                raise ValueError(
                    f"Cannot read TFRecord header at offset {byte_offset} in {path}."
                )
            length = struct.unpack("<Q", header[:8])[0]
            if length != int(expected_length):
                raise ValueError(
                    "Episode index does not match TFRecord payload length: "
                    f"{path}, record={record_index}, offset={byte_offset}, "
                    f"index_length={expected_length}, file_length={length}."
                )
            data = f.read(length)
            if len(data) != length:
                raise ValueError(
                    f"Truncated TFRecord payload in {path} at record {record_index}."
                )
            footer = f.read(4)
            if len(footer) != 4:
                raise ValueError(
                    f"Truncated TFRecord footer in {path} at record {record_index}."
                )
            yield int(record_index), data


@lru_cache(maxsize=8)
def _load_policy_episode_index(
    path: str,
    *,
    split: str,
) -> Tuple[
    Dict[int, Tuple[Tuple[int, int, int], ...]],
    Dict[int, Tuple[str, int, int]],
]:
    """Load a compact valid-episode index grouped by physical shard."""
    index_path = Path(resolve_path_string(path))
    if not index_path.is_file():
        raise FileNotFoundError(f"Policy episode index does not exist: {index_path}")

    required = {
        "split",
        "shard_number",
        "record_index",
        "byte_offset",
        "record_length",
        "indexed_shard_number",
        "indexed_shard_filename",
        "indexed_shard_file_size",
        "indexed_shard_record_count",
    }
    with np.load(index_path, allow_pickle=False) as index:
        missing = sorted(required - set(index.files))
        if missing:
            raise ValueError(f"Episode index {index_path} is missing arrays: {missing}")
        index_split = str(np.asarray(index["split"]).item())
        if index_split != split:
            raise ValueError(
                f"Episode index split mismatch: index={index_split!r}, requested={split!r}."
            )
        shard_numbers = np.asarray(index["shard_number"], dtype=np.int64)
        record_indices = np.asarray(index["record_index"], dtype=np.int64)
        byte_offsets = np.asarray(index["byte_offset"], dtype=np.int64)
        record_lengths = np.asarray(index["record_length"], dtype=np.int64)
        indexed_shards = np.asarray(index["indexed_shard_number"], dtype=np.int64)
        indexed_filenames = np.asarray(index["indexed_shard_filename"])
        indexed_file_sizes = np.asarray(index["indexed_shard_file_size"], dtype=np.int64)
        indexed_record_counts = np.asarray(index["indexed_shard_record_count"], dtype=np.int64)

    if len({len(shard_numbers), len(record_indices), len(byte_offsets), len(record_lengths)}) != 1:
        raise ValueError(f"Episode index arrays have inconsistent lengths in {index_path}.")
    if np.any(shard_numbers < 0) or np.any(record_indices < 0):
        raise ValueError(f"Episode index contains negative shard/record indices: {index_path}")
    if np.any(byte_offsets < 0) or np.any(record_lengths <= 0):
        raise ValueError(f"Episode index contains invalid TFRecord offsets/lengths: {index_path}")
    if len(
        {
            len(indexed_shards),
            len(indexed_filenames),
            len(indexed_file_sizes),
            len(indexed_record_counts),
        }
    ) != 1:
        raise ValueError(f"Episode index shard identity arrays are inconsistent: {index_path}")

    grouped: Dict[int, list[Tuple[int, int, int]]] = {}
    previous: Optional[Tuple[int, int]] = None
    for shard, record, offset, length in zip(
        shard_numbers.tolist(),
        record_indices.tolist(),
        byte_offsets.tolist(),
        record_lengths.tolist(),
    ):
        key = (int(shard), int(record))
        if previous is not None and key <= previous:
            raise ValueError(
                f"Episode index entries must be unique and sorted by shard/record: {index_path}"
            )
        previous = key
        grouped.setdefault(int(shard), []).append((int(record), int(offset), int(length)))

    shard_identities = {
        int(shard): (str(filename), int(file_size), int(record_count))
        for shard, filename, file_size, record_count in zip(
            indexed_shards.tolist(),
            indexed_filenames.tolist(),
            indexed_file_sizes.tolist(),
            indexed_record_counts.tolist(),
        )
    }
    if len(shard_identities) != len(indexed_shards):
        raise ValueError(f"Episode index contains duplicate shard identities: {index_path}")
    return (
        {shard: tuple(records) for shard, records in grouped.items()},
        shard_identities,
    )


def _feature_float_array(example: Any, key: str) -> Optional[np.ndarray]:
    feature = example.features.feature.get(key)
    if feature is None or feature.WhichOneof("kind") != "float_list":
        return None
    return np.asarray(feature.float_list.value, dtype=np.float32)


def _feature_bytes_list(example: Any, key: str) -> Optional[Sequence[bytes]]:
    feature = example.features.feature.get(key)
    if feature is None or feature.WhichOneof("kind") != "bytes_list":
        return None
    return list(feature.bytes_list.value)


def _reshape_flat_feature(values: np.ndarray, dim: int, key: str) -> np.ndarray:
    if values.size % dim != 0:
        raise ValueError(f"Feature {key!r} length {values.size} is not divisible by {dim}.")
    return values.reshape((-1, dim))


def _flatten_tfrecord_actions(
    example: Any,
    *,
    length: int,
    include_terminate: bool = False,
) -> np.ndarray:
    action = _feature_float_array(example, "steps/action")
    if action is not None:
        if action.size % length != 0:
            raise ValueError(
                f"Feature 'steps/action' length {action.size} is not divisible by trajectory length {length}."
            )
        action = action.reshape((length, action.size // length))
        if action.shape[-1] < 7:
            raise ValueError(f"Expected tensor action [T,>=7], got {action.shape}.")
        return action[:, :7]

    parts = []
    for key, dim in (
        ("steps/action/world_vector", 3),
        ("steps/action/rotation_delta", 3),
        ("steps/action/open_gripper", 1),
    ):
        values = _feature_float_array(example, key)
        if values is None:
            raise KeyError(key)
        part = _reshape_flat_feature(values, dim, key)
        if part.shape[0] != length:
            raise ValueError(f"Feature {key!r} length mismatch: {part.shape[0]} vs {length}.")
        parts.append(part)

    if include_terminate:
        values = _feature_float_array(example, "steps/action/terminate_episode")
        if values is not None:
            part = _reshape_flat_feature(values, 1, "steps/action/terminate_episode")
            if part.shape[0] != length:
                raise ValueError(f"terminate_episode length mismatch: {part.shape[0]} vs {length}.")
            parts.append(part)
    return np.concatenate(parts, axis=-1)


def _decode_jpeg_sequence(values: Sequence[bytes], key: str) -> np.ndarray:
    import io
    from PIL import Image

    images = []
    for value in values:
        with Image.open(io.BytesIO(value)) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if not images:
        raise ValueError(f"Feature {key!r} is empty.")
    return np.stack(images, axis=0)


def _tfrecord_shards(
    data_dir: str,
    split: str,
    shard_indices: Optional[Sequence[int]] = None,
) -> list[tuple[int, Path]]:
    data_path = Path(data_dir).expanduser()
    matches = sorted(data_path.glob(f"*-{split}.tfrecord-*-of-*"))
    shards = []
    for path in matches:
        match = re.search(r"\.tfrecord-(\d+)-of-\d+$", path.name)
        if match is not None:
            shards.append((int(match.group(1)), path))
    if shard_indices is not None:
        wanted = {int(index) for index in shard_indices}
        shards = [(index, path) for index, path in shards if index in wanted]
        missing = sorted(wanted - {index for index, _ in shards})
        if missing:
            raise ValueError(f"Shard indices not found for split {split!r}: {missing}")
    return shards


def _all_tfrecord_shards(data_dir: str) -> list[tuple[int, Path]]:
    """Return every physical TFRecord shard under ``data_dir`` as one split.

    The returned indices are global across the original TFDS splits. This is
    useful for terminal fine-tuning where train and validation episodes should
    all participate without copying/re-exporting the TFRecord payloads.
    """
    data_path = Path(data_dir).expanduser()
    paths = sorted(data_path.glob("*.tfrecord-*-of-*"))
    return list(enumerate(paths))


def _default_policy_norm_stats_path(data_dir: str) -> Optional[str]:
    train_stats = Path(DEFAULT_BRIDGE_TRAIN_NORM_STATS)
    if train_stats.exists():
        return str(train_stats)

    data_path = Path(data_dir)
    candidates = sorted(data_path.glob("dataset_statistics_*.json"))
    if candidates:
        return str(candidates[0])
    return None


def policy_episode_index_path(
    data_config: Dict[str, Any],
    split: str,
) -> Optional[str]:
    """Resolve the configured strict-valid episode index for a dataset split."""
    if not bool(data_config.get("valid_trajectories_only", False)):
        return None
    paths = data_config.get("valid_trajectory_index_paths")
    if paths is None:
        return str(resources()["BRIDGE_INDEX_DIR"] / f"bridge_1_0_0_{split}_semantic_valid_v1.npz")
    if not isinstance(paths, dict):
        raise TypeError("data.valid_trajectory_index_paths must be a split-to-path mapping.")
    path = paths.get(split)
    if not path:
        raise ValueError(
            "data.valid_trajectories_only=true but no index path is configured for "
            f"split {split!r} in data.valid_trajectory_index_paths."
        )
    return resolve_path_string(str(path))


def _policy_norm_arrays(
    stats: Dict[str, Any],
    *,
    state_dim: int,
    action_dim: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract state/action normalization arrays from supported Bridge stats files.

    Returns state_mean, state_std, action_mean, action_std, action_mask where
    action_mask=True means that dimension should be normalized. The gripper
    control and optional terminate flag are intentionally passthrough.
    """
    if "normalization" in stats:
        norm = stats["normalization"]
        state_stats = norm["steps/observation/state"]
        state_mean = np.asarray(state_stats["mean"], dtype=np.float32)
        state_std = _safe_std(state_stats.get("std_safe", state_stats["std"]), eps)
        action_mean = np.zeros((action_dim,), dtype=np.float32)
        action_std = np.ones((action_dim,), dtype=np.float32)
        source = stats.get("source", {}) if isinstance(stats.get("source", {}), dict) else {}
        source_is_state_delta = source.get("state_delta_action")
        action_stats = None
        if source_is_state_delta is False:
            action_stats = norm.get("steps/action/raw_action")
            if action_stats is None:
                raise KeyError("Raw-action normalization stats must contain 'steps/action/raw_action'.")
        elif source_is_state_delta is True:
            action_stats = norm.get("steps/action/state_delta")
            if action_stats is None:
                raise KeyError("State-delta normalization stats must contain 'steps/action/state_delta'.")
        elif "steps/action/state_delta" in norm:
            action_stats = norm["steps/action/state_delta"]
        elif "steps/action/raw_action" in norm:
            action_stats = norm["steps/action/raw_action"]

        if action_stats is not None:
            source_mean = np.asarray(action_stats["mean"], dtype=np.float32)
            source_std = _safe_std(action_stats.get("std_safe", action_stats["std"]), eps)
            action_mean[: min(6, source_mean.shape[0], action_dim)] = source_mean[: min(6, source_mean.shape[0], action_dim)]
            action_std[: min(6, source_std.shape[0], action_dim)] = source_std[: min(6, source_std.shape[0], action_dim)]
        else:
            world_stats = norm["steps/action/world_vector"]
            rot_stats = norm["steps/action/rotation_delta"]
            action_mean[:3] = np.asarray(world_stats["mean"], dtype=np.float32)
            action_std[:3] = _safe_std(world_stats.get("std_safe", world_stats["std"]), eps)
            action_mean[3:6] = np.asarray(rot_stats["mean"], dtype=np.float32)
            action_std[3:6] = _safe_std(rot_stats.get("std_safe", rot_stats["std"]), eps)
    else:
        # TFDS/OCTO style dataset_statistics file: action is already [world, rotation, gripper].
        state_stats = stats.get("proprio") or stats.get("state")
        action_stats = stats["action"]
        if state_stats is None:
            raise KeyError("Normalization stats must contain 'proprio'/'state' or 'normalization'.")
        state_mean = np.asarray(state_stats["mean"], dtype=np.float32)
        state_std = _safe_std(np.asarray(state_stats["std"], dtype=np.float32), eps)
        action_mean = np.zeros((action_dim,), dtype=np.float32)
        action_std = np.ones((action_dim,), dtype=np.float32)
        source_mean = np.asarray(action_stats["mean"], dtype=np.float32)
        source_std = _safe_std(np.asarray(action_stats["std"], dtype=np.float32), eps)
        action_mean[: min(6, source_mean.shape[0], action_dim)] = source_mean[: min(6, source_mean.shape[0], action_dim)]
        action_std[: min(6, source_std.shape[0], action_dim)] = source_std[: min(6, source_std.shape[0], action_dim)]

    if state_mean.shape != (state_dim,) or state_std.shape != (state_dim,):
        raise ValueError(
            f"State normalization stats must be [{state_dim}], got "
            f"mean={state_mean.shape}, std={state_std.shape}."
        )
    if action_dim < 7:
        raise ValueError(f"Bridge policy actions must have at least 7 dims, got {action_dim}.")

    action_mask = np.zeros((action_dim,), dtype=bool)
    action_mask[:6] = True
    return state_mean, state_std, action_mean, action_std, action_mask


def normalize_policy_state_action(
    states: torch.Tensor,
    actions: torch.Tensor,
    stats: Dict[str, Any],
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize complete Bridge trajectories before policy window slicing.

    State is z-scored in all 7 dimensions. Action dimensions 0:6 are
    z-scored; binary gripper and optional terminate dimensions are left
    unchanged so padded actions can remain exact zeros after slicing.
    """
    if states.ndim != 2:
        raise ValueError(f"states must be [T,D_state], got {tuple(states.shape)}.")
    if actions.ndim != 2:
        raise ValueError(f"actions must be [T,D_action], got {tuple(actions.shape)}.")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}.")
    state_mean, state_std, action_mean, action_std, action_mask = _policy_norm_arrays(
        stats,
        state_dim=int(states.shape[-1]),
        action_dim=int(actions.shape[-1]),
        eps=eps,
    )
    device = states.device
    norm_states = (states.float() - torch.as_tensor(state_mean, device=device)) / torch.as_tensor(
        state_std, device=device
    )
    norm_actions = actions.float().clone()
    action_mask_t = torch.as_tensor(action_mask, device=actions.device, dtype=torch.bool)
    action_mean_t = torch.as_tensor(action_mean, device=actions.device)
    action_std_t = torch.as_tensor(action_std, device=actions.device)
    norm_actions[:, action_mask_t] = (
        norm_actions[:, action_mask_t] - action_mean_t[action_mask_t]
    ) / action_std_t[action_mask_t]
    return norm_states, norm_actions


def load_policy_normalization_stats(
    data_dir: str = DEFAULT_BRIDGE_TFDS_DIR,
    normalization_stats_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the Bridge normalization stats used by ``PolicyWindowDataset``."""
    stats_path = normalization_stats_path or _default_policy_norm_stats_path(data_dir)
    if stats_path is None:
        raise FileNotFoundError(
            "No Bridge normalization stats file was found. Pass normalization_stats_path explicitly."
        )
    return _load_json_file(str(stats_path))


def denormalize_policy_actions(
    actions: torch.Tensor,
    stats: Dict[str, Any],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map normalized Bridge actions back to the original action space.

    Only action dimensions ``0:6`` are de-normalized. The binary gripper
    dimension and optional terminate flag are passed through unchanged, matching
    ``normalize_policy_state_action``. Padded zero actions should be restored by
    applying the padding mask after this function if desired.
    """
    if actions.ndim < 2:
        raise ValueError(f"actions must end with action dim, got {tuple(actions.shape)}.")
    action_dim = int(actions.shape[-1])
    dummy_state_dim = 7
    _, _, action_mean, action_std, action_mask = _policy_norm_arrays(
        stats,
        state_dim=dummy_state_dim,
        action_dim=action_dim,
        eps=eps,
    )
    out = actions.float().clone()
    action_mask_t = torch.as_tensor(action_mask, device=out.device, dtype=torch.bool)
    action_mean_t = torch.as_tensor(action_mean, device=out.device, dtype=out.dtype)
    action_std_t = torch.as_tensor(action_std, device=out.device, dtype=out.dtype)
    out[..., action_mask_t] = out[..., action_mask_t] * action_std_t[action_mask_t] + action_mean_t[action_mask_t]
    return out


def format_semantic_queries(
    queries: Any,
    num_slots: int,
    query_dim: int,
    mask: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad/truncate object-noun embeddings to ``[num_slots, query_dim]``.

    ``queries`` can be ``[D]`` for one instruction/query, ``[M, D]`` for object
    noun queries, or already ``[K, D]``. If no mask is provided, active slots are
    inferred from non-zero query rows.
    """

    arr = np.asarray(queries, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"semantic queries must be [D] or [M,D], got shape {arr.shape}.")
    if arr.shape[-1] != query_dim:
        raise ValueError(f"Expected query_dim={query_dim}, got {arr.shape[-1]}.")

    out = np.zeros((num_slots, query_dim), dtype=np.float32)
    count = min(num_slots, arr.shape[0])
    out[:count] = arr[:count]

    if mask is None:
        active = np.linalg.norm(out, axis=-1) > 0
    else:
        mask_arr = np.asarray(mask).astype(bool).reshape(-1)
        active = np.zeros((num_slots,), dtype=bool)
        active[: min(num_slots, mask_arr.shape[0])] = mask_arr[:num_slots]

    return torch.from_numpy(out), torch.from_numpy(active.astype(np.float32))


class BridgeClipDataset(IterableDataset):
    """Stream continuous clips from Bridge TFDS/RLDS.

    By default this reads the current server's shared Bridge TFDS directory.
    Semantic queries are expected to be object-noun embeddings in the dataset.
    The exact field names are configurable because different preprocessing
    passes may store object nouns under different keys.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_BRIDGE_TFDS_DIR,
        split: str = "train",
        clip_length: int = 10,
        stride: int = 1,
        num_slots: int = 5,
        query_dim: int = 512,
        image_path: PathSpec = ("steps", "observation", "image"),
        semantic_query_paths: Sequence[PathSpec] = (
            ("steps", "observation", "name_embedding"),
            ("steps", "observation", "object_noun_embeddings"),
            ("steps", "observation", "instruction_object_embeddings"),
            ("name_embedding",),
            ("object_noun_embeddings",),
            ("instruction_object_embeddings",),
        ),
        semantic_mask_paths: Sequence[PathSpec] = (
            ("steps", "observation", "contrastive_loss_mask"),
            ("contrastive_loss_mask",),
            ("object_noun_mask",),
        ),
        instruction_paths: Sequence[PathSpec] = (
            ("steps", "observation", "natural_language_instruction"),
            ("natural_language_instruction",),
        ),
        instruction_embedding_paths: Sequence[PathSpec] = (
            ("steps", "observation", "natural_language_embedding"),
            ("natural_language_embedding",),
        ),
        use_flow: bool = False,
        flow_paths: Sequence[PathSpec] = (
            ("steps", "observation", "flow"),
            ("steps", "observation", "flow_target"),
            ("steps", "observation", "flow_image"),
            ("flow",),
            ("flow_target",),
        ),
        flow_temporal_mode: str = "pair",
        allow_instruction_fallback: bool = True,
        windows_per_episode: Optional[int] = None,
        shuffle_start: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if clip_length <= 0:
            raise ValueError("clip_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        self.data_dir = str(data_dir)
        self.split = split
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.num_slots = int(num_slots)
        self.query_dim = int(query_dim)
        self.image_path = tuple(image_path)
        self.semantic_query_paths = tuple(tuple(p) for p in semantic_query_paths)
        self.semantic_mask_paths = tuple(tuple(p) for p in semantic_mask_paths)
        self.instruction_paths = tuple(tuple(p) for p in instruction_paths)
        self.instruction_embedding_paths = tuple(tuple(p) for p in instruction_embedding_paths)
        if flow_temporal_mode not in {"pair", "frame"}:
            raise ValueError("flow_temporal_mode must be 'pair' or 'frame'.")
        self.use_flow = bool(use_flow)
        self.flow_paths = tuple(tuple(p) for p in flow_paths)
        self.flow_temporal_mode = flow_temporal_mode
        self.allow_instruction_fallback = bool(allow_instruction_fallback)
        self.windows_per_episode = windows_per_episode
        self.shuffle_start = bool(shuffle_start)
        self.seed = int(seed)

    def _iter_episode_dicts(self) -> Iterator[Dict[str, Any]]:
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError("BridgeClipDataset requires tensorflow_datasets to read RLDS TFRecords.") from exc
        except RuntimeError:
            import tensorflow_datasets as tfds

        data_path = Path(self.data_dir)
        if not (data_path / "dataset_info.json").exists():
            raise FileNotFoundError(
                f"Expected a TFDS builder directory with dataset_info.json, got {self.data_dir}."
            )

        builder = tfds.builder_from_directory(str(data_path))
        dataset = builder.as_dataset(split=self.split, shuffle_files=self.shuffle_start)
        for episode in tfds.as_numpy(dataset):
            yield episode

    def _window_starts(self, num_steps: int, rng: np.random.Generator) -> Sequence[int]:
        span = (self.clip_length - 1) * self.stride + 1
        if num_steps < span:
            return []
        starts = np.arange(0, num_steps - span + 1)
        if self.shuffle_start:
            rng.shuffle(starts)
        if self.windows_per_episode is not None:
            starts = starts[: self.windows_per_episode]
        return starts.tolist()

    def _semantic_from_episode(self, episode: Dict[str, Any], start: int) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = _try_get_nested(episode, self.semantic_query_paths)
        mask = _try_get_nested(episode, self.semantic_mask_paths)
        if queries is not None:
            queries = _select_time(queries, start)
            if mask is not None:
                mask = _select_time(mask, start)
            return format_semantic_queries(queries, self.num_slots, self.query_dim, mask)

        if not self.allow_instruction_fallback:
            raise KeyError(
                "No object-noun semantic query field found. Configure semantic_query_paths "
                "or enable allow_instruction_fallback."
            )
        instruction_embedding = _try_get_nested(episode, self.instruction_embedding_paths)
        if instruction_embedding is None:
            raise KeyError("No object-noun query or instruction embedding field found.")
        instruction_embedding = _select_time(instruction_embedding, start)
        return format_semantic_queries(instruction_embedding, self.num_slots, self.query_dim)

    def _flow_from_episode(self, episode: Dict[str, Any], indices: np.ndarray) -> Optional[torch.Tensor]:
        if not self.use_flow:
            return None
        flow = _try_get_nested(episode, self.flow_paths)
        if flow is None:
            raise KeyError("use_flow=True but no configured flow field was found in the episode.")

        flow = np.asarray(flow)
        if flow.ndim < 4:
            raise ValueError(f"Expected flow [T,H,W,C] or similar, got {flow.shape}.")
        flow_indices = indices[:-1] if self.flow_temporal_mode == "pair" else indices
        if flow.shape[0] <= int(flow_indices.max()):
            raise ValueError(
                f"Flow has {flow.shape[0]} time steps but requested index {int(flow_indices.max())}."
            )
        return torch.from_numpy(flow[flow_indices].copy())

    def __iter__(self) -> Iterator[BridgeClipSample]:
        worker = torch.utils.data.get_worker_info()
        seed = self.seed if worker is None else self.seed + worker.id
        rng = np.random.default_rng(seed)

        for episode in self._iter_episode_dicts():
            steps = _materialize_steps(episode)
            if steps is not None and _is_step_path(self.image_path):
                images = _get_step_sequence(steps, self.image_path)
                instruction_value = _try_get_step_sequence(steps, self.instruction_paths)
            else:
                images = _get_nested(episode, self.image_path)
                instruction_value = _try_get_nested(episode, self.instruction_paths)
            images = np.asarray(images)
            if images.ndim != 4:
                raise ValueError(f"Expected images [T,H,W,C], got {images.shape}.")

            instruction = _decode_text(instruction_value)
            for start in self._window_starts(images.shape[0], rng):
                indices = start + np.arange(self.clip_length) * self.stride
                clip = torch.from_numpy(images[indices].copy())
                if steps is not None:
                    semantic_episode = dict(episode)
                    semantic_episode["steps"] = {
                        "observation": {
                            name: _try_get_step_sequence(steps, (("steps", "observation", name),))
                            for name in (
                                "name_embedding",
                                "object_noun_embeddings",
                                "instruction_object_embeddings",
                                "contrastive_loss_mask",
                                "natural_language_embedding",
                            )
                        }
                    }
                else:
                    semantic_episode = episode
                semantic_queries, contrastive_loss_mask = self._semantic_from_episode(semantic_episode, start)
                flow_target = self._flow_from_episode(episode, indices)
                yield BridgeClipSample(
                    images=clip,
                    semantic_queries=semantic_queries,
                    contrastive_loss_mask=contrastive_loss_mask,
                    instruction=instruction,
                    start_index=start,
                    flow_target=flow_target,
                )


class BridgeTrajectoryDataset(IterableDataset):
    """Stream full trajectories from Bridge TFDS/RLDS.

    This reader intentionally does not create semantic embeddings. The expected
    training path is: full trajectory -> object noun parser -> SigLIP2 text
    embeddings -> V-JEPA2 rolling-pair slot wrapper.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_BRIDGE_TFDS_DIR,
        split: str = "train",
        image_path: PathSpec = ("steps", "observation", "image"),
        instruction_paths: Sequence[PathSpec] = (
            ("steps", "observation", "natural_language_instruction"),
            ("natural_language_instruction",),
        ),
        use_flow: bool = False,
        flow_paths: Sequence[PathSpec] = (
            ("steps", "observation", "flow"),
            ("steps", "observation", "flow_target"),
            ("steps", "observation", "flow_image"),
            ("flow",),
            ("flow_target",),
        ),
        shuffle_files: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = str(data_dir)
        self.split = split
        self.image_path = tuple(image_path)
        self.instruction_paths = tuple(tuple(p) for p in instruction_paths)
        self.use_flow = bool(use_flow)
        self.flow_paths = tuple(tuple(p) for p in flow_paths)
        self.shuffle_files = bool(shuffle_files)

    def _iter_episode_dicts(self) -> Iterator[Dict[str, Any]]:
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError("BridgeTrajectoryDataset requires tensorflow_datasets.") from exc
        except RuntimeError:
            import tensorflow_datasets as tfds

        data_path = Path(self.data_dir)
        if not (data_path / "dataset_info.json").exists():
            raise FileNotFoundError(
                f"Expected a TFDS builder directory with dataset_info.json, got {self.data_dir}."
            )

        builder = tfds.builder_from_directory(str(data_path))
        dataset = builder.as_dataset(split=self.split, shuffle_files=self.shuffle_files)
        for episode in tfds.as_numpy(dataset):
            yield episode

    def __iter__(self) -> Iterator[BridgeTrajectorySample]:
        for trajectory_index, episode in enumerate(self._iter_episode_dicts()):
            steps = _materialize_steps(episode)
            if steps is not None and _is_step_path(self.image_path):
                images = _get_step_sequence(steps, self.image_path)
                instruction_value = _try_get_step_sequence(steps, self.instruction_paths)
            else:
                images = _get_nested(episode, self.image_path)
                instruction_value = _try_get_nested(episode, self.instruction_paths)
            images = np.asarray(images)
            if images.ndim != 4:
                raise ValueError(f"Expected trajectory images [T,H,W,C], got {images.shape}.")
            instruction = _decode_text(instruction_value)

            flow_target = None
            if self.use_flow:
                if steps is not None:
                    flow = _try_get_step_sequence(steps, self.flow_paths)
                else:
                    flow = _try_get_nested(episode, self.flow_paths)
                if flow is None:
                    raise KeyError("use_flow=True but no configured flow field was found.")
                flow_target = torch.from_numpy(np.asarray(flow).copy())

            yield BridgeTrajectorySample(
                images=torch.from_numpy(images.copy()),
                instruction=instruction,
                trajectory_index=trajectory_index,
                flow_target=flow_target,
            )


def make_policy_window(
    *,
    images: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    timestep: int,
    history_frames: int = 4,
    future_steps: int = 4,
    instruction: str = "",
    trajectory_index: int = 0,
    shard_index: int = 0,
    terminal_action_is_padding: bool = False,
    trajectory_metadata: Optional[Dict[str, Any]] = None,
) -> PolicyWindowSample:
    """Build one padded policy window around ``timestep``.

    History uses ``t-H ... t-1``. Future frames/states use ``t ... t+F``
    so that adjacent pairs form F rolling video windows; future actions use
    ``t ... t+F-1``. Images/states replicate the nearest boundary value while
    padded actions are zero. All masks use ``True`` for padding.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be [T,H,W,C] or [T,C,H,W], got {tuple(images.shape)}.")
    if states.ndim != 2:
        raise ValueError(f"states must be [T,D_state], got {tuple(states.shape)}.")
    if actions.ndim != 2:
        raise ValueError(f"actions must be [T,D_action], got {tuple(actions.shape)}.")
    if history_frames <= 0:
        raise ValueError("history_frames must be positive.")
    if future_steps <= 0:
        raise ValueError("future_steps must be positive.")

    trajectory_length = int(images.shape[0])
    if trajectory_length <= 0:
        raise ValueError("trajectory must contain at least one step.")
    if states.shape[0] != trajectory_length or actions.shape[0] != trajectory_length:
        raise ValueError(
            "images, states, and actions must have equal trajectory lengths, got "
            f"{trajectory_length}, {states.shape[0]}, and {actions.shape[0]}."
        )
    timestep = int(timestep)
    if timestep < 0 or timestep >= trajectory_length:
        raise ValueError(
            f"timestep must be in [0,{trajectory_length - 1}], got {timestep}."
        )
    if images.device != states.device or images.device != actions.device:
        raise ValueError("images, states, and actions must be on the same device.")

    device = images.device
    hist_indices = torch.arange(
        timestep - history_frames,
        timestep,
        device=device,
        dtype=torch.long,
    )
    hist_padding = hist_indices < 0
    hist_safe = hist_indices.clamp(min=0, max=trajectory_length - 1)
    hist_images = images.index_select(0, hist_safe)
    hist_states = states.index_select(0, hist_safe)
    hist_actions = actions.index_select(0, hist_safe).clone()
    hist_actions[hist_padding] = 0

    # F rolling pairs need F+1 raw frames: (t,t+1), ..., (t+F-1,t+F).
    future_frame_indices = torch.arange(
        timestep,
        timestep + future_steps + 1,
        device=device,
        dtype=torch.long,
    )
    future_frame_padding = future_frame_indices >= trajectory_length
    future_frame_safe = future_frame_indices.clamp(min=0, max=trajectory_length - 1)
    future_images = images.index_select(0, future_frame_safe)
    future_states = states.index_select(0, future_frame_safe)

    future_action_indices = torch.arange(
        timestep,
        timestep + future_steps,
        device=device,
        dtype=torch.long,
    )
    last_valid_action_index = trajectory_length - 2 if terminal_action_is_padding else trajectory_length - 1
    future_action_padding = future_action_indices > last_valid_action_index
    future_action_safe = future_action_indices.clamp(min=0, max=max(last_valid_action_index, 0))
    future_actions = actions.index_select(0, future_action_safe).clone()
    future_actions[future_action_padding] = 0

    # A rolling 2-frame V-JEPA2 token is padded if either source frame is padded.
    future_video_padding = future_frame_padding[:-1] | future_frame_padding[1:]
    return PolicyWindowSample(
        hist_images=hist_images,
        hist_states=hist_states,
        hist_actions=hist_actions,
        future_images=future_images,
        future_states=future_states,
        future_actions=future_actions,
        hist_image_padding_mask=hist_padding.clone(),
        hist_state_padding_mask=hist_padding.clone(),
        hist_action_padding_mask=hist_padding.clone(),
        future_frame_padding_mask=future_frame_padding,
        future_state_padding_mask=future_frame_padding.clone(),
        future_action_padding_mask=future_action_padding,
        future_video_padding_mask=future_video_padding,
        instruction=instruction,
        timestep=timestep,
        trajectory_index=int(trajectory_index),
        shard_index=int(shard_index),
        trajectory_metadata=dict(trajectory_metadata or {}),
    )


class PolicyWindowDataset(IterableDataset):
    """Read complete TFRecord shards and emit one policy window for every frame.

    A worker receives whole shards, so every trajectory in a selected data file
    stays together. By default all shards in ``split`` and all timesteps in each
    trajectory are used.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_BRIDGE_TFDS_DIR,
        split: str = "train",
        history_frames: int = 4,
        future_steps: int = 4,
        include_terminate_action: bool = False,
        state_delta_action: bool = True,
        normalize_state_action: bool = True,
        normalization_stats_path: Optional[str] = None,
        normalization_eps: float = 1e-6,
        shard_indices: Optional[Sequence[int]] = None,
        shuffle_shards: bool = False,
        seed: int = 0,
        start_cursor: Optional[Dict[str, Any]] = None,
        metadata_manifest_path: Optional[str] = None,
        require_metadata_manifest: bool = False,
        episode_index_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if history_frames <= 0:
            raise ValueError("history_frames must be positive.")
        if future_steps <= 0:
            raise ValueError("future_steps must be positive.")
        self.data_dir = str(data_dir)
        self.split = str(split)
        self.history_frames = int(history_frames)
        self.future_steps = int(future_steps)
        self.include_terminate_action = bool(include_terminate_action)
        self.state_delta_action = bool(state_delta_action)
        self.normalize_state_action = bool(normalize_state_action)
        self.normalization_stats_path = normalization_stats_path
        self.normalization_eps = float(normalization_eps)
        if self.normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive.")
        self._normalization_stats: Optional[Dict[str, Any]] = None
        self.shard_indices = None if shard_indices is None else tuple(int(i) for i in shard_indices)
        if self.shard_indices is not None and any(i < 0 for i in self.shard_indices):
            raise ValueError("shard_indices must be non-negative.")
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.start_cursor = dict(start_cursor) if start_cursor is not None else None
        self.metadata_manifest_path = metadata_manifest_path
        self.require_metadata_manifest = bool(require_metadata_manifest)
        self._trajectory_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None
        self.episode_index_path = (
            resolve_path_string(str(episode_index_path)) if episode_index_path else None
        )
        if self.episode_index_path:
            self._episode_index, self._episode_index_shards = _load_policy_episode_index(
                self.episode_index_path,
                split=self.split,
            )
        else:
            self._episode_index = None
            self._episode_index_shards = None

    def _load_trajectory_metadata(self) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Load optional per-trajectory labels emitted by the subset exporter."""
        if self._trajectory_metadata is not None:
            return self._trajectory_metadata

        manifest_path = self.metadata_manifest_path
        if manifest_path is None:
            candidate = Path(self.data_dir).expanduser() / "export_manifest.jsonl"
            if candidate.is_file():
                manifest_path = str(candidate)
        if manifest_path is None:
            if self.require_metadata_manifest:
                raise FileNotFoundError(
                    "This dataset requires export_manifest.jsonl. Pass "
                    "metadata_manifest_path or place it in the dataset directory."
                )
            self._trajectory_metadata = {}
        else:
            self.metadata_manifest_path = str(Path(manifest_path).expanduser())
            self._trajectory_metadata = load_trajectory_metadata_manifest(
                self.metadata_manifest_path
            )
        return self._trajectory_metadata

    def _load_normalization_stats(self) -> Optional[Dict[str, Any]]:
        if not self.normalize_state_action:
            return None
        if self._normalization_stats is None:
            stats_path = self.normalization_stats_path or _default_policy_norm_stats_path(self.data_dir)
            if stats_path is None:
                raise FileNotFoundError(
                    "normalize_state_action=True but no normalization stats file was found. "
                    "Pass normalization_stats_path explicitly."
            )
            self.normalization_stats_path = str(stats_path)
            self._normalization_stats = _load_json_file(str(stats_path))
            source = self._normalization_stats.get("source", {})
            if isinstance(source, dict) and "state_delta_action" in source:
                stats_state_delta = bool(source["state_delta_action"])
                if stats_state_delta != self.state_delta_action:
                    raise ValueError(
                        "Normalization stats action mode does not match PolicyWindowDataset: "
                        f"stats source.state_delta_action={stats_state_delta}, "
                        f"dataset state_delta_action={self.state_delta_action}. "
                        "Use a raw_action stats file for --raw-action or a state_delta stats file for "
                        "--state-delta-action."
                    )
        return self._normalization_stats

    def _builder_and_tfds(self):
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError("PolicyWindowDataset requires tensorflow and tensorflow_datasets.") from exc
        except RuntimeError:
            import tensorflow_datasets as tfds

        data_path = Path(self.data_dir)
        if not (data_path / "dataset_info.json").exists():
            raise FileNotFoundError(
                f"Expected a TFDS builder directory with dataset_info.json, got {self.data_dir}."
            )
        return tfds.builder_from_directory(str(data_path)), tfds

    def _assigned_shards(self, num_shards: int) -> Sequence[int]:
        if self.shard_indices is None:
            shards = list(range(num_shards))
        else:
            shards = list(self.shard_indices)
            invalid = [index for index in shards if index >= num_shards]
            if invalid:
                raise ValueError(
                    f"Shard indices {invalid} exceed split {self.split!r} with {num_shards} shards."
                )

        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        if self.shuffle_shards:
            # Every worker must use the same permutation before disjoint slicing.
            rng = np.random.default_rng(self.seed)
            rng.shuffle(shards)
        return shards[worker_id::num_workers]

    def _iter_shard_episodes(self, builder, tfds, shard_index: int):
        # TFDS shard slicing selects one physical file, not an example range.
        split_spec = f"{self.split}[{shard_index}shard]"
        dataset = builder.as_dataset(split=split_spec, shuffle_files=False)
        yield from tfds.as_numpy(dataset)

    def _iter_tfrecord_policy_arrays(
        self,
        path: Path,
        selected_records: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, str]]:
        example_class = _example_message_class()
        if selected_records is None:
            raw_iter = enumerate(_iter_tfrecord_records(path))
        else:
            raw_iter = _iter_selected_tfrecord_records(path, selected_records)
        for record_index, raw in raw_iter:
            example = example_class()
            example.ParseFromString(raw)
            state_values = _feature_float_array(example, "steps/observation/state")
            if state_values is None:
                raise KeyError("steps/observation/state")
            states = _reshape_flat_feature(state_values, 7, "steps/observation/state").copy()
            raw_actions = _flatten_tfrecord_actions(
                example,
                length=int(states.shape[0]),
                include_terminate=self.include_terminate_action,
            ).copy()

            image_values = None
            image_key = ""
            for key in ("steps/observation/image", "steps/observation/image_0"):
                image_values = _feature_bytes_list(example, key)
                if image_values is not None:
                    image_key = key
                    break
            if image_values is None:
                raise KeyError("steps/observation/image or steps/observation/image_0")
            if len(image_values) != states.shape[0]:
                raise ValueError(
                    f"Feature {image_key!r} length mismatch: {len(image_values)} vs {states.shape[0]}."
                )
            images = _decode_jpeg_sequence(image_values, image_key)

            instruction_values = None
            for key in (
                "steps/observation/natural_language_instruction",
                "steps/language_instruction",
            ):
                instruction_values = _feature_bytes_list(example, key)
                if instruction_values is not None:
                    break
            instruction = _decode_text(instruction_values)
            yield int(record_index), images, states, raw_actions, instruction

    def _iter_tfds_policy_arrays(
        self,
        builder: Any,
        tfds: Any,
        shard_index: int,
        selected_record_indices: Optional[set[int]] = None,
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, str]]:
        for record_index, episode in enumerate(self._iter_shard_episodes(builder, tfds, shard_index)):
            if selected_record_indices is not None and record_index not in selected_record_indices:
                continue
            steps = _materialize_steps(episode)
            if steps is None:
                raise ValueError("Expected RLDS episode['steps'] to be an iterable step dataset.")

            image_array = _try_get_step_sequence(
                steps,
                (
                    ("steps", "observation", "image"),
                    ("steps", "observation", "image_0"),
                ),
            )
            if image_array is None:
                raise KeyError("steps/observation/image or steps/observation/image_0")
            states = np.asarray(
                _get_step_sequence(steps, ("steps", "observation", "state")),
                dtype=np.float32,
            ).copy()
            raw_actions = _flatten_bridge_actions(
                steps,
                include_terminate=self.include_terminate_action,
            ).copy()
            instruction_value = _try_get_step_sequence(
                steps,
                (
                    ("steps", "observation", "natural_language_instruction"),
                    ("steps", "language_instruction"),
                ),
            )
            yield (
                int(record_index),
                np.asarray(image_array).copy(),
                states,
                raw_actions,
                _decode_text(instruction_value),
            )

    def __iter__(self) -> Iterator[PolicyWindowSample]:
        normalization_stats = self._load_normalization_stats()
        trajectory_metadata = self._load_trajectory_metadata()
        tfrecord_shards = (
            _all_tfrecord_shards(self.data_dir)
            if self.split == "all"
            else _tfrecord_shards(self.data_dir, self.split)
        )
        use_tfrecord = bool(tfrecord_shards)
        if use_tfrecord:
            shard_paths = dict(tfrecord_shards)
            num_shards = len(tfrecord_shards)
            builder = None
            tfds = None
        else:
            if self.split == "all":
                raise ValueError(
                    "split='all' currently requires direct TFRecord shards in data_dir."
                )
            if trajectory_metadata:
                raise ValueError(
                    "Trajectory metadata lookup requires direct TFRecord shards in data_dir; "
                    "the TFDS fallback does not expose physical output record locations."
                )
            builder, tfds = self._builder_and_tfds()
            if self.split not in builder.info.splits:
                raise ValueError(
                    f"Unknown split {self.split!r}; available splits are {list(builder.info.splits)}."
                )
            num_shards = int(builder.info.splits[self.split].num_shards)
            shard_paths = {}

        if self._episode_index is not None:
            missing_shards = sorted(set(self._episode_index) - set(range(num_shards)))
            if missing_shards:
                raise ValueError(
                    f"Episode index references shards outside split {self.split!r}: {missing_shards}"
                )
            if use_tfrecord and self._episode_index_shards is not None:
                for shard_index, path in shard_paths.items():
                    identity = self._episode_index_shards.get(shard_index)
                    if identity is None:
                        raise ValueError(
                            f"Episode index has no identity metadata for {self.split} shard {shard_index}."
                        )
                    expected_name, expected_size, _ = identity
                    actual_size = path.stat().st_size
                    if path.name != expected_name or actual_size != expected_size:
                        raise ValueError(
                            "Episode index was built for a different TFRecord shard: "
                            f"shard={shard_index}, expected={expected_name}/{expected_size}, "
                            f"actual={path.name}/{actual_size}."
                        )

        assigned_shards = list(self._assigned_shards(num_shards))
        start_shard_position = 0
        start_trajectory_index = 0
        start_timestep = -1
        if self.start_cursor is not None:
            worker = torch.utils.data.get_worker_info()
            if worker is not None and worker.num_workers > 1:
                raise ValueError("PolicyWindowDataset start_cursor currently requires num_workers=0.")
            cursor_shard = int(self.start_cursor["shard_index"])
            if cursor_shard not in assigned_shards:
                raise ValueError(
                    f"start_cursor shard_index={cursor_shard} is not assigned to this worker."
                )
            start_shard_position = assigned_shards.index(cursor_shard)
            start_trajectory_index = int(self.start_cursor["trajectory_index"])
            start_timestep = int(self.start_cursor["timestep"])

        for shard_position, shard_index in enumerate(assigned_shards):
            if shard_position < start_shard_position:
                continue
            selected_records = (
                None if self._episode_index is None else self._episode_index.get(shard_index, ())
            )
            if selected_records == ():
                continue
            if use_tfrecord:
                array_iter = self._iter_tfrecord_policy_arrays(
                    shard_paths[shard_index],
                    selected_records=selected_records,
                )
            else:
                selected_record_indices = (
                    None
                    if selected_records is None
                    else {record_index for record_index, _, _ in selected_records}
                )
                array_iter = self._iter_tfds_policy_arrays(
                    builder,
                    tfds,
                    shard_index,
                    selected_record_indices=selected_record_indices,
                )
            for trajectory_index, images_np, states_np, raw_actions_np, instruction in array_iter:
                if shard_position == start_shard_position and trajectory_index < start_trajectory_index:
                    continue

                metadata: Dict[str, Any] = {}
                if trajectory_metadata:
                    location = (shard_paths[shard_index].name, trajectory_index)
                    row = trajectory_metadata.get(location)
                    if row is None and self.require_metadata_manifest:
                        raise KeyError(
                            "No export metadata for TFRecord trajectory "
                            f"{location[0]} record {location[1]}."
                        )
                    metadata = dict(row or {})

                images = torch.from_numpy(np.asarray(images_np).copy())
                actions_np = make_policy_actions(
                    states_np,
                    raw_actions_np,
                    state_delta_action=self.state_delta_action,
                )
                states = torch.from_numpy(states_np)
                actions = torch.from_numpy(actions_np)
                if normalization_stats is not None:
                    states, actions = normalize_policy_state_action(
                        states,
                        actions,
                        normalization_stats,
                        eps=self.normalization_eps,
                    )

                timestep_start = 0
                if shard_position == start_shard_position and trajectory_index == start_trajectory_index:
                    timestep_start = start_timestep + 1
                for timestep in range(timestep_start, images.shape[0]):
                    yield make_policy_window(
                        images=images,
                        states=states,
                        actions=actions,
                        timestep=timestep,
                        history_frames=self.history_frames,
                        future_steps=self.future_steps,
                        instruction=instruction,
                        trajectory_index=trajectory_index,
                        shard_index=shard_index,
                        terminal_action_is_padding=self.state_delta_action,
                        trajectory_metadata=metadata,
                    )


def policy_window_collate(samples: Sequence[PolicyWindowSample]) -> Dict[str, Any]:
    """Stack fixed-shape policy windows into a PyTorch batch."""
    if not samples:
        raise ValueError("Cannot collate an empty policy window batch.")
    tensor_fields = (
        "hist_images",
        "hist_states",
        "hist_actions",
        "future_images",
        "future_states",
        "future_actions",
        "hist_image_padding_mask",
        "hist_state_padding_mask",
        "hist_action_padding_mask",
        "future_frame_padding_mask",
        "future_state_padding_mask",
        "future_action_padding_mask",
        "future_video_padding_mask",
    )
    batch = {
        field: torch.stack([getattr(sample, field) for sample in samples], dim=0)
        for field in tensor_fields
    }
    batch.update(
        instruction=[sample.instruction for sample in samples],
        timestep=torch.tensor([sample.timestep for sample in samples], dtype=torch.long),
        trajectory_index=torch.tensor(
            [sample.trajectory_index for sample in samples], dtype=torch.long
        ),
        shard_index=torch.tensor([sample.shard_index for sample in samples], dtype=torch.long),
        trajectory_metadata=[dict(sample.trajectory_metadata) for sample in samples],
    )
    return batch


def select_history_frames(
    images: torch.Tensor,
    history_frames: int = 4,
    start_index: int = 0,
) -> torch.Tensor:
    """Select a contiguous history clip from one trajectory without pair slicing.

    Args:
        images: trajectory frames ``[T,H,W,3]`` or ``[T,3,H,W]``.
        history_frames: number of contiguous history frames to return.
        start_index: first frame index.

    Returns:
        ``[history_frames,H,W,3]`` or ``[history_frames,3,H,W]`` matching input layout.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be one trajectory [T,H,W,3] or [T,3,H,W], got {tuple(images.shape)}.")
    if history_frames <= 0:
        raise ValueError("history_frames must be positive.")
    start = int(start_index)
    end = start + int(history_frames)
    if start < 0:
        raise ValueError("start_index must be non-negative.")
    if end > images.shape[0]:
        raise ValueError(
            f"Need {history_frames} history frames from start_index={start}, "
            f"but trajectory only has {images.shape[0]} frames."
        )
    return images[start:end]


def bridge_clip_collate(samples: Sequence[BridgeClipSample]) -> Dict[str, Any]:
    """Collate ``BridgeClipSample`` objects into a training batch."""
    batch = {
        "images": torch.stack([s.images for s in samples], dim=0),
        "semantic_queries": torch.stack([s.semantic_queries for s in samples], dim=0),
        "contrastive_loss_mask": torch.stack([s.contrastive_loss_mask for s in samples], dim=0),
        "instruction": [s.instruction for s in samples],
        "start_index": torch.tensor([s.start_index for s in samples], dtype=torch.long),
    }
    has_flow = [s.flow_target is not None for s in samples]
    if any(has_flow):
        if not all(has_flow):
            raise ValueError("Cannot collate a batch with mixed flow/no-flow samples.")
        batch["flow_target"] = torch.stack([s.flow_target for s in samples], dim=0)
    return batch
