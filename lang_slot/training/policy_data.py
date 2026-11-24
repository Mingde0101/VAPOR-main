"""Shared dataset/DataLoader construction for policy train and evaluation.

All policy entry points should build Bridge windows through this module.  This
keeps task-specific subset readers, action semantics, temporal horizons, and
worker settings identical between training, validation, and offline forward
passes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from lang_slot.data import (
    PolicyWindowDataset,
    SimplerPickPlacePolicyWindowDataset,
    policy_episode_index_path,
    policy_window_collate,
)

from .policy_utils import get_nested, parse_shard_indices


DATASET_CLASSES = {
    "bridge": PolicyWindowDataset,
    "simpler_pick_place": SimplerPickPlacePolicyWindowDataset,
}


def build_policy_window_dataset(
    config: Dict[str, Any],
    *,
    split: Optional[str] = None,
    shard_indices: Optional[Sequence[int]] = None,
    shuffle_shards: Optional[bool] = None,
    start_cursor: Optional[Dict[str, Any]] = None,
) -> PolicyWindowDataset:
    """Build the configured policy-window dataset with optional mode overrides."""
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "bridge"))
    dataset_class = DATASET_CLASSES.get(dataset_name)
    if dataset_class is None:
        raise ValueError(
            f"Unknown data.dataset={dataset_name!r}; expected one of {sorted(DATASET_CLASSES)}."
        )

    configured_shards = parse_shard_indices(data_cfg.get("shard_indices", None))
    selected_shards = configured_shards if shard_indices is None else tuple(int(v) for v in shard_indices)
    selected_split = str(split if split is not None else data_cfg.get("split", "train"))
    kwargs: Dict[str, Any] = {
        "data_dir": data_cfg.get("path"),
        "split": selected_split,
        "history_frames": int(data_cfg.get("history_frames", 4)),
        "future_steps": int(data_cfg.get("future_steps", 4)),
        "include_terminate_action": bool(data_cfg.get("include_terminate_action", False)),
        # Legacy configs predate state-delta relabeling and therefore mean raw.
        "state_delta_action": bool(data_cfg.get("state_delta_action", False)),
        "normalize_state_action": bool(data_cfg.get("normalize_state_action", True)),
        "normalization_stats_path": data_cfg.get("normalization_stats_path", None),
        "normalization_eps": float(data_cfg.get("normalization_eps", 1e-6)),
        "shard_indices": selected_shards,
        "shuffle_shards": (
            bool(data_cfg.get("shuffle_shards", True))
            if shuffle_shards is None
            else bool(shuffle_shards)
        ),
        "seed": int(get_nested(config, "experiment.seed", 0)),
        "start_cursor": start_cursor,
    }
    if data_cfg.get("metadata_manifest_path") is not None:
        kwargs["metadata_manifest_path"] = data_cfg["metadata_manifest_path"]
    if dataset_name == "bridge":
        kwargs["episode_index_path"] = policy_episode_index_path(data_cfg, selected_split)
    return dataset_class(**kwargs)


def build_policy_dataloader(
    config: Dict[str, Any],
    *,
    split: Optional[str] = None,
    shard_indices: Optional[Sequence[int]] = None,
    shuffle_shards: Optional[bool] = None,
    start_cursor: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader using the same policy-window contract in every mode."""
    data_cfg = config.get("data", {})
    dataset = build_policy_window_dataset(
        config,
        split=split,
        shard_indices=shard_indices,
        shuffle_shards=shuffle_shards,
        start_cursor=start_cursor,
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1) if batch_size is None else batch_size),
        num_workers=int(data_cfg.get("num_workers", 0) if num_workers is None else num_workers),
        collate_fn=policy_window_collate,
        pin_memory=torch.cuda.is_available(),
    )
