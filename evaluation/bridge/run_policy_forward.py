#!/usr/bin/env python
"""Run a trained flow-matching policy over Bridge policy windows.

The script reuses the training data/model pipeline, samples a 4-action chunk for
each dataset timestep, and saves only the first predicted action ``a_t``. Results
are keyed by ``(split, shard_index, trajectory_index, timestep)`` so predictions
can be aligned back to the source RLDS trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from lang_slot.data import denormalize_policy_actions, load_policy_normalization_stats
from lang_slot.paths import check_policy_files, resources
from lang_slot.training.policy_data import build_policy_dataloader
from lang_slot.training.policy_utils import (
    attach_text_features,
    build_policy_from_config,
    build_text_embedder,
    get_nested,
    latest_checkpoint,
    load_checkpoint,
    load_yaml_config,
    move_policy_batch_to_device,
    parse_shard_indices,
    resolve_device,
    set_nested,
    set_seed,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/bridge_val.yaml"),
        help="Training config used to build the policy.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(resources()["POLICY_CHECKPOINT"]),
        help="Checkpoint to load. Defaults to POLICY_CHECKPOINT.",
    )
    parser.add_argument("--split", default="", help="Dataset split to run, e.g. train or test.")
    parser.add_argument("--data-dir", default="", help="Override data.path.")
    parser.add_argument("--output-dir", default="", help="Prediction output directory override.")
    parser.add_argument("--device", default="", help="Override device. Use cuda/auto/cpu.")
    parser.add_argument("--precision", default="", choices=["", "fp32", "fp16", "bf16"], help="CUDA autocast precision.")
    parser.add_argument("--batch-size", type=int, default=0, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Use 0 for exact order.")
    parser.add_argument("--shard-indices", default="", help="Comma-separated physical shard ids to run.")
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="Stop after this many complete trajectories in dataset order. 0 means all.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Stop after this many windows. 0 means all.")
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after this many batches. 0 means all.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Rows per saved NPZ chunk.")
    parser.add_argument("--sample-steps", type=int, default=0, help="Flow sampling steps. Defaults to config.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed. 0 uses experiment.seed.")
    parser.add_argument("--save-normalized", action="store_true", help="Also save normalized predictions/targets.")
    parser.add_argument("--save-instructions", action="store_true", help="Write per-sample instructions to JSONL.")
    parser.add_argument("--no-plots", action="store_true", help="Disable per-trajectory comparison plots.")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], help="Plot file format.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output dir.")
    return parser


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.split:
        set_nested(config, "data.split", args.split)
    if args.data_dir:
        set_nested(config, "data.path", args.data_dir)
    if args.device:
        set_nested(config, "training.device", args.device)
    if args.precision:
        set_nested(config, "training.precision", args.precision)
    if args.batch_size > 0:
        set_nested(config, "data.batch_size", args.batch_size)
    if args.num_workers >= 0:
        set_nested(config, "data.num_workers", args.num_workers)
    if args.shard_indices:
        set_nested(config, "data.shard_indices", list(parse_shard_indices(args.shard_indices) or ()))
    set_nested(config, "data.shuffle_shards", False)
    return config


def make_dataloader(config: Dict[str, Any]) -> DataLoader:
    return build_policy_dataloader(config, shuffle_shards=False)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown precision {precision!r}; expected fp32/fp16/bf16.")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def slice_batch(batch: Dict[str, Any], indices: List[int]) -> Dict[str, Any]:
    if not indices:
        raise ValueError("indices must be non-empty.")
    out: Dict[str, Any] = {}
    index_tensor: Optional[torch.Tensor] = None
    batch_size = int(batch["future_actions"].shape[0])
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            if index_tensor is None or index_tensor.device != value.device:
                index_tensor = torch.as_tensor(indices, device=value.device, dtype=torch.long)
            out[key] = value.index_select(0, index_tensor)
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = [value[index] for index in indices]
        else:
            out[key] = value
    return out


def allowed_indices_for_trajectory_limit(
    batch: Dict[str, Any],
    *,
    seen_trajectories: set[tuple[int, int]],
    max_trajectories: int,
) -> tuple[List[int], bool]:
    """Return row indices that belong to the first ``max_trajectories`` trajectories.

    The dataset emits windows ordered by ``(shard_index, trajectory_index,
    timestep)`` when ``num_workers=0`` and shard shuffling is disabled. Once a
    new trajectory exceeds the limit, later rows are outside the requested
    prefix and can be skipped.
    """
    if max_trajectories <= 0:
        return list(range(int(batch["future_actions"].shape[0]))), False

    shard_indices = batch["shard_index"].detach().cpu().tolist()
    trajectory_indices = batch["trajectory_index"].detach().cpu().tolist()
    allowed: List[int] = []
    reached_limit = False
    for row_index, (shard_index, trajectory_index) in enumerate(zip(shard_indices, trajectory_indices)):
        key = (int(shard_index), int(trajectory_index))
        if key not in seen_trajectories:
            if len(seen_trajectories) >= max_trajectories:
                reached_limit = True
                break
            seen_trajectories.add(key)
        allowed.append(row_index)
    return allowed, reached_limit


def denormalize_actions(
    actions: torch.Tensor,
    *,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = denormalize_policy_actions(
        actions,
        stats,
        eps=float(get_nested(config, "data.normalization_eps", 1e-6)),
    )
    if padding_mask is not None:
        out = out.masked_fill(padding_mask.to(device=out.device, dtype=torch.bool).unsqueeze(-1), 0.0)
    return out


def default_output_dir(config: Dict[str, Any], checkpoint_path: str) -> Path:
    split = str(get_nested(config, "data.split", "train"))
    ckpt_name = Path(checkpoint_path).stem
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return resources()["POLICY_OUTPUT_DIR"] / "policy_forward" / f"{split}_{ckpt_name}_{stamp}"


def flush_chunk(
    *,
    output_dir: Path,
    chunk_index: int,
    rows: List[Dict[str, np.ndarray]],
    save_normalized: bool,
    compress: bool = True,
) -> Path:
    if not rows:
        raise ValueError("Cannot flush an empty chunk.")
    arrays: Dict[str, np.ndarray] = {
        "sample_index": np.concatenate([row["sample_index"] for row in rows], axis=0),
        "shard_index": np.concatenate([row["shard_index"] for row in rows], axis=0),
        "trajectory_index": np.concatenate([row["trajectory_index"] for row in rows], axis=0),
        "timestep": np.concatenate([row["timestep"] for row in rows], axis=0),
        "valid_action": np.concatenate([row["valid_action"] for row in rows], axis=0),
        "pred_action": np.concatenate([row["pred_action"] for row in rows], axis=0),
        "target_action": np.concatenate([row["target_action"] for row in rows], axis=0),
        "abs_error": np.concatenate([row["abs_error"] for row in rows], axis=0),
    }
    if save_normalized:
        arrays.update(
            pred_action_normalized=np.concatenate([row["pred_action_normalized"] for row in rows], axis=0),
            target_action_normalized=np.concatenate([row["target_action_normalized"] for row in rows], axis=0),
        )
    path = output_dir / "chunks" / f"predictions_{chunk_index:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if compress else np.savez
    saver(path, **arrays)
    return path


def append_instructions(path: Path, batch: Dict[str, Any], sample_indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for local_index, sample_index in enumerate(sample_indices.tolist()):
            row = {
                "sample_index": int(sample_index),
                "shard_index": int(batch["shard_index"][local_index].detach().cpu()),
                "trajectory_index": int(batch["trajectory_index"][local_index].detach().cpu()),
                "timestep": int(batch["timestep"][local_index].detach().cpu()),
                "instruction": batch["instruction"][local_index],
                "object_nouns": batch.get("object_nouns", [None] * len(batch["instruction"]))[local_index],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def require_plot_dependencies() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Per-trajectory plots require matplotlib. Install it in the active "
            "environment, or pass --no-plots to only save NPZ predictions."
        ) from exc


def make_empty_trajectory_buffer(key: tuple[int, int]) -> Dict[str, Any]:
    shard_index, trajectory_index = key
    return {
        "shard_index": int(shard_index),
        "trajectory_index": int(trajectory_index),
        "timestep": [],
        "pred_action": [],
        "target_action": [],
        "valid_action": [],
    }


def add_row_to_trajectory_buffer(
    buffer: Dict[str, Any],
    *,
    timestep: int,
    pred_action: np.ndarray,
    target_action: np.ndarray,
    valid_action: bool,
) -> None:
    buffer["timestep"].append(int(timestep))
    buffer["pred_action"].append(np.asarray(pred_action, dtype=np.float32))
    buffer["target_action"].append(np.asarray(target_action, dtype=np.float32))
    buffer["valid_action"].append(bool(valid_action))


def plot_trajectory_buffer(
    *,
    output_dir: Path,
    buffer: Dict[str, Any],
    plot_format: str,
) -> Optional[Dict[str, Any]]:
    if not buffer["timestep"]:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = np.asarray(buffer["timestep"], dtype=np.int64)
    pred = np.stack(buffer["pred_action"], axis=0)
    target = np.stack(buffer["target_action"], axis=0)
    valid = np.asarray(buffer["valid_action"], dtype=bool)
    order = np.argsort(timesteps)
    timesteps = timesteps[order]
    pred = pred[order]
    target = target[order]
    valid = valid[order]

    pred_valid = pred[valid]
    target_valid = target[valid]
    time_valid = timesteps[valid]
    if pred_valid.size == 0:
        return None

    shard_index = int(buffer["shard_index"])
    trajectory_index = int(buffer["trajectory_index"])
    stem = f"shard_{shard_index:05d}_traj_{trajectory_index:06d}"
    plot_dir = output_dir / "plots" / f"shard_{shard_index:05d}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pred_xyz = np.cumsum(pred_valid[:, :3], axis=0)
    target_xyz = np.cumsum(target_valid[:, :3], axis=0)
    spatial_path = plot_dir / f"{stem}_spatial.{plot_format}"
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2], label="target", linewidth=2.0)
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="pred", linewidth=2.0)
    ax.scatter(target_xyz[0, 0], target_xyz[0, 1], target_xyz[0, 2], marker="o", s=35, label="target start")
    ax.scatter(pred_xyz[0, 0], pred_xyz[0, 1], pred_xyz[0, 2], marker="^", s=35, label="pred start")
    ax.set_title(f"Action Translation Trajectory shard={shard_index} traj={trajectory_index}")
    ax.set_xlabel("cumsum world_vector x")
    ax.set_ylabel("cumsum world_vector y")
    ax.set_zlabel("cumsum world_vector z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(spatial_path, dpi=160)
    plt.close(fig)

    dim_path = plot_dir / f"{stem}_action_dims.{plot_format}"
    labels = [
        "world_x",
        "world_y",
        "world_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "open_gripper",
    ]
    fig, axes = plt.subplots(7, 1, figsize=(11.5, 13.0), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(time_valid, target_valid[:, dim], label="target", linewidth=1.5)
        ax.plot(time_valid, pred_valid[:, dim], label="pred", linewidth=1.3, alpha=0.9)
        ax.set_ylabel(labels[dim])
        ax.grid(True, linewidth=0.4, alpha=0.35)
        if dim == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"Action Dimensions shard={shard_index} traj={trajectory_index}", y=0.995)
    fig.tight_layout()
    fig.savefig(dim_path, dpi=160)
    plt.close(fig)

    return {
        "shard_index": shard_index,
        "trajectory_index": trajectory_index,
        "num_steps": int(valid.sum()),
        "spatial_plot": str(spatial_path.relative_to(output_dir)),
        "action_dims_plot": str(dim_path.relative_to(output_dir)),
    }


def update_trajectory_plots(
    *,
    output_dir: Path,
    plot_format: str,
    current_buffer: Optional[Dict[str, Any]],
    plot_index: List[Dict[str, Any]],
    shard_indices: np.ndarray,
    trajectory_indices: np.ndarray,
    timesteps: np.ndarray,
    pred_actions: np.ndarray,
    target_actions: np.ndarray,
    valid_actions: np.ndarray,
) -> Optional[Dict[str, Any]]:
    for row_index in range(pred_actions.shape[0]):
        key = (int(shard_indices[row_index]), int(trajectory_indices[row_index]))
        if current_buffer is None:
            current_buffer = make_empty_trajectory_buffer(key)
        current_key = (int(current_buffer["shard_index"]), int(current_buffer["trajectory_index"]))
        if key != current_key:
            entry = plot_trajectory_buffer(output_dir=output_dir, buffer=current_buffer, plot_format=plot_format)
            if entry is not None:
                plot_index.append(entry)
            current_buffer = make_empty_trajectory_buffer(key)
        add_row_to_trajectory_buffer(
            current_buffer,
            timestep=int(timesteps[row_index]),
            pred_action=pred_actions[row_index],
            target_action=target_actions[row_index],
            valid_action=bool(valid_actions[row_index]),
        )
    return current_buffer


def main(argv=None, *, config=None, build_policy=build_policy_from_config, restore=load_checkpoint) -> None:
    args = build_argparser().parse_args(argv)
    config = apply_overrides(config if config is not None else load_yaml_config(args.config), args)
    device = resolve_device(get_nested(config, "training.device", "cuda"))
    precision = str(get_nested(config, "training.precision", "bf16"))

    seed = int(args.seed or get_nested(config, "experiment.seed", 0))
    set_seed(seed)

    checkpoint_path = args.checkpoint or latest_checkpoint(str(get_nested(config, "experiment.output_dir")))
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint provided and no latest.pt found under experiment.output_dir.")
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())

    check_policy_files(config, checkpoint=checkpoint_path, split=config["data"]["split"])
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(config, checkpoint_path)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite or choose another path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots and int(get_nested(config, "data.num_workers", 0)) != 0:
        raise ValueError("Per-trajectory plots require --num-workers 0 so trajectory order is deterministic.")
    if not args.no_plots:
        require_plot_dependencies()

    print(f"[setup] device={device} precision={precision} split={get_nested(config, 'data.split')}", flush=True)
    print(f"[setup] checkpoint={checkpoint_path}", flush=True)
    print(f"[setup] output_dir={output_dir}", flush=True)

    policy = build_policy(config, device)
    text_embedder = build_text_embedder(config, device)
    checkpoint = restore(checkpoint_path, model=policy, map_location=device)
    set_seed(seed)
    policy.eval()

    dataloader = make_dataloader(config)
    normalization_stats = load_policy_normalization_stats(
        data_dir=get_nested(config, "data.path"),
        normalization_stats_path=get_nested(config, "data.normalization_stats_path"),
    )
    num_slots = int(get_nested(config, "text.num_slots", get_nested(config, "slot.num_slots", 5)))
    sample_steps = int(args.sample_steps or get_nested(config, "action_decoder.sample_steps", 20))
    discretize_gripper = bool(get_nested(config, "action_decoder.discretize_gripper", False))
    gripper_index = int(get_nested(config, "action_decoder.gripper_index", 6))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": checkpoint_path,
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "loaded_trainable_tensors": int(checkpoint.get("loaded_trainable_tensors", 0)),
        "skipped_trainable_tensors": int(checkpoint.get("skipped_trainable_tensors", 0)),
        "split": get_nested(config, "data.split"),
        "data_path": get_nested(config, "data.path"),
        "shard_indices": get_nested(config, "data.shard_indices", None),
        "batch_size": int(get_nested(config, "data.batch_size", 1)),
        "num_workers": int(get_nested(config, "data.num_workers", 0)),
        "max_trajectories": int(args.max_trajectories),
        "history_frames": int(get_nested(config, "data.history_frames", 4)),
        "future_steps": int(get_nested(config, "data.future_steps", 4)),
        "sample_steps": sample_steps,
        "seed": seed,
        "plots_enabled": not args.no_plots,
        "plot_format": args.plot_format,
        "saved_action": "sampled action chunk index 0, aligned to dataset action at timestep t",
        "action_layout": ["world_vector_x", "world_vector_y", "world_vector_z", "rotation_delta_x", "rotation_delta_y", "rotation_delta_z", "open_gripper"],
        "files": [],
    }
    write_json(output_dir / "manifest.json", manifest)

    rows: List[Dict[str, np.ndarray]] = []
    chunk_index = 0
    total_samples = 0
    total_batches = 0
    seen_trajectories: set[tuple[int, int]] = set()
    current_plot_buffer: Optional[Dict[str, Any]] = None
    plot_index: List[Dict[str, Any]] = []
    start_time = time.time()
    progress = tqdm(desc="policy forward", unit="sample", dynamic_ncols=True) if tqdm is not None else None

    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            if args.max_trajectories > 0 and int(get_nested(config, "data.num_workers", 0)) != 0:
                raise ValueError("--max-trajectories requires --num-workers 0 for deterministic trajectory order.")
            allowed_indices, reached_trajectory_limit = allowed_indices_for_trajectory_limit(
                raw_batch,
                seen_trajectories=seen_trajectories,
                max_trajectories=int(args.max_trajectories),
            )
            if not allowed_indices:
                break
            if len(allowed_indices) != int(raw_batch["future_actions"].shape[0]):
                raw_batch = slice_batch(raw_batch, allowed_indices)

            batch = move_policy_batch_to_device(raw_batch, device)
            batch = attach_text_features(batch, text_embedder, num_slots=num_slots, device=device)
            if args.max_samples > 0:
                remaining = args.max_samples - total_samples
                if remaining <= 0:
                    break
                current = int(batch["future_actions"].shape[0])
                if current > remaining:
                    batch = {
                        key: value[:remaining]
                        if torch.is_tensor(value) and value.shape[:1] == (current,)
                        else value[:remaining]
                        if isinstance(value, list) and len(value) == current
                        else value
                        for key, value in batch.items()
                    }

            with autocast_context(device, precision):
                sample = policy.sample_actions(
                    batch,
                    num_steps=sample_steps,
                    discretize_gripper=discretize_gripper,
                    gripper_index=gripper_index,
                )

            pred_norm_all = sample["action_pred"]
            target_norm_all = batch["future_actions"]
            action_mask_all = batch["future_action_padding_mask"]
            pred_original_all = denormalize_actions(
                pred_norm_all,
                config=config,
                stats=normalization_stats,
                padding_mask=action_mask_all,
            )
            target_original_all = denormalize_actions(
                target_norm_all,
                config=config,
                stats=normalization_stats,
                padding_mask=action_mask_all,
            )

            batch_size = int(pred_norm_all.shape[0])
            sample_indices = np.arange(total_samples, total_samples + batch_size, dtype=np.int64)
            pred_action = tensor_to_numpy(pred_original_all[:, 0, :]).astype(np.float32)
            target_action = tensor_to_numpy(target_original_all[:, 0, :]).astype(np.float32)
            row: Dict[str, np.ndarray] = {
                "sample_index": sample_indices,
                "shard_index": tensor_to_numpy(batch["shard_index"]).astype(np.int64),
                "trajectory_index": tensor_to_numpy(batch["trajectory_index"]).astype(np.int64),
                "timestep": tensor_to_numpy(batch["timestep"]).astype(np.int64),
                "valid_action": (~batch["future_action_padding_mask"][:, 0].detach().cpu()).numpy().astype(np.bool_),
                "pred_action": pred_action,
                "target_action": target_action,
                "abs_error": np.abs(pred_action - target_action).astype(np.float32),
            }
            if args.save_normalized:
                row["pred_action_normalized"] = tensor_to_numpy(pred_norm_all[:, 0, :]).astype(np.float32)
                row["target_action_normalized"] = tensor_to_numpy(target_norm_all[:, 0, :]).astype(np.float32)
            rows.append(row)

            if args.save_instructions:
                append_instructions(output_dir / "instructions.jsonl", batch, sample_indices)
            if not args.no_plots:
                current_plot_buffer = update_trajectory_plots(
                    output_dir=output_dir,
                    plot_format=args.plot_format,
                    current_buffer=current_plot_buffer,
                    plot_index=plot_index,
                    shard_indices=row["shard_index"],
                    trajectory_indices=row["trajectory_index"],
                    timesteps=row["timestep"],
                    pred_actions=row["pred_action"],
                    target_actions=row["target_action"],
                    valid_actions=row["valid_action"],
                )

            total_samples += batch_size
            total_batches += 1
            if progress is not None:
                progress.update(batch_size)
                progress.set_postfix(batch=total_batches, chunk=chunk_index, traj=len(seen_trajectories))

            buffered = sum(int(row_part["pred_action"].shape[0]) for row_part in rows)
            if buffered >= args.chunk_size:
                path = flush_chunk(
                    output_dir=output_dir,
                    chunk_index=chunk_index,
                    rows=rows,
                    save_normalized=args.save_normalized,
                )
                manifest["files"].append(str(path.relative_to(output_dir)))
                write_json(output_dir / "manifest.json", manifest)
                rows.clear()
                chunk_index += 1
            if reached_trajectory_limit:
                break

    if rows:
        path = flush_chunk(
            output_dir=output_dir,
            chunk_index=chunk_index,
            rows=rows,
            save_normalized=args.save_normalized,
        )
        manifest["files"].append(str(path.relative_to(output_dir)))
        chunk_index += 1
    if not args.no_plots and current_plot_buffer is not None:
        entry = plot_trajectory_buffer(output_dir=output_dir, buffer=current_plot_buffer, plot_format=args.plot_format)
        if entry is not None:
            plot_index.append(entry)
        write_json(output_dir / "plots" / "plot_index.json", {"trajectories": plot_index})

    if progress is not None:
        progress.close()

    elapsed = time.time() - start_time
    manifest.update(
        total_samples=int(total_samples),
        total_batches=int(total_batches),
        total_trajectories=int(len(seen_trajectories)),
        total_plotted_trajectories=int(len(plot_index)),
        num_chunks=int(chunk_index),
        elapsed_sec=float(elapsed),
        samples_per_sec=float(total_samples / elapsed) if elapsed > 0 else None,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"[done] samples={total_samples} batches={total_batches} chunks={chunk_index} "
        f"elapsed_sec={elapsed:.1f} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
