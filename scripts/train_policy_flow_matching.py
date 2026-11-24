#!/usr/bin/env python
"""Train Bridge V-JEPA2/slot conditioned flow-matching action policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None

from lang_slot.paths import check_policy_files
from lang_slot.training.policy_data import build_policy_dataloader
from lang_slot.data import policy_episode_index_path
from lang_slot.training.policy_utils import (
    apply_trainable_prefixes,
    attach_text_features,
    build_optimizer,
    build_policy_from_config,
    build_text_embedder,
    get_nested,
    json_default,
    latest_checkpoint,
    load_checkpoint,
    load_yaml_config,
    maybe_load_partial_checkpoint_from_config,
    maybe_denormalize_actions,
    move_policy_batch_to_device,
    parse_shard_indices,
    resolve_device,
    save_checkpoint,
    save_yaml_config,
    set_frozen_backbones_eval,
    set_nested,
    set_seed,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/stage1.yaml"),
        help="YAML config path.",
    )
    parser.add_argument("--output-dir", default="", help="Override experiment.output_dir.")
    parser.add_argument("--device", default="", help="Override training.device.")
    parser.add_argument(
        "--dataset",
        choices=("bridge", "simpler_pick_place"),
        default="",
        help="Override data.dataset (the reader implementation).",
    )
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Override data.path with a Bridge/filtered-subset TFDS directory.",
    )
    parser.add_argument(
        "--metadata-manifest-path",
        default="",
        help=(
            "Override data.metadata_manifest_path. Filtered subsets normally use "
            "their export_manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--data-split",
        choices=("train", "val", "all"),
        default="",
        help="Override data.split; 'all' trains on both exported train and val shards.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Override training.max_steps.")
    parser.add_argument(
        "--max-epochs",
        "--max-epoch",
        dest="max_epochs",
        type=int,
        default=-1,
        help="Override training.max_epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Override data.batch_size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Override data.num_workers; 0 keeps loading in the training process.",
    )
    parser.add_argument(
        "--vjepa2-window-batch-size",
        type=int,
        default=0,
        help="Override vjepa2.window_batch_size; larger values batch more V-JEPA2 windows per forward.",
    )
    parser.add_argument("--shard-indices", default="", help="Comma-separated shard ids override; empty uses config.")
    parser.add_argument(
        "--trainable-prefixes",
        default="",
        help=(
            "Comma-separated parameter name prefixes to train, e.g. 'action_decoder.'. "
            "Empty keeps the config/default trainability."
        ),
    )
    parser.add_argument("--resume", default="", help="Checkpoint path override.")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable latest.pt auto-resume.")
    parser.add_argument("--partial-checkpoint", default="", help="Override partial_init.checkpoint warm-start path.")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--state-delta-action",
        action="store_true",
        help="Use state[t+1,:6]-state[t,:6] for action dims 0:6 and raw gripper for dim 6.",
    )
    action_group.add_argument(
        "--raw-action",
        action="store_true",
        help="Use raw Bridge world_vector/rotation_delta/open_gripper actions.",
    )
    data_filter_group = parser.add_mutually_exclusive_group()
    data_filter_group.add_argument(
        "--valid-trajectories-only",
        action="store_true",
        help="Train only on trajectories in the configured strict-valid episode index.",
    )
    data_filter_group.add_argument(
        "--all-trajectories",
        action="store_true",
        help="Ignore valid-episode indexes and train on the complete Bridge split.",
    )
    slot_group = parser.add_mutually_exclusive_group()
    slot_group.add_argument(
        "--freeze-slot",
        action="store_true",
        help="Freeze the future slot encoder and disable slot auxiliary losses.",
    )
    slot_group.add_argument(
        "--train-slot",
        action="store_true",
        help="Train the future slot encoder according to slot.compute_auxiliary_loss.",
    )
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--latest-only-checkpoint",
        action="store_true",
        help="Save only checkpoints/latest.pt at each save point.",
    )
    ckpt_group.add_argument(
        "--keep-step-checkpoints",
        action="store_true",
        help="Also save numbered checkpoints/step_*.pt at each save point.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build everything and run one forward/backward-free batch.")
    return parser


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir:
        set_nested(config, "experiment.output_dir", args.output_dir)
    if args.device:
        set_nested(config, "training.device", args.device)
    if args.dataset:
        set_nested(config, "data.dataset", args.dataset)
    if args.dataset_path:
        set_nested(config, "data.path", str(Path(args.dataset_path).expanduser().resolve()))
    if args.metadata_manifest_path:
        set_nested(
            config,
            "data.metadata_manifest_path",
            str(Path(args.metadata_manifest_path).expanduser().resolve()),
        )
    if args.data_split:
        set_nested(config, "data.split", args.data_split)
    if args.max_steps > 0:
        set_nested(config, "training.max_steps", args.max_steps)
    if args.max_epochs >= 0:
        set_nested(config, "training.max_epochs", args.max_epochs)
    if args.batch_size > 0:
        set_nested(config, "data.batch_size", args.batch_size)
    if args.num_workers >= 0:
        set_nested(config, "data.num_workers", args.num_workers)
    if args.vjepa2_window_batch_size > 0:
        set_nested(config, "vjepa2.window_batch_size", args.vjepa2_window_batch_size)
    if args.shard_indices:
        set_nested(config, "data.shard_indices", list(parse_shard_indices(args.shard_indices) or ()))
    if args.trainable_prefixes:
        set_nested(
            config,
            "training.trainable_prefixes",
            [part.strip() for part in args.trainable_prefixes.split(",") if part.strip()],
        )
    if args.resume:
        set_nested(config, "training.resume", args.resume)
    if args.no_auto_resume:
        set_nested(config, "training.auto_resume", False)
    if args.partial_checkpoint:
        set_nested(config, "partial_init.checkpoint", args.partial_checkpoint)
    if args.state_delta_action:
        set_nested(config, "data.state_delta_action", True)
    if args.raw_action:
        set_nested(config, "data.state_delta_action", False)
    if args.valid_trajectories_only:
        set_nested(config, "data.valid_trajectories_only", True)
    if args.all_trajectories:
        set_nested(config, "data.valid_trajectories_only", False)
    if args.freeze_slot:
        set_nested(config, "slot.freeze", True)
    if args.train_slot:
        set_nested(config, "slot.freeze", False)
    if args.latest_only_checkpoint:
        set_nested(config, "training.keep_step_checkpoints", False)
    if args.keep_step_checkpoints:
        set_nested(config, "training.keep_step_checkpoints", True)
    return config


def make_dataloader(config: Dict[str, Any], start_cursor: Optional[Dict[str, Any]] = None) -> DataLoader:
    return build_policy_dataloader(config, start_cursor=start_cursor)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown precision {precision!r}; expected fp32/fp16/bf16.")


def scheduled_learning_rate(config: Dict[str, Any], step_number: int) -> float:
    optim_cfg = config.get("optimizer", {})
    base_lr = float(optim_cfg.get("lr", 1e-4))
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))
    if warmup_steps <= 0:
        return base_lr
    warmup_progress = min(max(int(step_number), 1) / float(warmup_steps), 1.0)
    return base_lr * warmup_progress


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def maybe_make_scaler(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "fp16":
        return torch.cuda.amp.GradScaler()
    return None


def normalized_shard_indices(config: Dict[str, Any]) -> Optional[list[int]]:
    parsed = parse_shard_indices(get_nested(config, "data.shard_indices", None))
    return None if parsed is None else [int(index) for index in parsed]


def validate_resume_cursor(
    cursor: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not cursor:
        return None
    data_cfg = config.get("data", {})
    expected = {
        "split": data_cfg.get("split", "train"),
        "seed": int(get_nested(config, "experiment.seed", 0)),
        "shuffle_shards": bool(data_cfg.get("shuffle_shards", True)),
        "shard_indices": normalized_shard_indices(config),
        "valid_trajectories_only": bool(data_cfg.get("valid_trajectories_only", False)),
        "episode_index_path": policy_episode_index_path(
            data_cfg,
            str(data_cfg.get("split", "train")),
        ),
    }
    for key, value in expected.items():
        if cursor.get(key) != value:
            print(
                f"[data] checkpoint data_cursor {key}={cursor.get(key)!r} does not match "
                f"current config {value!r}; starting dataset from beginning",
                flush=True,
            )
            return None
    if int(data_cfg.get("num_workers", 0)) != 0:
        print("[data] data_cursor resume requires data.num_workers=0; starting from beginning", flush=True)
        return None
    return dict(cursor)


def data_cursor_from_batch(
    batch: Dict[str, Any],
    config: Dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    samples_seen: int,
) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    last = -1
    return {
        "split": data_cfg.get("split", "train"),
        "seed": int(get_nested(config, "experiment.seed", 0)),
        "shuffle_shards": bool(data_cfg.get("shuffle_shards", True)),
        "shard_indices": normalized_shard_indices(config),
        "valid_trajectories_only": bool(data_cfg.get("valid_trajectories_only", False)),
        "episode_index_path": policy_episode_index_path(
            data_cfg,
            str(data_cfg.get("split", "train")),
        ),
        "shard_index": int(batch["shard_index"][last].detach().cpu()),
        "trajectory_index": int(batch["trajectory_index"][last].detach().cpu()),
        "timestep": int(batch["timestep"][last].detach().cpu()),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "samples_seen": int(samples_seen),
    }


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_summary_writer(
    config: Dict[str, Any],
    output_dir: Path,
    *,
    purge_step: Optional[int] = None,
) -> Optional[Any]:
    if not bool(get_nested(config, "tensorboard.enabled", True)):
        return None
    from torch.utils.tensorboard import SummaryWriter

    configured = get_nested(config, "tensorboard.log_dir", None)
    log_dir = Path(configured) if configured else output_dir / "tensorboard"
    return SummaryWriter(
        log_dir=str(log_dir),
        flush_secs=int(get_nested(config, "tensorboard.flush_secs", 30)),
        purge_step=purge_step,
    )


def slot_statistics(slots: torch.Tensor) -> Dict[str, float]:
    """Return scalar statistics for every recurrent time/slot token."""
    if slots.ndim != 4:
        raise ValueError(f"slots must be [B,T,K,D], got {tuple(slots.shape)}.")
    values = slots.detach().float()
    stats: Dict[str, float] = {}
    per_slot = values.permute(2, 0, 1, 3)
    for slot_index in range(values.shape[2]):
        slot = per_slot[slot_index]
        stats[f"slot_{slot_index}/mean"] = float(slot.mean().cpu())
        stats[f"slot_{slot_index}/std"] = float(slot.std(unbiased=False).cpu())
        stats[f"slot_{slot_index}/l2"] = float(slot.norm(dim=-1).mean().cpu())
    norms = values.norm(dim=-1).mean(dim=0)
    for time_index in range(values.shape[1]):
        for slot_index in range(values.shape[2]):
            stats[f"time_{time_index}/slot_{slot_index}_l2"] = float(
                norms[time_index, slot_index].cpu()
            )
    return stats


def log_step_tensorboard(
    writer: Optional[Any],
    *,
    out: Dict[str, Any],
    global_step: int,
    samples_seen: int,
    grad_norm: Optional[float],
    histogram_every: int,
    log_slot_stats: bool = True,
) -> Dict[str, float]:
    future_outputs = out["condition"]["future_outputs"]
    slots = future_outputs.get("slots") if isinstance(future_outputs, dict) else None
    stats = slot_statistics(slots) if log_slot_stats and torch.is_tensor(slots) else {}
    if writer is None:
        return stats
    writer.add_scalar("step/loss_total", out["loss"].detach().float(), global_step)
    writer.add_scalar("step/loss_action", out["action_loss"].detach().float(), global_step)
    if log_slot_stats and "loss" in future_outputs:
        writer.add_scalar("step/loss_slot_total", future_outputs["loss"].detach().float(), global_step)
    if log_slot_stats and "reconstruction_loss" in future_outputs:
        writer.add_scalar(
            "step/loss_slot_reconstruction",
            future_outputs["reconstruction_loss"].detach().float(),
            global_step,
        )
    if log_slot_stats and "slot_lang_contrastive_loss" in future_outputs:
        writer.add_scalar(
            "step/loss_slot_lang_contrastive",
            future_outputs["slot_lang_contrastive_loss"].detach().float(),
            global_step,
        )
    if log_slot_stats and "flow_reconstruction_loss" in future_outputs:
        writer.add_scalar(
            "step/loss_slot_flow_reconstruction",
            future_outputs["flow_reconstruction_loss"].detach().float(),
            global_step,
        )
    writer.add_scalar("step/samples_seen", samples_seen, global_step)
    if grad_norm is not None:
        writer.add_scalar("step/gradient_norm", grad_norm, global_step)
    for name, value in stats.items():
        writer.add_scalar(f"slots_step/{name}", value, global_step)
    if log_slot_stats and torch.is_tensor(slots) and histogram_every > 0 and global_step % histogram_every == 0:
        slots_float = slots.detach().float()
        for slot_index in range(slots.shape[2]):
            writer.add_histogram(
                f"slots_histogram/slot_{slot_index}",
                slots_float[:, :, slot_index],
                global_step,
            )
    return stats


def log_epoch_tensorboard(
    writer: Optional[Any],
    *,
    epoch: int,
    totals: Dict[str, float],
    batches: int,
    slot_totals: Dict[str, float],
    partial: bool,
) -> None:
    if writer is None or batches <= 0:
        return
    writer.add_scalar("epoch/loss_total", totals["loss"] / batches, epoch)
    writer.add_scalar("epoch/loss_action", totals["action_loss"] / batches, epoch)
    if totals["grad_norm_count"] > 0:
        writer.add_scalar(
            "epoch/gradient_norm",
            totals["grad_norm"] / totals["grad_norm_count"],
            epoch,
        )
    writer.add_scalar("epoch/batches", batches, epoch)
    writer.add_scalar("epoch/is_partial", float(partial), epoch)
    for name, value in slot_totals.items():
        writer.add_scalar(f"slots_epoch/{name}", value / batches, epoch)
    writer.flush()


def save_training_checkpoint(
    output_dir: Path,
    *,
    config: Dict[str, Any],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    global_step: int,
    samples_seen: int,
    epoch: int,
    data_cursor: Optional[Dict[str, Any]],
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    latest_path = ckpt_dir / "latest.pt"
    save_checkpoint(
        str(latest_path),
        model=policy,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        global_step=global_step,
        samples_seen=samples_seen,
        epoch=epoch,
        data_cursor=data_cursor,
    )
    if bool(get_nested(config, "training.keep_step_checkpoints", True)):
        step_path = ckpt_dir / f"step_{global_step:08d}.pt"
        save_checkpoint(
            str(step_path),
            model=policy,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            global_step=global_step,
            samples_seen=samples_seen,
            epoch=epoch,
            data_cursor=data_cursor,
        )


def slice_policy_batch(batch: Dict[str, Any], max_samples: int) -> Dict[str, Any]:
    if max_samples <= 0:
        return batch
    batch_size = int(batch["future_actions"].shape[0])
    if batch_size <= max_samples:
        return batch
    sliced: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            sliced[key] = value[:max_samples]
        elif isinstance(value, list) and len(value) == batch_size:
            sliced[key] = value[:max_samples]
        else:
            sliced[key] = value
    return sliced


def sample_and_log_actions(
    *,
    policy,
    batch: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    global_step: int,
) -> None:
    original_batch_size = int(batch["future_actions"].shape[0])
    sample_batch_size = int(get_nested(config, "training.sample_log_batch_size", 8))
    batch = slice_policy_batch(batch, sample_batch_size)
    logged_batch_size = int(batch["future_actions"].shape[0])

    was_training = policy.training
    policy.eval()
    set_frozen_backbones_eval(policy)
    with torch.no_grad():
        sample = policy.sample_actions(
            batch,
            num_steps=int(get_nested(config, "action_decoder.sample_steps", 20)),
            discretize_gripper=bool(get_nested(config, "action_decoder.discretize_gripper", False)),
            gripper_index=int(get_nested(config, "action_decoder.gripper_index", 6)),
        )
    pred_norm = sample["action_pred"]
    target_norm = batch["future_actions"]
    mask = batch["future_action_padding_mask"]
    pred_original = maybe_denormalize_actions(pred_norm, config, padding_mask=mask)
    target_original = maybe_denormalize_actions(target_norm, config, padding_mask=mask)
    row = {
        "global_step": global_step,
        "original_batch_size": original_batch_size,
        "logged_batch_size": logged_batch_size,
        "timestep": batch["timestep"].detach().cpu().tolist(),
        "trajectory_index": batch["trajectory_index"].detach().cpu().tolist(),
        "instruction": batch["instruction"],
        "object_nouns": batch.get("object_nouns", []),
        "action_pred_original": pred_original.detach().cpu().tolist(),
        "action_target_original": target_original.detach().cpu().tolist(),
        "future_action_padding_mask": mask.detach().cpu().tolist(),
    }
    write_jsonl(output_dir / "samples_original_actions.jsonl", row)
    if was_training:
        policy.train()
        set_frozen_backbones_eval(policy)


def main() -> None:
    args = build_argparser().parse_args()
    config = apply_overrides(load_yaml_config(args.config), args)
    check_policy_files(config, checkpoint=config["training"].get("resume"), split=config["data"]["split"])
    output_dir = Path(get_nested(config, "experiment.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, str(output_dir / "config_resolved.yaml"))

    set_seed(int(get_nested(config, "experiment.seed", 0)))
    device = resolve_device(get_nested(config, "training.device", "cuda"))
    precision = str(get_nested(config, "training.precision", "bf16"))

    print(f"[setup] device={device} precision={precision} output_dir={output_dir}", flush=True)
    train_split = str(get_nested(config, "data.split", "train"))
    train_episode_index = policy_episode_index_path(config.get("data", {}), train_split)
    print(
        f"[data] split={train_split} valid_trajectories_only={train_episode_index is not None} "
        f"episode_index={train_episode_index or '<all trajectories>'}",
        flush=True,
    )
    policy = build_policy_from_config(config, device)
    text_embedder = build_text_embedder(config, device)

    global_step = 0
    samples_seen = 0
    epoch = 0
    data_cursor: Optional[Dict[str, Any]] = None
    resume_path = get_nested(config, "training.resume", None)
    if not resume_path and bool(get_nested(config, "training.auto_resume", True)):
        resume_path = latest_checkpoint(str(output_dir))

    if not resume_path:
        partial_result = maybe_load_partial_checkpoint_from_config(policy, config, map_location="cpu")
        if partial_result is not None:
            print(
                "[partial_init] "
                f"path={partial_result.get('path')} "
                f"checkpoint_step={partial_result.get('checkpoint_global_step')} "
                f"loaded_tensors={partial_result.get('loaded_tensors')} "
                f"missing={partial_result.get('missing_tensors')} "
                f"shape_mismatch={partial_result.get('shape_mismatch_tensors')} "
                f"include_prefixes={partial_result.get('include_prefixes')}",
                flush=True,
            )
            if partial_result.get("missing_tensors") or partial_result.get("shape_mismatch_tensors"):
                print(
                    "[partial_init] details="
                    + json.dumps(
                        {
                            "missing_names": partial_result.get("missing_names", []),
                            "shape_mismatches": partial_result.get("shape_mismatches", []),
                        },
                        default=json_default,
                    ),
                    flush=True,
                )

    trainable_result = apply_trainable_prefixes(policy, get_nested(config, "training.trainable_prefixes", None))
    if trainable_result.get("enabled"):
        print(
            "[trainable] "
            f"include_prefixes={trainable_result.get('include_prefixes')} "
            f"matched_tensors={trainable_result.get('matched_parameter_tensors')} "
            f"trainable_parameters={trainable_result.get('trainable_parameter_count')} "
            f"frozen_parameters={trainable_result.get('frozen_parameter_count')}",
            flush=True,
        )

    optimizer = build_optimizer(config, policy)
    scaler = maybe_make_scaler(device, precision)

    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=policy,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        global_step = int(checkpoint.get("global_step", 0))
        samples_seen = int(checkpoint.get("samples_seen", 0))
        epoch = int(checkpoint.get("epoch", 0))
        data_cursor = validate_resume_cursor(checkpoint.get("data_cursor"), config)
        print(
            f"[resume] path={resume_path} step={global_step} samples_seen={samples_seen} "
            f"loaded_tensors={checkpoint.get('loaded_trainable_tensors')}",
            flush=True,
        )
        if checkpoint.get("optimizer_loaded") is False:
            print(
                f"[resume] optimizer state was not loaded: {checkpoint.get('optimizer_load_error')}",
                flush=True,
            )
        if checkpoint.get("scheduler_loaded") is False:
            print(
                f"[resume] scheduler state was not loaded: {checkpoint.get('scheduler_load_error')}",
                flush=True,
            )
        if data_cursor is None:
            print("[data] no compatible data_cursor in checkpoint; starting dataset from beginning", flush=True)
        else:
            print(f"[data] resuming dataset after cursor={data_cursor}", flush=True)

    writer = create_summary_writer(
        config,
        output_dir,
        purge_step=global_step if resume_path else None,
    )
    if writer is not None:
        writer.add_text("run/config", json.dumps(config, ensure_ascii=False, indent=2), global_step)
        writer.add_scalar(
            "run/trainable_parameters",
            sum(param.numel() for param in policy.parameters() if param.requires_grad),
            global_step,
        )
        print(f"[tensorboard] log_dir={writer.log_dir}", flush=True)

    max_steps = int(get_nested(config, "training.max_steps", 10000))
    max_epochs = int(get_nested(config, "training.max_epochs", get_nested(config, "training.max_epoch", 0)))
    use_epoch_limit = max_epochs > 0
    grad_accum = int(get_nested(config, "training.grad_accum_steps", 1))
    if grad_accum <= 0:
        raise ValueError("training.grad_accum_steps must be positive.")
    log_every = int(get_nested(config, "training.log_every", 10))
    save_every = int(get_nested(config, "training.save_every", 500))
    sample_every = int(get_nested(config, "training.sample_every", 200))
    grad_clip = float(get_nested(config, "training.grad_clip_norm", 1.0))
    slot_enabled = bool(get_nested(config, "slot.enabled", True)) and bool(
        get_nested(config, "condition_fusion.use_slot_tokens", True)
    )
    freeze_slot = bool(get_nested(config, "slot.freeze", False))
    compute_slot_loss = slot_enabled and bool(get_nested(config, "slot.compute_auxiliary_loss", False)) and not freeze_slot
    num_slots = int(get_nested(config, "text.num_slots", get_nested(config, "slot.num_slots", 5)))
    tb_step_every = int(get_nested(config, "tensorboard.step_log_every", log_every))
    histogram_every = int(get_nested(config, "tensorboard.histogram_every", 100))
    log_slot_stats = slot_enabled and bool(get_nested(config, "tensorboard.log_slot_stats", True))

    logs_path = output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    set_optimizer_lr(optimizer, scheduled_learning_rate(config, global_step + 1))
    start_time = time.time()
    run_start_step = global_step
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=None if use_epoch_limit else max_steps,
            initial=0 if use_epoch_limit else global_step,
            desc=(
                f"train epoch {epoch + 1}/{max_epochs} step {global_step}"
                if use_epoch_limit
                else f"train step {global_step}/{max_steps}"
            ),
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

    while (epoch < max_epochs) if use_epoch_limit else (global_step < max_steps):
        dataloader = make_dataloader(config, start_cursor=data_cursor)
        policy.train()
        set_frozen_backbones_eval(policy)
        epoch_totals = {
            "loss": 0.0,
            "action_loss": 0.0,
            "grad_norm": 0.0,
            "grad_norm_count": 0.0,
        }
        epoch_slot_totals: Dict[str, float] = {}
        epoch_batches = 0
        reached_training_limit = False
        for batch_index, raw_batch in enumerate(dataloader):
            if not use_epoch_limit and global_step >= max_steps:
                reached_training_limit = True
                break
            if progress_bar is not None:
                progress_bar.set_description(
                    f"train epoch {epoch + 1}/{max_epochs} step {global_step + 1}"
                    if use_epoch_limit
                    else f"train step {global_step + 1}/{max_steps}"
                )
            batch = move_policy_batch_to_device(raw_batch, device)
            batch = attach_text_features(batch, text_embedder, num_slots=num_slots, device=device)

            with autocast_context(device, precision):
                out = policy(batch, compute_slot_loss=compute_slot_loss)
                loss = out["loss"] / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_norm_value: Optional[float] = None
            if (global_step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                trainable_params = [p for p in policy.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    grad_clip if grad_clip > 0 else float("inf"),
                )
                grad_norm_value = float(grad_norm.detach().float().cpu())
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = int(batch["future_actions"].shape[0])
            samples_seen += batch_size
            global_step += 1
            data_cursor = data_cursor_from_batch(
                batch,
                config,
                epoch=epoch,
                global_step=global_step,
                samples_seen=samples_seen,
            )
            set_optimizer_lr(optimizer, scheduled_learning_rate(config, global_step + 1))

            current_loss = float(out["loss"].detach().float().cpu())
            current_action_loss = float(out["action_loss"].detach().float().cpu())
            elapsed = time.time() - start_time
            completed_this_run = max(global_step - run_start_step, 1)
            sec_per_step = elapsed / completed_this_run
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{current_loss:.4g}",
                    action=f"{current_action_loss:.4g}",
                    bs=batch_size,
                    s_step=f"{sec_per_step:.2f}",
                )
            future_outputs = out["condition"]["future_outputs"]
            slots = future_outputs.get("slots") if isinstance(future_outputs, dict) else None
            current_slot_stats = slot_statistics(slots) if log_slot_stats and torch.is_tensor(slots) else {}
            epoch_totals["loss"] += current_loss
            epoch_totals["action_loss"] += current_action_loss
            if grad_norm_value is not None:
                epoch_totals["grad_norm"] += grad_norm_value
                epoch_totals["grad_norm_count"] += 1.0
            for name, value in current_slot_stats.items():
                epoch_slot_totals[name] = epoch_slot_totals.get(name, 0.0) + value
            epoch_batches += 1

            if tb_step_every > 0 and (global_step % tb_step_every == 0 or global_step == 1):
                log_step_tensorboard(
                    writer,
                    out=out,
                    global_step=global_step,
                    samples_seen=samples_seen,
                    grad_norm=grad_norm_value,
                    histogram_every=histogram_every,
                    log_slot_stats=log_slot_stats,
                )

            if global_step % log_every == 0 or global_step == 1:
                steps_remaining = None if use_epoch_limit else max(max_steps - global_step, 0)
                eta_sec = None if steps_remaining is None else steps_remaining * sec_per_step
                row = {
                    "global_step": global_step,
                    "max_steps": None if use_epoch_limit else max_steps,
                    "max_epochs": max_epochs if use_epoch_limit else None,
                    "progress": (
                        f"epoch {epoch + 1}/{max_epochs}, step {global_step}"
                        if use_epoch_limit
                        else f"{global_step}/{max_steps}"
                    ),
                    "progress_percent": (
                        None
                        if use_epoch_limit
                        else round(100.0 * global_step / max(max_steps, 1), 3)
                    ),
                    "steps_remaining": steps_remaining,
                    "epoch": epoch,
                    "samples_seen": samples_seen,
                    "loss": current_loss,
                    "action_loss": current_action_loss,
                    "gradient_norm": grad_norm_value,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_sec": round(elapsed, 3),
                    "sec_per_step": round(sec_per_step, 3),
                    "eta_sec": None if eta_sec is None else round(eta_sec, 3),
                    "eta_hms": None if eta_sec is None else format_duration(eta_sec),
                    "hist_images_shape": list(batch["hist_images"].shape),
                    "future_images_shape": list(batch["future_images"].shape),
                    "future_actions_shape": list(batch["future_actions"].shape),
                }
                log_line = json.dumps(row, ensure_ascii=False)
                if progress_bar is not None:
                    progress_bar.write(log_line)
                else:
                    print(log_line, flush=True)
                write_jsonl(logs_path, row)

            if sample_every > 0 and global_step % sample_every == 0:
                del out, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sample_and_log_actions(
                    policy=policy,
                    batch=batch,
                    config=config,
                    output_dir=output_dir,
                    global_step=global_step,
                )

            if save_every > 0 and global_step % save_every == 0:
                save_training_checkpoint(
                    output_dir,
                    config=config,
                    policy=policy,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    samples_seen=samples_seen,
                    epoch=epoch,
                    data_cursor=data_cursor,
                )

            if not use_epoch_limit and global_step >= max_steps:
                reached_training_limit = True
                break

            if args.dry_run:
                log_epoch_tensorboard(
                    writer,
                    epoch=epoch,
                    totals=epoch_totals,
                    batches=epoch_batches,
                    slot_totals=epoch_slot_totals,
                    partial=True,
                )
                if writer is not None:
                    writer.close()
                if progress_bar is not None:
                    progress_bar.close()
                print("[dry-run] completed one batch", flush=True)
                return
        log_epoch_tensorboard(
            writer,
            epoch=epoch,
            totals=epoch_totals,
            batches=epoch_batches,
            slot_totals=epoch_slot_totals,
            partial=reached_training_limit,
        )
        if reached_training_limit:
            break
        data_cursor = None
        epoch += 1

    save_training_checkpoint(
        output_dir,
        config=config,
        policy=policy,
        optimizer=optimizer,
        scaler=scaler,
        global_step=global_step,
        samples_seen=samples_seen,
        epoch=epoch,
        data_cursor=data_cursor,
    )
    if writer is not None:
        writer.flush()
        writer.close()
    if progress_bar is not None:
        progress_bar.close()
    print(f"[done] step={global_step} samples_seen={samples_seen} output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
