#!/usr/bin/env python
"""Phase B: run a SIMPLER/ManiSkill3 rollout conditioned on a Cosmos reference."""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
import sys
from typing import Any, Deque, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lang_slot.paths import data_path, resources
from evaluation.simpler.eval_logger import EvalLogger
from evaluation.simpler.observation_history import ObservationHistory
from evaluation.simpler.reference_buffer import ReferenceBuffer
from evaluation.simpler.reference_loader import load_reference
from evaluation.simpler.reference_preprocess import resample_frames_nearest
from evaluation.simpler.simpler_env_utils import (
    configure_simulation,
    bridge_action_to_maniskill_action,
    default_camera_name,
    extract_bridge_state,
    extract_rgb,
    get_instruction,
    make_maniskill3_env,
    parse_seed_list,
    reset_episode,
    resolve_task,
    torch_action_for_env,
)


DEFAULT_REFERENCE_ROOT = Path(os.environ.get("COSMOS3_HOME") or data_path("cosmos3")) / "reference_videos"
DEFAULT_RESULTS_ROOT = resources()["POLICY_OUTPUT_DIR"] / "simpler" / "results"
DEFAULT_RUNS_ROOT = resources()["POLICY_OUTPUT_DIR"] / "simpler" / "runs"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="widowx_carrot_on_plate")
    parser.add_argument("--env-id", default="")
    parser.add_argument("--tasks", default="", help="Comma-separated SIMPLER task names for batch rollout.")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds. Overrides --start-seed/--num-episodes.")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--camera-name", default="")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--obs-mode", default="rgb+segmentation")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--reference", default="", help="Explicit .npz reference for single task/seed.")
    parser.add_argument(
        "--run-name",
        default="",
        help="Run identifier used to create <run-root>/<run-name>/results.",
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument(
        "--results-root",
        default="",
        help="Explicit results directory. Overrides the directory derived from --run-name.",
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/simpler.yaml"))
    parser.add_argument("--checkpoint", default=str(resources()["POLICY_CHECKPOINT"]))
    parser.add_argument("--device", default="")
    parser.add_argument("--precision", default="")
    parser.add_argument("--sample-steps", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument(
        "--future-horizon",
        type=int,
        default=0,
        help="Reference horizon; 0 reads data.future_steps from the policy config.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses env truncation.")
    parser.add_argument("--no-reset-by-episode-id", action="store_true")
    return parser


def resolve_run_output_path(args: argparse.Namespace) -> None:
    """Resolve an isolated result directory while preserving the old default."""
    run_name = str(args.run_name).strip()
    if run_name:
        if Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("--run-name must be a single path-safe directory name.")
        if not args.results_root:
            args.results_root = str(Path(args.run_root).expanduser().resolve() / run_name / "results")
    elif not args.results_root:
        args.results_root = str(DEFAULT_RESULTS_ROOT)

    args.run_name = run_name
    args.results_root = str(Path(args.results_root).expanduser().resolve())


def find_reference(root: str | Path, *, task_name: str, seed: int, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit reference does not exist: {path}")
        return path
    root_path = Path(root).expanduser().resolve()
    seed_dir = f"seed_{int(seed):03d}"
    candidates = [
        root_path / task_name / seed_dir / f"{task_name}_{seed_dir}.npz",
        root_path / task_name / seed_dir / f"{seed_dir}.npz",
        root_path / task_name / f"{seed_dir}.npz",
        root_path / f"{task_name}_{seed_dir}" / f"{task_name}_{seed_dir}.npz",
        root_path / f"{task_name}_{seed_dir}.npz",
    ]
    for pattern_dir in (root_path / task_name / seed_dir, root_path / f"{task_name}_{seed_dir}"):
        if pattern_dir.is_dir():
            candidates.extend(sorted(pattern_dir.glob("*.npz")))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"No reference npz found for task={task_name!r} seed={seed} under {root_path}. "
        "Pass --reference explicitly or place files under <root>/<task>/seed_000/."
    )


def bool_any(value: Any) -> bool:
    try:
        import torch

        if torch.is_tensor(value):
            return bool(value.detach().cpu().bool().any().item())
    except Exception:
        pass
    return bool(np.asarray(value).astype(bool).any())


def jsonable_info(info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in info.items():
        try:
            import torch

            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(value)
        out[key] = arr.tolist() if arr.shape != () else arr.item()
    return out


def padded_history(items: Deque[np.ndarray], *, length: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    valid = min(len(items), length)
    padded = length - valid
    mask = np.asarray([True] * padded + [False] * valid, dtype=bool)
    if valid:
        values = list(items)[-valid:]
    else:
        values = []
    while len(values) < length:
        values.insert(0, np.zeros((dim,), dtype=np.float32))
    return np.stack(values[-length:], axis=0).astype(np.float32), mask


def run_one(
    args: argparse.Namespace,
    *,
    task_name: str,
    seed: int,
    adapter: Optional[Any] = None,
    reference_override: str | Path = "",
    manifest_entry: Optional[Dict[str, Any]] = None,
) -> Path:
    run_name = str(getattr(args, "run_name", "")).strip()
    resolved_task_name, env_id, env_kwargs = resolve_task(task_name, args.env_id)
    explicit_reference = str(reference_override or getattr(args, "reference", ""))
    reference_path = find_reference(args.reference_root, task_name=resolved_task_name, seed=seed, explicit=explicit_reference)
    reference = load_reference(reference_path)
    if reference.task_name and reference.task_name != resolved_task_name:
        raise ValueError(
            f"Reference task mismatch: expected {resolved_task_name!r}, got {reference.task_name!r} "
            f"from {reference_path}."
        )
    if reference.seed is not None and int(reference.seed) != int(seed):
        raise ValueError(
            f"Reference seed mismatch: expected {seed}, got {reference.seed} from {reference_path}."
        )
    reference_5hz, resample_indices = resample_frames_nearest(
        reference.frames,
        source_fps=reference.fps,
        target_fps=float(args.target_fps),
    )
    ref_buffer = ReferenceBuffer(reference_5hz, pad_strategy="pad_last")

    env = make_maniskill3_env(
        task_name=task_name,
        env_id=args.env_id,
        shader=args.shader,
        num_envs=1,
        obs_mode=args.obs_mode,
    )
    logger = EvalLogger(args.results_root, task_name=resolved_task_name, seed=seed)
    if adapter is None:
        from evaluation.simpler.model_adapter import ModelAdapter

        adapter = ModelAdapter(
            config_path=args.config,
            checkpoint=args.checkpoint,
            device=args.device,
            precision=args.precision,
            sample_steps=args.sample_steps,
            seed=seed,
        )
    future_horizon = int(getattr(args, "future_horizon", 0) or adapter.future_steps)
    if future_horizon != adapter.future_steps:
        raise ValueError(
            f"--future-horizon={future_horizon} does not match policy future_steps={adapter.future_steps}."
        )

    info: Dict[str, Any] = {}
    terminated = False
    truncated = False
    success = False
    step_idx = 0
    try:
        obs, reset_info = reset_episode(env, seed, reset_by_episode_id=not args.no_reset_by_episode_id)
        camera_name = args.camera_name or default_camera_name(env)
        instruction = get_instruction(env)
        current_rgb = extract_rgb(obs, env=env, camera_name=camera_name)
        current_state = extract_bridge_state(env)

        obs_history = ObservationHistory(length=adapter.history_image_frames)
        state_history: Deque[np.ndarray] = deque(maxlen=adapter.history_state_frames)
        action_history: Deque[np.ndarray] = deque(maxlen=adapter.history_frames)
        obs_history.append(current_rgb)
        state_history.append(current_state)
        logger.append_frame(current_rgb)

        while True:
            if args.max_steps > 0 and step_idx >= args.max_steps:
                break
            try:
                future_4 = ref_buffer.get_future(step_idx=step_idx, horizon=future_horizon)
            except StopIteration:
                break

            past_4 = obs_history.get()
            image_mask = obs_history.padding_mask()
            past_states, state_mask = padded_history(
                state_history, length=adapter.history_state_frames, dim=7
            )
            past_actions, action_mask = padded_history(
                action_history, length=adapter.history_frames, dim=7
            )
            policy_action = adapter.predict_action(
                current_rgb=current_rgb,
                past_frames=past_4,
                future_frames=future_4,
                past_states=past_states,
                past_actions=past_actions,
                past_image_padding_mask=image_mask,
                past_state_padding_mask=state_mask,
                past_action_padding_mask=action_mask,
                instruction=instruction,
            )
            env_action = bridge_action_to_maniskill_action(policy_action)
            logger.append_policy_action_bridge(policy_action)
            logger.append_action(env_action)

            obs, reward, terminated_value, truncated_value, info = env.step(torch_action_for_env(env_action, env))
            terminated = bool_any(terminated_value)
            truncated = bool_any(truncated_value)
            success = bool_any(info.get("success", False))
            step_idx += 1
            if terminated or truncated:
                break

            action_history.append(policy_action)
            current_rgb = extract_rgb(obs, env=env, camera_name=camera_name)
            current_state = extract_bridge_state(env)
            obs_history.append(current_rgb)
            state_history.append(current_state)
            logger.append_frame(current_rgb)
    except Exception as exc:
        metrics = {
            "run_name": run_name or None,
            "task_name": resolved_task_name,
            "seed": int(seed),
            "success": bool(success),
            "episode_length": int(step_idx),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "exception": repr(exc),
            "info": jsonable_info(info) if isinstance(info, dict) else str(info),
            "policy_action_semantics": getattr(adapter, "action_semantics", "unknown"),
        }
        reference_info = {
            "run_name": run_name or None,
            "fixed_reference": True,
            "reference_path": str(reference_path),
            "reference_original_fps": float(reference.fps),
            "reference_resampled_fps": float(args.target_fps),
            "rollout_video_fps": float(args.target_fps),
            "reference_original_frames": int(reference.frames.shape[0]),
            "reference_resampled_frames": int(reference_5hz.shape[0]),
            "resample_indices": resample_indices.tolist(),
            "checkpoint": adapter.checkpoint_path,
            "checkpoint_epoch": adapter.checkpoint_epoch,
            "checkpoint_global_step": adapter.checkpoint_global_step,
            "policy_action_semantics": getattr(adapter, "action_semantics", "unknown"),
            "reference_window_start_offset": 1,
            "reference_tail_policy": "pad_last",
            "manifest_entry": manifest_entry or {},
        }
        out = logger.finalize(metrics=metrics, reference_info=reference_info, fps=int(round(float(args.target_fps))))
        raise RuntimeError(f"Rollout failed; partial logs saved to {out}") from exc
    finally:
        env.close()

    metrics = {
        "run_name": run_name or None,
        "task_name": resolved_task_name,
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "seed": int(seed),
        "success": bool(success),
        "episode_length": int(step_idx),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": jsonable_info(info) if isinstance(info, dict) else str(info),
        "policy_action_semantics": getattr(adapter, "action_semantics", "unknown"),
    }
    reference_info = {
        "run_name": run_name or None,
        "fixed_reference": True,
        "reference_path": str(reference_path),
        "reference_original_fps": float(reference.fps),
        "reference_resampled_fps": float(args.target_fps),
        "rollout_video_fps": float(args.target_fps),
        "reference_original_frames": int(reference.frames.shape[0]),
        "reference_resampled_frames": int(reference_5hz.shape[0]),
        "resample_indices": resample_indices.tolist(),
        "reference_task_name": reference.task_name,
        "reference_seed": reference.seed,
        "reference_instruction": reference.instruction,
        "rollout_instruction": instruction,
        "camera_name": camera_name,
        "model_checkpoint": adapter.checkpoint_path,
        "checkpoint_epoch": adapter.checkpoint_epoch,
        "checkpoint_global_step": adapter.checkpoint_global_step,
        "model_config": str(Path(args.config).expanduser().resolve()),
        "policy_action_semantics": getattr(adapter, "action_semantics", "unknown"),
        "future_horizon": int(future_horizon),
        "reference_window_start_offset": 1,
        "reference_tail_policy": "pad_last",
        "reference_last_distinct_policy_step": max(0, int(reference_5hz.shape[0]) - 2),
        "manifest_entry": manifest_entry or {},
    }
    return logger.finalize(metrics=metrics, reference_info=reference_info, fps=int(round(float(args.target_fps))))


def main() -> None:
    args = build_argparser().parse_args()
    configure_simulation()
    resolve_run_output_path(args)
    tasks = [part.strip() for part in args.tasks.split(",") if part.strip()] or [args.task_name]
    seeds = parse_seed_list(args.seeds, start_seed=args.start_seed, num_episodes=args.num_episodes)
    from evaluation.simpler.model_adapter import ModelAdapter

    adapter = ModelAdapter(
        config_path=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        precision=args.precision,
        sample_steps=args.sample_steps,
        seed=seeds[0] if seeds else 0,
    )
    outputs = []
    for task_name in tasks:
        for seed in seeds:
            out = run_one(args, task_name=task_name, seed=seed, adapter=adapter)
            outputs.append(str(out))
            print(out, flush=True)
    completed_metrics = []
    for output in outputs:
        metrics_path = Path(output) / "metrics.json"
        if metrics_path.is_file():
            completed_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    num_successes = sum(bool(item.get("success", False)) for item in completed_metrics)
    summary = Path(args.results_root).expanduser().resolve() / "rollout_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "run_name": args.run_name or None,
                "fixed_reference": True,
                "model_config": str(Path(args.config).expanduser().resolve()),
                "model_checkpoint": adapter.checkpoint_path,
                "checkpoint_epoch": adapter.checkpoint_epoch,
                "checkpoint_global_step": adapter.checkpoint_global_step,
                "policy_action_semantics": adapter.action_semantics,
                "reference_root": str(Path(args.reference_root).expanduser().resolve()),
                "results_root": args.results_root,
                "target_fps": float(args.target_fps),
                "future_horizon": int(args.future_horizon or adapter.future_steps),
                "reference_window_start_offset": 1,
                "reference_tail_policy": "pad_last",
                "num_episodes": len(completed_metrics),
                "num_successes": num_successes,
                "success_rate": (
                    num_successes / len(completed_metrics) if completed_metrics else None
                ),
                "outputs": outputs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
