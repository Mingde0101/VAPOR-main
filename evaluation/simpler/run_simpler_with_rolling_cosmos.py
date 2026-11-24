#!/usr/bin/env python
"""Closed-loop SIMPLER rollout with rolling Cosmos image/video references."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path
import sys
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from lang_slot.paths import data_path, resources
from evaluation.simpler.cosmos_reference_client import CosmosReferenceGenerator, write_rgb_png, write_rgb_video
from evaluation.simpler.eval_logger import EvalLogger
from evaluation.simpler.observation_history import ObservationHistory
from evaluation.simpler.reference_buffer import ReferenceBuffer
from evaluation.simpler.reference_loader import load_reference
from evaluation.simpler.reference_preprocess import drop_reference_prefix_frames, resample_frames_nearest
from evaluation.simpler.run_simpler_with_reference import bool_any, jsonable_info, padded_history
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
    step_with_rgb_capture,
    resolve_task,
    torch_action_for_env,
)


DEFAULT_RESULTS_ROOT = resources()["POLICY_OUTPUT_DIR"] / "simpler" / "rolling_cosmos_results"
DEFAULT_COSMOS_OUTPUT_ROOT = resources()["POLICY_OUTPUT_DIR"] / "simpler" / "rolling_cosmos_references"
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
    parser.add_argument(
        "--run-name",
        default="",
        help="Run identifier used to create <run-root>/<run-name>/{results,references}.",
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
        "--rollout-video-fps",
        type=float,
        default=10.0,
        help="FPS used for real SIMPLER observations supplied to V2V and saved to rollout video.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses env truncation.")
    parser.add_argument("--no-reset-by-episode-id", action="store_true")

    parser.add_argument(
        "--rollout-window-sec",
        type=float,
        default=0.0,
        help=(
            "Seconds of policy rollout per generated reference. A positive value overrides "
            "--rollout-chunk-steps and must contain an integer number of policy steps."
        ),
    )
    parser.add_argument("--rollout-chunk-steps", type=int, default=5, help="Policy steps to run per generated reference.")
    parser.add_argument("--future-horizon", type=int, default=4)
    parser.add_argument("--conditioning-frames", type=int, default=5, help="Real observation frames used for video2video.")
    parser.add_argument("--conditioning-fps", type=int, default=10)
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="Optional text prepended to the initial image2video prompt.",
    )
    parser.add_argument(
        "--video2video-prompt-prefix",
        default="Continue the task from the current robot state: ",
        help="Text prepended to the instruction for rolling video2video segments.",
    )

    parser.add_argument(
        "--cosmos-output-root",
        default="",
        help="Explicit reference directory. Overrides the directory derived from --run-name.",
    )
    parser.add_argument(
        "--cosmos-backend",
        default="vllm_omni",
        choices=["cosmos_framework", "vllm_omni"],
        help="Cosmos API backend. vllm_omni uses /v1/videos/sync; cosmos_framework uses /generate.",
    )
    parser.add_argument("--cosmos-host", default=os.getenv("COSMOS3_SERVE_HOST", "127.0.0.1"))
    parser.add_argument("--cosmos-port", type=int, default=int(os.getenv("COSMOS3_SERVE_PORT", "8001")))
    parser.add_argument("--cosmos-timeout", type=int, default=1800)
    parser.add_argument("--cosmos-model", default=os.getenv("COSMOS3_SERVED_MODEL") or Path(os.getenv("COSMOS3_MODEL_DIR") or "Cosmos3-Nano").name)
    parser.add_argument(
        "--cosmos-vllm-size",
        default="",
        help="vLLM-Omni video size such as 736x544; empty derives it from resolution/aspect ratio.",
    )
    parser.add_argument(
        "--cosmos-negative-prompt",
        default="",
        help="Negative prompt sent to vLLM-Omni; ignored by cosmos_framework.",
    )
    parser.add_argument("--cosmos-fps", type=int, default=10)
    parser.add_argument(
        "--cosmos-duration-sec",
        type=float,
        default=0.0,
        help=(
            "Requested Cosmos duration. Zero selects the minimum duration needed for one "
            "rollout window plus the future horizon after V2V prefix removal."
        ),
    )
    parser.add_argument("--cosmos-resolution", default="480", choices=["256", "480", "720", "768"])
    parser.add_argument("--cosmos-aspect-ratio", default="4,3", choices=["1,1", "4,3", "3,4", "16,9", "9,16"])
    parser.add_argument("--cosmos-num-steps", type=int, default=35)
    parser.add_argument("--cosmos-guidance", type=float, default=6.0)
    parser.add_argument("--cosmos-shift", type=float, default=10.0)
    parser.add_argument("--cosmos-sigma-max", type=float, default=80.0)
    parser.add_argument("--cosmos-normalize-cfg", action="store_true")
    parser.add_argument("--cosmos-prompt-upsampling", action="store_true")
    parser.add_argument(
        "--video2video-drop-prefix-frames",
        type=int,
        default=4,
        help="Drop echoed V2V conditioning frames before resampling the reference to policy FPS.",
    )
    return parser


def resolve_temporal_settings(args: argparse.Namespace) -> None:
    """Resolve a rollout window and a reference duration on their native timelines."""
    target_fps = float(args.target_fps)
    cosmos_fps = float(args.cosmos_fps)
    if target_fps <= 0:
        raise ValueError("--target-fps must be positive.")
    if cosmos_fps <= 0:
        raise ValueError("--cosmos-fps must be positive.")
    if int(args.future_horizon) <= 0:
        raise ValueError("--future-horizon must be positive.")
    if int(args.video2video_drop_prefix_frames) < 0:
        raise ValueError("--video2video-drop-prefix-frames cannot be negative.")

    requested_window_sec = float(args.rollout_window_sec)
    if requested_window_sec > 0:
        exact_steps = requested_window_sec * target_fps
        chunk_steps = int(round(exact_steps))
        if chunk_steps <= 0 or not math.isclose(exact_steps, chunk_steps, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"--rollout-window-sec={requested_window_sec} does not contain an integer "
                f"number of steps at --target-fps={target_fps}."
            )
        args.rollout_chunk_steps = chunk_steps
    elif int(args.rollout_chunk_steps) <= 0:
        raise ValueError("--rollout-chunk-steps must be positive.")

    args.rollout_window_sec = float(args.rollout_chunk_steps) / target_fps

    # Reference frame 0 is the current frame. At the final local policy step,
    # future indexing reaches chunk_steps + horizon - 1 on the 5 Hz timeline.
    # V2V additionally echoes the prefix that is removed on the Cosmos timeline.
    required_duration_sec = (
        float(args.video2video_drop_prefix_frames) / cosmos_fps
        + float(int(args.rollout_chunk_steps) + int(args.future_horizon) - 1) / target_fps
    )
    requested_duration_sec = float(args.cosmos_duration_sec)
    args.cosmos_duration_auto = requested_duration_sec <= 0
    if args.cosmos_duration_auto:
        args.cosmos_duration_sec = required_duration_sec
    elif requested_duration_sec + 1e-9 < required_duration_sec:
        raise ValueError(
            f"--cosmos-duration-sec={requested_duration_sec} is too short; this configuration "
            f"requires at least {required_duration_sec:.6g}s for "
            f"{args.rollout_chunk_steps} policy steps and horizon={args.future_horizon}."
        )
    args.required_cosmos_duration_sec = required_duration_sec
    args.required_reference_frames = int(args.rollout_chunk_steps) + int(args.future_horizon)


def resolve_run_output_paths(args: argparse.Namespace) -> None:
    """Resolve isolated result/reference directories while retaining legacy defaults."""
    run_name = str(args.run_name).strip()
    if run_name:
        if Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("--run-name must be a single path-safe directory name.")
        run_dir = Path(args.run_root).expanduser().resolve() / run_name
        if not args.results_root:
            args.results_root = str(run_dir / "results")
        if not args.cosmos_output_root:
            args.cosmos_output_root = str(run_dir / "references")
    else:
        if not args.results_root:
            args.results_root = str(DEFAULT_RESULTS_ROOT)
        if not args.cosmos_output_root:
            args.cosmos_output_root = str(DEFAULT_COSMOS_OUTPUT_ROOT)

    args.run_name = run_name
    args.results_root = str(Path(args.results_root).expanduser().resolve())
    args.cosmos_output_root = str(Path(args.cosmos_output_root).expanduser().resolve())


def padded_conditioning_frames(items: Deque[np.ndarray], *, length: int) -> List[np.ndarray]:
    if length <= 0:
        raise ValueError("conditioning length must be positive.")
    if not items:
        raise RuntimeError("No observations available for video2video conditioning.")
    frames = list(items)[-int(length) :]
    while len(frames) < int(length):
        frames.insert(0, frames[0])
    return [np.asarray(frame, dtype=np.uint8) for frame in frames[-int(length) :]]


def make_segment_prompt(*, instruction: str, segment_index: int, prompt_prefix: str, video2video_prompt_prefix: str) -> str:
    instruction = str(instruction).strip()
    if segment_index == 0:
        return f"{prompt_prefix}{instruction}".strip()
    return f"{video2video_prompt_prefix}{instruction}".strip()


def generate_reference_segment(
    *,
    args: argparse.Namespace,
    generator: CosmosReferenceGenerator,
    task_name: str,
    env_id: str,
    seed: int,
    camera_name: str,
    instruction: str,
    segment_index: int,
    rollout_step: int,
    current_rgb: np.ndarray,
    observation_cache: Deque[np.ndarray],
    segment_root: Path,
) -> Tuple[ReferenceBuffer, Dict[str, Any]]:
    segment_name = f"{task_name}_seed_{int(seed):03d}_seg_{int(segment_index):03d}"
    input_dir = segment_root / "conditioning_inputs" / segment_name
    input_dir.mkdir(parents=True, exist_ok=True)
    if segment_index == 0:
        model_mode = "image2video"
        vision_path = write_rgb_png(input_dir / "initial_rgb.png", current_rgb)
    else:
        model_mode = "video2video"
        conditioning = padded_conditioning_frames(observation_cache, length=int(args.conditioning_frames))
        vision_path = write_rgb_video(
            input_dir / "conditioning_obs.mp4",
            conditioning,
            fps=int(args.conditioning_fps),
            source_fps=float(args.rollout_video_fps),
        )

    prompt = make_segment_prompt(
        instruction=instruction,
        segment_index=segment_index,
        prompt_prefix=args.prompt_prefix,
        video2video_prompt_prefix=args.video2video_prompt_prefix,
    )
    sample = generator.build_sample(
        name=segment_name,
        model_mode=model_mode,
        prompt=prompt,
        vision_path=vision_path,
        seed=int(seed) + int(segment_index),
        extra={
            "task_name": task_name,
            "env_id": env_id,
            "seed": int(seed),
            "camera_name": camera_name,
            "instruction": instruction,
            "segment_index": int(segment_index),
            "rollout_step": int(rollout_step),
            "conditioning_fps": int(args.conditioning_fps),
            "policy_control_fps": float(args.target_fps),
            "observation_sample_fps": float(args.rollout_video_fps),
            "conditioning_saved_frames": 1 if segment_index == 0 else int(args.conditioning_frames),
            "conditioning_frames": 1 if segment_index == 0 else int(args.conditioning_frames),
            "source": "evaluation/simpler/run_simpler_with_rolling_cosmos.py",
        },
    )
    print(
        json.dumps(
            {
                "event": "cosmos_generate_start",
                "task_name": task_name,
                "seed": int(seed),
                "segment_index": int(segment_index),
                "rollout_step": int(rollout_step),
                "model_mode": model_mode,
                "vision_path": str(vision_path),
                "fps": int(sample["fps"]),
                "num_frames": int(sample["num_frames"]),
                "resolution": str(sample["resolution"]),
                "backend": str(getattr(generator, "backend", "unknown")),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    result = generator.submit(sample)
    reference = load_reference(result.npz_path)
    reference_frames = reference.frames
    dropped_prefix_frames = 0
    if model_mode == "video2video":
        reference_frames, dropped_prefix_frames = drop_reference_prefix_frames(
            reference.frames,
            num_prefix_frames=int(args.video2video_drop_prefix_frames),
        )
    reference_5hz, resample_indices = resample_frames_nearest(
        reference_frames,
        source_fps=reference.fps,
        target_fps=float(args.target_fps),
    )
    if reference_5hz.shape[0] < int(args.required_reference_frames):
        raise RuntimeError(
            f"Cosmos returned only {reference_5hz.shape[0]} usable frames at "
            f"{args.target_fps} Hz after prefix removal; at least "
            f"{args.required_reference_frames} are required for "
            f"window={args.rollout_window_sec}s and horizon={args.future_horizon}."
        )
    original_resample_indices = resample_indices + int(dropped_prefix_frames)
    metadata = {
        "segment_index": int(segment_index),
        "rollout_step": int(rollout_step),
        "model_mode": model_mode,
        "prompt": prompt,
        "vision_path": str(vision_path),
        "sample_dir": str(result.sample_dir),
        "npz_path": str(result.npz_path),
        "mp4_path": None if result.mp4_path is None else str(result.mp4_path),
        "compat_mp4_path": None if result.compat_mp4_path is None else str(result.compat_mp4_path),
        "cosmos_elapsed_sec": float(result.elapsed_sec),
        "requested_fps": int(sample["fps"]),
        "requested_num_frames": int(sample["num_frames"]),
        "requested_duration_sec": float(args.cosmos_duration_sec),
        "reference_original_fps": float(reference.fps),
        "reference_original_frames": int(reference.frames.shape[0]),
        "reference_drop_prefix_frames": int(dropped_prefix_frames),
        "reference_frames_after_prefix_drop": int(reference_frames.shape[0]),
        "reference_resampled_fps": float(args.target_fps),
        "reference_resampled_frames": int(reference_5hz.shape[0]),
        "required_reference_frames": int(args.required_reference_frames),
        "resample_indices": resample_indices.tolist(),
        "resample_indices_in_original_reference": original_resample_indices.tolist(),
    }
    print(
        json.dumps(
            {
                "event": "cosmos_generate_done",
                "task_name": task_name,
                "seed": int(seed),
                "segment_index": int(segment_index),
                "elapsed_sec": round(float(result.elapsed_sec), 3),
                "npz_path": str(result.npz_path),
                "reference_frames": int(reference.frames.shape[0]),
                "reference_drop_prefix_frames": int(dropped_prefix_frames),
                "reference_frames_after_prefix_drop": int(reference_frames.shape[0]),
                "reference_5hz_frames": int(reference_5hz.shape[0]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return ReferenceBuffer(reference_5hz, pad_strategy="pad_last"), metadata


def make_generator(args, output_root):
    return CosmosReferenceGenerator(
        output_root=output_root,
        host=args.cosmos_host,
        port=args.cosmos_port,
        timeout=args.cosmos_timeout,
        model=args.cosmos_model,
        fps=args.cosmos_fps,
        duration_sec=args.cosmos_duration_sec,
        resolution=args.cosmos_resolution,
        aspect_ratio=args.cosmos_aspect_ratio,
        num_steps=args.cosmos_num_steps,
        guidance=args.cosmos_guidance,
        shift=args.cosmos_shift,
        sigma_max=args.cosmos_sigma_max,
        normalize_cfg=args.cosmos_normalize_cfg,
        prompt_upsampling=args.cosmos_prompt_upsampling,
        backend=args.cosmos_backend,
        vllm_size=args.cosmos_vllm_size,
        negative_prompt=args.cosmos_negative_prompt,
    )


def run_one(
    args: argparse.Namespace,
    *,
    task_name: str,
    seed: int,
    adapter: Optional[Any] = None,
) -> Path:
    resolved_task_name, env_id, env_kwargs = resolve_task(task_name, args.env_id)
    env = make_maniskill3_env(
        task_name=task_name,
        env_id=args.env_id,
        shader=args.shader,
        num_envs=1,
        obs_mode=args.obs_mode,
    )
    logger = EvalLogger(args.results_root, task_name=resolved_task_name, seed=seed)
    segment_root = Path(args.cosmos_output_root).expanduser().resolve() / resolved_task_name / f"seed_{int(seed):03d}"
    generator = make_generator(args, segment_root / "references")
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

    info: Dict[str, Any] = {}
    terminated = False
    truncated = False
    success = False
    step_idx = 0
    segment_index = 0
    segments: List[Dict[str, Any]] = []
    instruction = ""
    camera_name = ""
    try:
        obs, reset_info = reset_episode(env, seed, reset_by_episode_id=not args.no_reset_by_episode_id)
        camera_name = args.camera_name or default_camera_name(env)
        instruction = get_instruction(env)
        current_rgb = extract_rgb(obs, env=env, camera_name=camera_name)
        current_state = extract_bridge_state(env)

        obs_history = ObservationHistory(length=4)
        state_history: Deque[np.ndarray] = deque(maxlen=4)
        action_history: Deque[np.ndarray] = deque(maxlen=4)
        observation_cache: Deque[np.ndarray] = deque(maxlen=max(1, int(args.conditioning_frames)))

        obs_history.append(current_rgb)
        state_history.append(current_state)
        observation_cache.append(current_rgb)
        logger.append_frame(current_rgb)

        while True:
            if args.max_steps > 0 and step_idx >= args.max_steps:
                break
            ref_buffer, segment_meta = generate_reference_segment(
                args=args,
                generator=generator,
                task_name=resolved_task_name,
                env_id=env_id,
                seed=seed,
                camera_name=camera_name,
                instruction=instruction,
                segment_index=segment_index,
                rollout_step=step_idx,
                current_rgb=current_rgb,
                observation_cache=observation_cache,
                segment_root=segment_root,
            )
            segments.append(segment_meta)

            for local_step in range(int(args.rollout_chunk_steps)):
                if args.max_steps > 0 and step_idx >= args.max_steps:
                    break
                future_4 = ref_buffer.get_future(step_idx=local_step, horizon=int(args.future_horizon))
                past_4 = obs_history.get()
                image_mask = obs_history.padding_mask()
                past_states, state_mask = padded_history(state_history, length=4, dim=7)
                past_actions, action_mask = padded_history(action_history, length=4, dim=7)
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

                obs, reward, terminated_value, truncated_value, info, captured_rgbs = step_with_rgb_capture(
                    env,
                    torch_action_for_env(env_action, env),
                    camera_name=camera_name,
                    capture_fps=float(args.rollout_video_fps),
                )
                terminated = bool_any(terminated_value)
                truncated = bool_any(truncated_value)
                success = bool_any(info.get("success", False))
                step_idx += 1
                for rgb in captured_rgbs:
                    logger.append_frame(rgb)
                    observation_cache.append(rgb)
                if terminated or truncated:
                    break

                action_history.append(policy_action)
                current_rgb = captured_rgbs[-1] if captured_rgbs else extract_rgb(obs, env=env, camera_name=camera_name)
                current_state = extract_bridge_state(env)
                obs_history.append(current_rgb)
                state_history.append(current_state)

            if terminated or truncated:
                break
            segment_index += 1
    except Exception as exc:
        metrics = {
            "run_name": args.run_name or None,
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
            "run_name": args.run_name or None,
            "rolling_cosmos": True,
            "segments": segments,
            "segment_root": str(segment_root),
            "checkpoint": adapter.checkpoint_path,
            "checkpoint_epoch": adapter.checkpoint_epoch,
            "checkpoint_global_step": adapter.checkpoint_global_step,
            "policy_action_semantics": getattr(adapter, "action_semantics", "unknown"),
            "rollout_chunk_steps": int(args.rollout_chunk_steps),
            "rollout_window_sec": float(args.rollout_window_sec),
            "conditioning_frames": int(args.conditioning_frames),
            "conditioning_fps": int(args.conditioning_fps),
            "policy_control_fps": float(args.target_fps),
            "observation_sample_fps": float(args.rollout_video_fps),
            "rollout_video_fps": float(args.rollout_video_fps),
            "cosmos_fps": int(args.cosmos_fps),
            "cosmos_backend": str(args.cosmos_backend),
            "cosmos_duration_sec": float(args.cosmos_duration_sec),
            "required_cosmos_duration_sec": float(args.required_cosmos_duration_sec),
            "cosmos_duration_auto": bool(args.cosmos_duration_auto),
            "required_reference_frames": int(args.required_reference_frames),
            "video2video_drop_prefix_frames": int(args.video2video_drop_prefix_frames),
        }
        out = logger.finalize(
            metrics=metrics,
            reference_info=reference_info,
            fps=int(round(float(args.rollout_video_fps))),
            source_fps=float(args.rollout_video_fps),
        )
        raise RuntimeError(f"Rolling Cosmos rollout failed; partial logs saved to {out}") from exc
    finally:
        env.close()

    metrics = {
        "run_name": args.run_name or None,
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
        "run_name": args.run_name or None,
        "rolling_cosmos": True,
        "segments": segments,
        "segment_root": str(segment_root),
        "rollout_instruction": instruction,
        "camera_name": camera_name,
        "model_checkpoint": adapter.checkpoint_path,
        "checkpoint_epoch": adapter.checkpoint_epoch,
        "checkpoint_global_step": adapter.checkpoint_global_step,
        "model_config": str(Path(args.config).expanduser().resolve()),
        "policy_action_semantics": getattr(adapter, "action_semantics", "unknown"),
        "rollout_chunk_steps": int(args.rollout_chunk_steps),
        "rollout_window_sec": float(args.rollout_window_sec),
        "future_horizon": int(args.future_horizon),
        "target_fps": float(args.target_fps),
        "policy_control_fps": float(args.target_fps),
        "rollout_video_fps": float(args.rollout_video_fps),
        "observation_sample_fps": float(args.rollout_video_fps),
        "conditioning_frames": int(args.conditioning_frames),
        "conditioning_fps": int(args.conditioning_fps),
        "cosmos_fps": int(args.cosmos_fps),
        "cosmos_backend": str(args.cosmos_backend),
        "cosmos_duration_sec": float(args.cosmos_duration_sec),
        "required_cosmos_duration_sec": float(args.required_cosmos_duration_sec),
        "cosmos_duration_auto": bool(args.cosmos_duration_auto),
        "required_reference_frames": int(args.required_reference_frames),
        "cosmos_resolution": str(args.cosmos_resolution),
        "cosmos_aspect_ratio": str(args.cosmos_aspect_ratio),
        "cosmos_num_steps": int(args.cosmos_num_steps),
        "video2video_drop_prefix_frames": int(args.video2video_drop_prefix_frames),
    }
    return logger.finalize(
        metrics=metrics,
        reference_info=reference_info,
        fps=int(round(float(args.rollout_video_fps))),
        source_fps=float(args.rollout_video_fps),
    )


def main() -> None:
    args = build_argparser().parse_args()
    configure_simulation()
    resolve_run_output_paths(args)
    resolve_temporal_settings(args)
    if args.conditioning_frames <= 0:
        raise ValueError("--conditioning-frames must be positive.")
    if abs(float(args.conditioning_fps) - float(args.rollout_video_fps)) > 1e-6:
        raise ValueError("--conditioning-fps must match --rollout-video-fps for real captured observations.")
    captures_per_policy_step = float(args.rollout_video_fps) / float(args.target_fps)
    rounded_captures = int(round(captures_per_policy_step))
    if rounded_captures < 1 or not math.isclose(
        captures_per_policy_step, rounded_captures, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("--rollout-video-fps must be an integer multiple of --target-fps.")
    real_frames_before_first_v2v = 1 + int(args.rollout_chunk_steps) * rounded_captures
    if real_frames_before_first_v2v < int(args.conditioning_frames):
        raise ValueError(
            "The rollout window is too short to collect the requested number of distinct "
            f"SAPIEN conditioning frames before the first V2V call: have "
            f"{real_frames_before_first_v2v}, need {args.conditioning_frames}."
        )
    tasks = [part.strip() for part in args.tasks.split(",") if part.strip()] or [args.task_name]
    seeds = parse_seed_list(args.seeds, start_seed=args.start_seed, num_episodes=args.num_episodes)
    make_generator(args, args.cosmos_output_root).check_server()
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
    summary = Path(args.results_root) / "rolling_rollout_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "run_name": args.run_name or None,
                "model_config": str(Path(args.config).expanduser().resolve()),
                "model_checkpoint": adapter.checkpoint_path,
                "checkpoint_epoch": adapter.checkpoint_epoch,
                "checkpoint_global_step": adapter.checkpoint_global_step,
                "policy_action_semantics": adapter.action_semantics,
                "results_root": args.results_root,
                "cosmos_output_root": args.cosmos_output_root,
                "rollout_window_sec": float(args.rollout_window_sec),
                "rollout_chunk_steps": int(args.rollout_chunk_steps),
                "future_horizon": int(args.future_horizon),
                "target_fps": float(args.target_fps),
                "conditioning_frames": int(args.conditioning_frames),
                "conditioning_fps": int(args.conditioning_fps),
                "cosmos_duration_sec": float(args.cosmos_duration_sec),
                "cosmos_duration_auto": bool(args.cosmos_duration_auto),
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
