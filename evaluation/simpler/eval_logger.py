"""Minimal rollout logging for SIMPLER reference-conditioned evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def write_task_summary(task_dir: str | Path) -> Path:
    """Aggregate completed episode metrics under one task directory."""
    task_path = Path(task_dir).expanduser().resolve()
    metric_paths = sorted(task_path.glob("seed_*/metrics.json"))
    episodes = []
    task_names = set()
    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("task_name"):
            task_names.add(str(metrics["task_name"]))
        seed = int(metrics.get("seed", metrics_path.parent.name.rsplit("_", 1)[-1]))
        episodes.append(
            {
                "seed": seed,
                "success": bool(metrics.get("success", False)),
                "episode_length": int(metrics.get("episode_length", 0)),
                "terminated": bool(metrics.get("terminated", False)),
                "truncated": bool(metrics.get("truncated", False)),
                "has_exception": bool(metrics.get("exception")),
            }
        )
    episodes.sort(key=lambda item: item["seed"])
    if len(task_names) > 1:
        raise ValueError(f"Conflicting task names under {task_path}: {sorted(task_names)}")
    successful = [item for item in episodes if item["success"]]
    total = len(episodes)
    payload = {
        "task_name": next(iter(task_names)) if task_names else task_path.name,
        "num_episodes": total,
        "num_successes": len(successful),
        "success_rate": float(len(successful) / total) if total else 0.0,
        "evaluated_seeds": [item["seed"] for item in episodes],
        "success_seeds": [item["seed"] for item in successful],
        "successful_episodes": [
            {"seed": item["seed"], "episode_length": item["episode_length"]} for item in successful
        ],
        "success_steps_by_seed": {
            f"{item['seed']:03d}": item["episode_length"] for item in successful
        },
        "failed_seeds": [item["seed"] for item in episodes if not item["success"]],
        "episode_steps_by_seed": {
            f"{item['seed']:03d}": item["episode_length"] for item in episodes
        },
        "terminated_count": sum(item["terminated"] for item in episodes),
        "truncated_count": sum(item["truncated"] for item in episodes),
        "exception_count": sum(item["has_exception"] for item in episodes),
    }
    output_path = task_path / "task_summary.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def resample_rgb_frames_for_video(frames: List[np.ndarray], *, source_fps: float, target_fps: float) -> List[np.ndarray]:
    if not frames:
        return []
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"source_fps and target_fps must be positive, got {source_fps}, {target_fps}.")
    if abs(float(source_fps) - float(target_fps)) < 1e-6:
        return [np.asarray(frame, dtype=np.uint8) for frame in frames]
    target_count = max(1, int(round(len(frames) * float(target_fps) / float(source_fps))))
    indices = np.floor(np.arange(target_count, dtype=np.float64) * float(source_fps) / float(target_fps)).astype(np.int64)
    indices = np.clip(indices, 0, len(frames) - 1)
    return [np.asarray(frames[int(index)], dtype=np.uint8) for index in indices]


def _as_even_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Video frames must be [H,W,3] uint8 RGB, got {arr.shape}.")
    height, width = arr.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height <= 0 or even_width <= 0:
        raise ValueError(f"Video frame is too small: {arr.shape}.")
    if even_height != height or even_width != width:
        arr = arr[:even_height, :even_width]
    return np.ascontiguousarray(arr)


def write_compatible_mp4(path: Path, frames: List[np.ndarray], *, fps: int) -> None:
    import imageio.v2 as imageio

    video_frames = [_as_even_rgb(frame) for frame in frames]
    imageio.mimsave(
        path,
        video_frames,
        fps=int(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_params=["-movflags", "+faststart"],
    )


class EvalLogger:
    def __init__(self, root: str | Path, *, task_name: str, seed: int, rollout_index: int | None = None) -> None:
        self.output_dir = Path(root).expanduser().resolve() / task_name / f"seed_{int(seed):03d}"
        self.rollout_index = int(seed if rollout_index is None else rollout_index)
        self.rollout_video_name = f"rollout_{self.rollout_index:03d}.mp4"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.policy_actions_bridge: List[np.ndarray] = []

    def append_frame(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8))

    def append_action(self, action: np.ndarray) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32))

    def append_policy_action_bridge(self, action: np.ndarray) -> None:
        self.policy_actions_bridge.append(np.asarray(action, dtype=np.float32))

    def finalize(
        self,
        *,
        metrics: Dict[str, Any],
        reference_info: Dict[str, Any],
        fps: int = 10,
        source_fps: float | None = None,
    ) -> Path:
        if self.actions:
            np.save(self.output_dir / "actions.npy", np.stack(self.actions, axis=0))
        else:
            np.save(self.output_dir / "actions.npy", np.zeros((0, 7), dtype=np.float32))
        if self.policy_actions_bridge:
            np.save(self.output_dir / "policy_actions_bridge.npy", np.stack(self.policy_actions_bridge, axis=0))
        else:
            np.save(self.output_dir / "policy_actions_bridge.npy", np.zeros((0, 7), dtype=np.float32))
        saved_frames = self.frames
        if self.frames and source_fps is not None:
            saved_frames = resample_rgb_frames_for_video(
                self.frames,
                source_fps=float(source_fps),
                target_fps=float(fps),
            )
        if saved_frames:
            write_compatible_mp4(self.output_dir / self.rollout_video_name, saved_frames, fps=fps)
        reference_info = dict(reference_info)
        reference_info["rollout_video_filename"] = self.rollout_video_name
        reference_info["rollout_video_path"] = str(self.output_dir / self.rollout_video_name)
        reference_info["rollout_video_saved_frames"] = len(saved_frames)
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "reference_info.json").write_text(
            json.dumps(reference_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_task_summary(self.output_dir.parent)
        return self.output_dir
