"""Time and model-input alignment for Cosmos reference frames."""

from __future__ import annotations

import numpy as np


def resample_frames_nearest(
    frames: np.ndarray,
    *,
    source_fps: float,
    target_fps: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor timestamp resampling from ``source_fps`` to ``target_fps``."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}.")
    if frames.shape[0] <= 0:
        raise ValueError("frames must be non-empty.")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"source_fps and target_fps must be positive, got {source_fps}, {target_fps}.")
    duration = (frames.shape[0] - 1) / float(source_fps)
    target_times = np.arange(0.0, duration + 1e-9, 1.0 / float(target_fps), dtype=np.float64)
    indices = np.rint(target_times * float(source_fps)).astype(np.int64)
    indices = np.clip(indices, 0, frames.shape[0] - 1)
    return np.ascontiguousarray(frames[indices]), indices


def drop_reference_prefix_frames(frames: np.ndarray, *, num_prefix_frames: int) -> tuple[np.ndarray, int]:
    """Drop known conditioning frames echoed by a generated V2V reference."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}.")
    if frames.shape[0] <= 0:
        raise ValueError("frames must be non-empty.")
    drop = int(num_prefix_frames)
    if drop < 0:
        raise ValueError(f"num_prefix_frames must be non-negative, got {drop}.")
    if drop == 0:
        return np.ascontiguousarray(frames), 0
    if frames.shape[0] <= drop:
        raise ValueError(f"Cannot drop {drop} prefix frames from only {frames.shape[0]} reference frames.")
    return np.ascontiguousarray(frames[drop:]), drop


def to_policy_image_tensor(frames: np.ndarray):
    """Return raw uint8 NHWC frames for the existing policy image pipeline.

    The policy's V-JEPA2 encoder performs the actual training-time image
    transform internally: uint8/255, resize shortest edge, center crop, and
    ImageNet normalization. Keeping this function thin avoids duplicating that
    transform in benchmark code.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    import torch

    return torch.from_numpy(np.ascontiguousarray(frames))
