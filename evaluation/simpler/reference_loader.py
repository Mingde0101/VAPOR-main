"""Load Cosmos reference videos saved as NPZ frame bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ReferenceData:
    frames: np.ndarray
    fps: float
    task_name: Optional[str] = None
    seed: Optional[int] = None
    instruction: Optional[str] = None
    path: Optional[str] = None


def _scalar_string(data: np.lib.npyio.NpzFile, key: str) -> Optional[str]:
    if key not in data:
        return None
    value = data[key]
    if value.shape == ():
        return str(value.item())
    if value.size == 0:
        return None
    return str(value.reshape(-1)[0])


def _scalar_int(data: np.lib.npyio.NpzFile, key: str) -> Optional[int]:
    if key not in data:
        return None
    value = data[key]
    if value.shape == ():
        item = value.item()
    elif value.size > 0:
        item = value.reshape(-1)[0]
    else:
        return None
    ivalue = int(item)
    return None if ivalue < 0 else ivalue


def load_reference_npz(path: str | Path) -> ReferenceData:
    ref_path = Path(path).expanduser().resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing Cosmos reference npz: {ref_path}")
    with np.load(ref_path, allow_pickle=False) as data:
        if "frames" not in data:
            raise KeyError(f"{ref_path} must contain a 'frames' array.")
        frames = np.asarray(data["frames"])
        if "fps" not in data:
            raise KeyError(f"{ref_path} must contain an 'fps' scalar.")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        task_name = _scalar_string(data, "task_name")
        seed = _scalar_int(data, "seed")
        instruction = _scalar_string(data, "instruction") or _scalar_string(data, "prompt")

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Reference frames must be [T,H,W,3], got {frames.shape}.")
    if frames.shape[0] <= 0:
        raise ValueError("Reference must contain at least one frame.")
    if fps <= 0:
        raise ValueError(f"Reference fps must be positive, got {fps}.")
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating) and frames.max(initial=0.0) <= 1.0:
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        else:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
    return ReferenceData(
        frames=np.ascontiguousarray(frames),
        fps=fps,
        task_name=task_name,
        seed=seed,
        instruction=instruction,
        path=str(ref_path),
    )


def load_reference(path: str | Path) -> ReferenceData:
    ref_path = Path(path).expanduser().resolve()
    if ref_path.suffix.lower() != ".npz":
        raise ValueError(f"Only .npz references are supported in this first version, got {ref_path}.")
    return load_reference_npz(ref_path)
