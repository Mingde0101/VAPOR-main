"""Future-frame sliding-window buffer for 5 Hz reference videos."""

from __future__ import annotations

import numpy as np


class ReferenceBuffer:
    def __init__(self, frames: np.ndarray, *, pad_strategy: str = "pad_last") -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}.")
        if frames.shape[0] <= 0:
            raise ValueError("ReferenceBuffer requires at least one frame.")
        if pad_strategy not in {"pad_last", "raise", "stop"}:
            raise ValueError(f"Unknown pad_strategy={pad_strategy!r}.")
        self.frames = np.ascontiguousarray(frames)
        self.pad_strategy = pad_strategy

    def get_future(self, *, step_idx: int, horizon: int = 4) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        start = int(step_idx) + 1
        indices = np.arange(start, start + int(horizon), dtype=np.int64)
        if indices[-1] >= self.frames.shape[0]:
            if self.pad_strategy == "raise":
                raise IndexError(
                    f"Reference too short for step_idx={step_idx}, horizon={horizon}, T={self.frames.shape[0]}."
                )
            if self.pad_strategy == "stop":
                raise StopIteration
            indices = np.clip(indices, 0, self.frames.shape[0] - 1)
        return np.ascontiguousarray(self.frames[indices])
