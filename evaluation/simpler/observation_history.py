"""Past-observation buffer for SIMPLER rollout."""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np


class ObservationHistory:
    def __init__(self, *, length: int = 4, pad_strategy: str = "repeat_first") -> None:
        if length <= 0:
            raise ValueError("length must be positive.")
        if pad_strategy != "repeat_first":
            raise ValueError("Only repeat_first padding is implemented.")
        self.length = int(length)
        self.pad_strategy = pad_strategy
        self._frames: Deque[np.ndarray] = deque(maxlen=self.length)

    def append(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"frame must be [H,W,3], got {frame.shape}.")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._frames.append(np.ascontiguousarray(frame))

    def get(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("ObservationHistory is empty; append at least one frame first.")
        frames: List[np.ndarray] = list(self._frames)
        while len(frames) < self.length:
            frames.insert(0, frames[0])
        return np.stack(frames[-self.length :], axis=0)

    def padding_mask(self) -> np.ndarray:
        """True marks repeated-prefix padding, matching Bridge policy windows."""
        valid = min(len(self._frames), self.length)
        padded = self.length - valid
        return np.asarray([True] * padded + [False] * valid, dtype=bool)
