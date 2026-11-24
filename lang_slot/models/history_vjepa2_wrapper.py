"""History-video V-JEPA2 wrapper without rolling-window slicing."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ..text import dtype_from_name
from ..vision import DEFAULT_VJEPA2_MODEL_DIR, VJEPA2Tokens, VJEPA2VideoEncoder


class VJEPA2HistoryVideoWrapper(nn.Module):
    """Encode a fixed history clip as one full V-JEPA2 video.

    This wrapper intentionally does not split the history clip into rolling
    2-frame windows. For the default ``history_frames=4`` and V-JEPA2
    ``tubelet_size=2``, the encoded temporal grid is ``T'=2``:

    ``video [B,4,H,W,3] -> frame_tokens [B,2,256,1024]``.
    """

    def __init__(
        self,
        *,
        history_frames: int = 4,
        vjepa2_model_dir: str = DEFAULT_VJEPA2_MODEL_DIR,
        tubelet_size: int = 2,
        image_size: int = 256,
        resize_shortest_edge: Optional[int] = 292,
        freeze_vjepa2: bool = False,
        vjepa2_torch_dtype: Optional[str] = None,
        local_files_only: bool = True,
        ignore_mismatched_sizes: Optional[bool] = None,
        vision_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if history_frames <= 0:
            raise ValueError("history_frames must be positive.")
        if tubelet_size <= 0:
            raise ValueError("tubelet_size must be positive.")
        self.history_frames = int(history_frames)

        if vision_encoder is None:
            vision_encoder = VJEPA2VideoEncoder(
                model_dir=vjepa2_model_dir,
                tubelet_size=tubelet_size,
                image_size=image_size,
                resize_shortest_edge=resize_shortest_edge,
                freeze=freeze_vjepa2,
                local_files_only=local_files_only,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                torch_dtype=dtype_from_name(vjepa2_torch_dtype),
            )
        self.vision_encoder = vision_encoder

    @staticmethod
    def _check_video(video: torch.Tensor) -> Tuple[int, int]:
        if video.ndim != 5:
            raise ValueError(
                f"video must be [B,T,H,W,3] or [B,T,3,H,W], got {tuple(video.shape)}."
            )
        if video.shape[2] != 3 and video.shape[-1] != 3:
            raise ValueError(f"Could not infer channel axis for video shape {tuple(video.shape)}.")
        return int(video.shape[0]), int(video.shape[1])

    def select_history(self, video: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        """Return ``history_frames`` contiguous frames without rolling slicing."""
        _, time = self._check_video(video)
        start = int(start_index)
        end = start + self.history_frames
        if start < 0:
            raise ValueError("start_index must be non-negative.")
        if end > time:
            raise ValueError(
                f"Need {self.history_frames} history frames from start_index={start}, "
                f"but video only has {time} frames."
            )
        return video[:, start:end]

    def encode_history(
        self,
        video: torch.Tensor,
        *,
        start_index: int = 0,
        skip_predictor: bool = True,
    ) -> VJEPA2Tokens:
        history_video = self.select_history(video, start_index=start_index)
        return self.vision_encoder(history_video, skip_predictor=skip_predictor)

    def forward(
        self,
        video: torch.Tensor,
        *,
        start_index: int = 0,
        skip_predictor: bool = True,
    ) -> Dict[str, torch.Tensor]:
        history_video = self.select_history(video, start_index=start_index)
        encoded = self.vision_encoder(history_video, skip_predictor=skip_predictor)
        return {
            "history_video": history_video,
            "frame_tokens": encoded.frame_tokens,
            "video_tokens": encoded.video_tokens,
            "grid_shape": encoded.grid_shape,
        }
