"""Future-frame V-JEPA2 wrapper for no-slot policy ablations."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ..text import dtype_from_name
from ..vision import (
    DEFAULT_VJEPA2_MODEL_DIR,
    VJEPA2Tokens,
    VJEPA2VideoEncoder,
    frame_tokens_to_spatial_grid,
)


class VJEPA2FutureVideoWrapper(nn.Module):
    """Encode future frames as adjacent two-frame V-JEPA2 windows.

    This matches the policy slot path's ``sliding_pair`` temporal alignment
    without running Slot Attention. ``T`` raw future frames become ``T-1``
    condition steps, e.g. ``t..t+4 -> 4`` future-video token grids.
    """

    def __init__(
        self,
        *,
        future_steps: int = 4,
        vjepa2_model_dir: str = DEFAULT_VJEPA2_MODEL_DIR,
        tubelet_size: int = 2,
        image_size: int = 256,
        resize_shortest_edge: Optional[int] = 292,
        freeze_vjepa2: bool = False,
        vjepa2_torch_dtype: Optional[str] = None,
        window_batch_size: Optional[int] = None,
        local_files_only: bool = True,
        ignore_mismatched_sizes: Optional[bool] = None,
        vision_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if future_steps <= 0:
            raise ValueError("future_steps must be positive.")
        if tubelet_size != 2 and vision_encoder is None:
            raise ValueError(
                "Future video encoding uses adjacent two-frame clips, so the default "
                "wrapper expects pretrained tubelet_size=2. Pass a custom vision_encoder "
                "if you intentionally want another temporal kernel."
            )
        self.future_steps = int(future_steps)
        self.window_batch_size = None if window_batch_size is None else int(window_batch_size)
        if self.window_batch_size is not None and self.window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive when provided.")

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

    @staticmethod
    def _make_sliding_pair_windows(video: torch.Tensor) -> torch.Tensor:
        batch_size, steps = int(video.shape[0]), int(video.shape[1]) - 1
        windows = torch.stack((video[:, :-1], video[:, 1:]), dim=2)
        return windows.reshape(batch_size * steps, *windows.shape[2:])

    def _encode_windows(self, windows: torch.Tensor, *, skip_predictor: bool) -> VJEPA2Tokens:
        if self.window_batch_size is None or windows.shape[0] <= self.window_batch_size:
            return self.vision_encoder(windows, skip_predictor=skip_predictor)

        frame_chunks = []
        video_chunks = []
        grid_shape = None
        for chunk in windows.split(self.window_batch_size, dim=0):
            encoded = self.vision_encoder(chunk, skip_predictor=skip_predictor)
            if grid_shape is None:
                grid_shape = encoded.grid_shape
            elif encoded.grid_shape != grid_shape:
                raise RuntimeError(f"Inconsistent V-JEPA2 grid shapes: {encoded.grid_shape} vs {grid_shape}.")
            frame_chunks.append(encoded.frame_tokens)
            video_chunks.append(encoded.video_tokens)
        assert grid_shape is not None
        return VJEPA2Tokens(
            frame_tokens=torch.cat(frame_chunks, dim=0),
            video_tokens=torch.cat(video_chunks, dim=0),
            grid_shape=grid_shape,
        )

    @staticmethod
    def _reshape_window_tokens(
        encoded: VJEPA2Tokens,
        *,
        batch_size: int,
        steps: int,
    ) -> torch.Tensor:
        frame_tokens = encoded.frame_tokens
        if frame_tokens.ndim != 4:
            raise ValueError(
                f"vision_encoder.frame_tokens must be [B_window,T',N,D], got {tuple(frame_tokens.shape)}."
            )
        if frame_tokens.shape[0] != batch_size * steps:
            raise ValueError(
                f"Expected {batch_size * steps} encoded windows, got {frame_tokens.shape[0]}."
            )
        if frame_tokens.shape[1] != 1:
            raise ValueError(
                "Future video window encoding expects one V-JEPA2 token step per "
                f"2-frame clip; got T'={frame_tokens.shape[1]}."
            )
        _, _, num_patches, input_dim = frame_tokens.shape
        return frame_tokens[:, 0].reshape(batch_size, steps, num_patches, input_dim)

    def forward(
        self,
        video: torch.Tensor,
        *,
        video_padding_mask: Optional[torch.Tensor] = None,
        skip_predictor: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, input_time = self._check_video(video)
        if input_time != self.future_steps + 1:
            raise ValueError(
                f"Expected future_steps+1={self.future_steps + 1} frames, got {input_time}."
            )
        windows = self._make_sliding_pair_windows(video)
        encoded = self._encode_windows(windows, skip_predictor=skip_predictor)
        frame_tokens = self._reshape_window_tokens(
            encoded,
            batch_size=batch_size,
            steps=self.future_steps,
        )
        grid_shape = (self.future_steps, encoded.grid_shape[1], encoded.grid_shape[2])

        outputs = {
            "frame_tokens": frame_tokens,
            "spatial_tokens": frame_tokens_to_spatial_grid(frame_tokens, grid_shape),
            "grid_shape": grid_shape,
            "window_grid_shape": encoded.grid_shape,
        }
        if video_padding_mask is not None:
            if video_padding_mask.shape != (batch_size, input_time):
                raise ValueError(
                    "video_padding_mask must match input video [B,T], got "
                    f"{tuple(video_padding_mask.shape)} for video {tuple(video.shape)}."
                )
            if video_padding_mask.device != video.device:
                raise ValueError("video_padding_mask and video must be on the same device.")
            encoded_padding_mask = video_padding_mask.to(dtype=torch.bool)[:, :-1] | video_padding_mask.to(
                dtype=torch.bool
            )[:, 1:]
            num_patches = frame_tokens.shape[2]
            frame_token_padding_mask = encoded_padding_mask.unsqueeze(-1).expand(
                batch_size,
                self.future_steps,
                num_patches,
            )
            outputs["frame_padding_mask"] = encoded_padding_mask
            outputs["frame_token_padding_mask"] = frame_token_padding_mask
            outputs["video_token_padding_mask"] = frame_token_padding_mask.reshape(
                batch_size,
                self.future_steps * num_patches,
            )
        return outputs
