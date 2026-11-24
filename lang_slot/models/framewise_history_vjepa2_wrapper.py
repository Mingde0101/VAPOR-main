"""Framewise history V-JEPA2 wrapper for policy conditioning."""

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


class VJEPA2FramewiseHistoryWrapper(nn.Module):
    """Encode each history frame as an independent duplicated 2-frame clip.

    This wrapper keeps V-JEPA2's pretrained ``tubelet_size=2`` temporal patch
    kernel while returning one token step per selected history image:

    ``video [B,T,H,W,3] -> windows [B*T,2,H,W,3] -> frame_tokens [B,T,N,D]``.

    It is intended for policy/action conditioning where every history image
    should keep its own time step instead of being compressed by full-clip
    tubelets.
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
        window_batch_size: Optional[int] = None,
        local_files_only: bool = True,
        ignore_mismatched_sizes: Optional[bool] = None,
        vision_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if history_frames <= 0:
            raise ValueError("history_frames must be positive.")
        if tubelet_size != 2 and vision_encoder is None:
            raise ValueError(
                "Framewise history encoding duplicates each image into a 2-frame clip, "
                "so the default wrapper expects pretrained tubelet_size=2. Pass a "
                "custom vision_encoder if you intentionally want another temporal kernel."
            )
        self.history_frames = int(history_frames)
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

    def select_history(self, video: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        """Return ``history_frames`` contiguous frames before framewise encoding."""
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

    @staticmethod
    def _make_single_frame_windows(history_video: torch.Tensor) -> torch.Tensor:
        # Duplicate every selected image into a 2-frame clip. Shape:
        # [B,T,...] -> [B,T,2,...] -> [B*T,2,...].
        batch_size, steps = history_video.shape[:2]
        windows = torch.stack((history_video, history_video), dim=2)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
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
                "Framewise history encoding expects one V-JEPA2 token step per duplicated "
                f"2-frame clip; got T'={frame_tokens.shape[1]}."
            )

        _, _, num_patches, input_dim = frame_tokens.shape
        history_frame_tokens = frame_tokens[:, 0].reshape(batch_size, steps, num_patches, input_dim)
        history_video_tokens = history_frame_tokens.reshape(batch_size, steps * num_patches, input_dim)
        _, h_prime, w_prime = encoded.grid_shape
        return history_frame_tokens, history_video_tokens, (steps, h_prime, w_prime)

    def encode_history(
        self,
        video: torch.Tensor,
        *,
        start_index: int = 0,
        skip_predictor: bool = True,
    ) -> VJEPA2Tokens:
        history_video = self.select_history(video, start_index=start_index)
        batch_size, steps = history_video.shape[:2]
        windows = self._make_single_frame_windows(history_video)
        encoded = self._encode_windows(windows, skip_predictor=skip_predictor)
        frame_tokens, video_tokens, grid_shape = self._reshape_window_tokens(
            encoded,
            batch_size=batch_size,
            steps=steps,
        )
        return VJEPA2Tokens(frame_tokens=frame_tokens, video_tokens=video_tokens, grid_shape=grid_shape)

    def forward(
        self,
        video: torch.Tensor,
        *,
        start_index: int = 0,
        history_padding_mask: Optional[torch.Tensor] = None,
        skip_predictor: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, input_time = self._check_video(video)
        history_video = self.select_history(video, start_index=start_index)
        _, steps = history_video.shape[:2]
        windows = self._make_single_frame_windows(history_video)
        encoded = self._encode_windows(windows, skip_predictor=skip_predictor)
        frame_tokens, video_tokens, grid_shape = self._reshape_window_tokens(
            encoded,
            batch_size=batch_size,
            steps=steps,
        )
        outputs = {
            "history_video": history_video,
            "history_frame_windows": windows,
            "frame_tokens": frame_tokens,
            "spatial_tokens": frame_tokens_to_spatial_grid(frame_tokens, grid_shape),
            "video_tokens": video_tokens,
            "grid_shape": grid_shape,
            "window_grid_shape": encoded.grid_shape,
        }
        if history_padding_mask is not None:
            if history_padding_mask.shape != (batch_size, input_time):
                raise ValueError(
                    "history_padding_mask must match input video [B,T], got "
                    f"{tuple(history_padding_mask.shape)} for video {tuple(video.shape)}."
                )
            if history_padding_mask.device != video.device:
                raise ValueError("history_padding_mask and video must be on the same device.")
            history_padding_mask = history_padding_mask.to(dtype=torch.bool)
            start = int(start_index)
            history_padding_mask = history_padding_mask[:, start : start + steps]
            num_patches = frame_tokens.shape[2]
            frame_token_padding_mask = history_padding_mask.unsqueeze(-1).expand(
                batch_size, steps, num_patches
            )
            outputs["history_padding_mask"] = history_padding_mask
            outputs["frame_token_padding_mask"] = frame_token_padding_mask
            outputs["video_token_padding_mask"] = frame_token_padding_mask.reshape(
                batch_size, steps * num_patches
            )
        return outputs
