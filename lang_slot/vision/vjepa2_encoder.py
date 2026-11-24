"""Online V-JEPA2 token encoder wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import torch
from torch import nn

from lang_slot.paths import data_path
from torch.nn import functional as F


DEFAULT_VJEPA2_MODEL_DIR = str(data_path("pretrained_models", "vjepa2-vitl-fpc64-256"))


def frame_tokens_to_spatial_grid(
    frame_tokens: torch.Tensor,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Reshape V-JEPA2 frame tokens to a spatial feature map.

    Args:
        frame_tokens: ``[B,T,N,D]`` tokens from ``VJEPA2VideoEncoder`` or a
            window wrapper.
        grid_shape: ``(T,H,W)`` metadata returned with those tokens.

    Returns:
        ``[B,T,H,W,D]`` for ``ConditionAggregationTransformer``.
    """
    if frame_tokens.ndim != 4:
        raise ValueError(
            f"frame_tokens must be [B,T,N,D], got {tuple(frame_tokens.shape)}."
        )
    if len(grid_shape) != 3:
        raise ValueError(f"grid_shape must be (T,H,W), got {grid_shape}.")
    t_prime, h_prime, w_prime = (int(x) for x in grid_shape)
    if min(t_prime, h_prime, w_prime) <= 0:
        raise ValueError(f"grid_shape entries must be positive, got {grid_shape}.")
    if frame_tokens.shape[1] != t_prime:
        raise ValueError(
            f"Token time dimension {frame_tokens.shape[1]} does not match grid_shape T={t_prime}."
        )
    expected_patches = h_prime * w_prime
    if frame_tokens.shape[2] != expected_patches:
        raise ValueError(
            f"Token patch dimension {frame_tokens.shape[2]} does not match H*W={expected_patches} "
            f"for grid_shape={grid_shape}."
        )
    batch_size, _, _, dim = frame_tokens.shape
    return frame_tokens.reshape(batch_size, t_prime, h_prime, w_prime, dim)


@dataclass
class VJEPA2Tokens:
    """V-JEPA2 tokens with explicit temporal/spatial grid metadata."""

    frame_tokens: torch.Tensor
    video_tokens: torch.Tensor
    grid_shape: Tuple[int, int, int]


def _to_channels_first(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"video must be [B,T,H,W,3] or [B,T,3,H,W], got {tuple(video.shape)}.")
    if video.shape[2] == 3:
        return video
    if video.shape[-1] == 3:
        return video.permute(0, 1, 4, 2, 3)
    raise ValueError(f"Could not infer channel axis for video shape {tuple(video.shape)}.")


def _center_crop(video: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, _, _, height, width = video.shape
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Cannot center-crop {crop_size} from video spatial size {(height, width)}."
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return video[:, :, :, top : top + crop_size, left : left + crop_size]


def preprocess_video_tensor(
    video: torch.Tensor,
    image_size: int = 256,
    resize_shortest_edge: Optional[int] = 292,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Preprocess raw Bridge frames for HF V-JEPA2.

    Input can be uint8 ``[B,T,H,W,3]`` from TFDS or float channels-first
    ``[B,T,3,H,W]``. Output is float ``[B,T,3,image_size,image_size]``.
    """
    video = _to_channels_first(video)
    if not torch.is_floating_point(video):
        video = video.float()
    if video.max() > 2.0:
        video = video / 255.0

    batch, time, channels, height, width = video.shape
    flat = video.reshape(batch * time, channels, height, width)

    if resize_shortest_edge is not None:
        if height <= width:
            new_height = resize_shortest_edge
            new_width = round(width * resize_shortest_edge / height)
        else:
            new_width = resize_shortest_edge
            new_height = round(height * resize_shortest_edge / width)
        flat = F.interpolate(flat, size=(new_height, new_width), mode="bilinear", align_corners=False)

    video = flat.reshape(batch, time, channels, flat.shape[-2], flat.shape[-1])
    video = _center_crop(video, image_size)

    mean_t = torch.tensor(mean, device=video.device, dtype=video.dtype).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(std, device=video.device, dtype=video.dtype).view(1, 1, 3, 1, 1)
    return (video - mean_t) / std_t


class VJEPA2VideoEncoder(nn.Module):
    """Wrap HF V-JEPA2 and expose tokens as ``[B,T',N,D]``.

    ``tubelet_size`` defaults to 2 to match the local pretrained checkpoint.
    Overriding it is supported, but changing it will reinitialize the Conv3D
    patch embedding because the temporal kernel shape no longer matches.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_VJEPA2_MODEL_DIR,
        tubelet_size: int = 2,
        image_size: int = 256,
        resize_shortest_edge: Optional[int] = 292,
        freeze: bool = False,
        local_files_only: bool = True,
        ignore_mismatched_sizes: Optional[bool] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if tubelet_size <= 0:
            raise ValueError("tubelet_size must be positive.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("VJEPA2VideoEncoder requires transformers.") from exc

        config = AutoConfig.from_pretrained(model_dir, local_files_only=local_files_only)
        original_tubelet_size = int(config.tubelet_size)
        original_crop_size = int(config.crop_size)
        config.tubelet_size = int(tubelet_size)
        config.crop_size = int(image_size)
        config.image_size = int(image_size)

        if ignore_mismatched_sizes is None:
            ignore_mismatched_sizes = (
                int(tubelet_size) != original_tubelet_size or int(image_size) != original_crop_size
            )

        self.model = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=local_files_only,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            torch_dtype=torch_dtype,
        )
        self.image_size = int(image_size)
        self.resize_shortest_edge = resize_shortest_edge
        self.tubelet_size = int(tubelet_size)
        self.patch_size = int(config.patch_size)
        self.hidden_size = int(config.hidden_size)
        self.original_tubelet_size = original_tubelet_size

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, video: torch.Tensor, skip_predictor: bool = True) -> VJEPA2Tokens:
        pixel_values = preprocess_video_tensor(
            video,
            image_size=self.image_size,
            resize_shortest_edge=self.resize_shortest_edge,
        )
        model_dtype = next(self.model.parameters()).dtype
        if torch.is_floating_point(pixel_values) and pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(dtype=model_dtype)
        outputs = self.model(pixel_values_videos=pixel_values, skip_predictor=skip_predictor)
        video_tokens = outputs.last_hidden_state

        batch, time = pixel_values.shape[:2]
        t_prime = (time - self.tubelet_size) // self.tubelet_size + 1
        h_prime = self.image_size // self.patch_size
        w_prime = self.image_size // self.patch_size
        expected_tokens = t_prime * h_prime * w_prime
        if video_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected V-JEPA2 token count: "
                f"got {video_tokens.shape[1]}, expected {expected_tokens} "
                f"for grid_shape={(t_prime, h_prime, w_prime)}."
            )
        frame_tokens = video_tokens.view(batch, t_prime, h_prime * w_prime, video_tokens.shape[-1])
        return VJEPA2Tokens(
            frame_tokens=frame_tokens,
            video_tokens=video_tokens,
            grid_shape=(t_prime, h_prime, w_prime),
        )
