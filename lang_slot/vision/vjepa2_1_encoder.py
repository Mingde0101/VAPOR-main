"""V-JEPA 2.1 encoder adapter for the local policy token interface."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from typing import Mapping, Optional

import torch
from torch import nn

from lang_slot.paths import resources

from .vjepa2_encoder import VJEPA2Tokens, preprocess_video_tensor


DEFAULT_VJEPA21_REPO_DIR = str(resources()["VJEPA2_REPO_DIR"])
DEFAULT_VJEPA21_CHECKPOINT = str(resources()["VJEPA2_CHECKPOINT"])


def _import_official_vjepa21_vision_transformer(repo_dir: str):
    repo_path = Path(repo_dir).expanduser().resolve()
    module_path = repo_path / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            "V-JEPA 2.1 vision transformer source was not found at "
            f"{module_path}. Set VJEPA2_REPO_DIR or pass repo_dir explicitly."
        )

    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    try:
        module = import_module("app.vjepa_2_1.models.vision_transformer")
    except ImportError as exc:
        raise ImportError(
            f"Could not import the official V-JEPA 2.1 implementation from {repo_path}. "
            "Install the official repository dependencies (including timm and einops)."
        ) from exc
    if Path(module.__file__).resolve() != module_path.resolve():
        raise ImportError(f"V-JEPA was loaded from {module.__file__}, expected {module_path}")
    return module


def _build_official_vjepa21_vitl_encoder(
    *,
    repo_dir: str,
    image_size: int,
    patch_size: int,
    tubelet_size: int,
    num_frames: int,
) -> nn.Module:
    vision_transformer = _import_official_vjepa21_vision_transformer(repo_dir)
    return vision_transformer.vit_large(
        patch_size=patch_size,
        img_size=(image_size, image_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        n_output_distillation=1,
    )


def _clean_encoder_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("backbone."):
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("backbone."):
                key = key[len("backbone.") :]
        cleaned[key] = value
    return cleaned


def _load_checkpoint(path: str) -> Mapping[str, object]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"V-JEPA 2.1 checkpoint was not found at {checkpoint_path}."
        )

    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(checkpoint_path, weights_only=True, mmap=True, **load_kwargs)
    except TypeError:
        return torch.load(checkpoint_path, **load_kwargs)


class VJEPA21VideoEncoder(nn.Module):
    """Run the official V-JEPA 2.1 ViT-L/16 encoder at 384 resolution.

    The adapter preserves the interface used by the existing history and future
    wrappers. A duplicated two-frame history window or an adjacent future pair
    produces one temporal step with a 24 x 24 patch grid.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_VJEPA21_CHECKPOINT,
        repo_dir: str = DEFAULT_VJEPA21_REPO_DIR,
        *,
        image_size: int = 384,
        resize_shortest_edge: Optional[int] = 438,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 64,
        checkpoint_key: str = "ema_encoder",
        pretrained: bool = True,
        freeze: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        for name, value in {
            "image_size": image_size,
            "patch_size": patch_size,
            "tubelet_size": tubelet_size,
            "num_frames": num_frames,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}."
            )

        self.model = _build_official_vjepa21_vitl_encoder(
            repo_dir=repo_dir,
            image_size=int(image_size),
            patch_size=int(patch_size),
            tubelet_size=int(tubelet_size),
            num_frames=int(num_frames),
        )

        if pretrained:
            checkpoint = _load_checkpoint(checkpoint_path)
            if checkpoint_key not in checkpoint:
                raise KeyError(
                    f"Checkpoint {checkpoint_path} does not contain key {checkpoint_key!r}; "
                    f"available keys are {sorted(checkpoint.keys())}."
                )
            encoder_state = checkpoint[checkpoint_key]
            if not isinstance(encoder_state, Mapping):
                raise TypeError(
                    f"Checkpoint entry {checkpoint_key!r} must be a state dict, "
                    f"got {type(encoder_state).__name__}."
                )
            self.model.load_state_dict(_clean_encoder_state_dict(encoder_state), strict=True)
            del checkpoint

        self.image_size = int(image_size)
        self.resize_shortest_edge = resize_shortest_edge
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.hidden_size = int(getattr(self.model, "embed_dim", 1024))
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.autocast_dtype = torch_dtype
        self._frozen = bool(freeze)

        if self._frozen:
            self.model.eval()
            self.model.requires_grad_(False)

    def train(self, mode: bool = True) -> "VJEPA21VideoEncoder":
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def forward(self, video: torch.Tensor, skip_predictor: bool = True) -> VJEPA2Tokens:
        if not skip_predictor:
            raise ValueError("VJEPA21VideoEncoder wraps the encoder only; predictor output is unavailable.")

        pixel_values = preprocess_video_tensor(
            video,
            image_size=self.image_size,
            resize_shortest_edge=self.resize_shortest_edge,
        )
        if pixel_values.shape[1] < self.tubelet_size:
            raise ValueError(
                f"Video has {pixel_values.shape[1]} frame(s), but tubelet_size={self.tubelet_size}."
            )

        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=model_dtype).permute(0, 2, 1, 3, 4).contiguous()
        use_autocast = (
            pixel_values.device.type == "cuda"
            and self.autocast_dtype in (torch.float16, torch.bfloat16)
        )
        with torch.autocast(
            device_type=pixel_values.device.type,
            dtype=self.autocast_dtype if use_autocast else None,
            enabled=use_autocast,
        ):
            video_tokens = self.model(pixel_values)
        if not torch.is_tensor(video_tokens) or video_tokens.ndim != 3:
            shape = getattr(video_tokens, "shape", None)
            raise RuntimeError(
                "Unexpected V-JEPA 2.1 encoder output; expected [B,N,D], "
                f"got type={type(video_tokens).__name__}, shape={shape}."
            )

        batch_size, _, time, height, width = pixel_values.shape
        t_prime = (time - self.tubelet_size) // self.tubelet_size + 1
        h_prime = height // self.patch_size
        w_prime = width // self.patch_size
        expected_tokens = t_prime * h_prime * w_prime
        if video_tokens.shape != (batch_size, expected_tokens, self.hidden_size):
            raise RuntimeError(
                "Unexpected V-JEPA 2.1 token shape: "
                f"got {tuple(video_tokens.shape)}, expected "
                f"{(batch_size, expected_tokens, self.hidden_size)} for "
                f"grid_shape={(t_prime, h_prime, w_prime)}."
            )

        frame_tokens = video_tokens.reshape(
            batch_size,
            t_prime,
            h_prime * w_prime,
            self.hidden_size,
        )
        return VJEPA2Tokens(
            frame_tokens=frame_tokens,
            video_tokens=video_tokens,
            grid_shape=(t_prime, h_prime, w_prime),
        )
