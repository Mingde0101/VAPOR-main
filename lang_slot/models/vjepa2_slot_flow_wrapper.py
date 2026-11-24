"""Raw-video wrapper for V-JEPA2 tokens, recurrent slots, and flow decoding."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from ..text import dtype_from_name
from ..vision import (
    DEFAULT_VJEPA2_MODEL_DIR,
    VJEPA2Tokens,
    VJEPA2VideoEncoder,
    frame_tokens_to_spatial_grid,
)
from .video_slot_autoencoder import VideoSlotAutoEncoder


class VJEPA2SlotFlowWrapper(nn.Module):
    """Encode raw video frames with V-JEPA2 and train slot/token/flow heads.

    ``encoding_mode`` controls how a raw ``[B,T,H,W,3]`` or ``[B,T,3,H,W]`` video
    is converted into a token sequence before the recurrent slot encoder:

    - ``sliding_pair``: default. Use windows ``(0,1), (1,2), ...`` so T frames
      produce ``T-1`` token steps. This matches a pretrained tubelet_size=2
      V-JEPA2 while still advancing slots one adjacent-frame transition at a
      time.
    - ``single_frame``: keep one slot step per input frame by duplicating each
      frame into a 2-frame clip before V-JEPA2. This preserves the existing
      one-frame extraction interface without changing the pretrained tubelet.
    - ``full_clip``: pass the whole clip to V-JEPA2 directly, so the time length
      is whatever the encoder returns, e.g. roughly ``T/2`` for tubelet_size=2.
    """

    VALID_ENCODING_MODES = {"sliding_pair", "single_frame", "full_clip"}

    def __init__(
        self,
        *,
        input_dim: int = 1024,
        slot_dim: int = 256,
        num_slots: int = 5,
        num_iterations: int = 3,
        num_patches: int = 256,
        query_dim: int = 1536,
        encoding_mode: str = "sliding_pair",
        vjepa2_model_dir: str = DEFAULT_VJEPA2_MODEL_DIR,
        tubelet_size: int = 2,
        image_size: int = 256,
        resize_shortest_edge: Optional[int] = 292,
        freeze_vjepa2: bool = False,
        vjepa2_torch_dtype: Optional[str] = None,
        window_batch_size: Optional[int] = None,
        local_files_only: bool = True,
        ignore_mismatched_sizes: Optional[bool] = None,
        predictor_type: str = "transformer",
        num_predictor_layers: int = 1,
        num_predictor_heads: int = 4,
        slot_mlp_hidden_dim: Optional[int] = None,
        decoder_hidden_dim: int = 2048,
        decoder_num_layers: int = 3,
        decoder_conditioned: bool = True,
        embedding_dim: int = 4096,
        contrastive_temperature: float = 0.1,
        recon_loss_weight: float = 0.3,
        contrastive_loss_weight: float = 0.5,
        contrastive_matching: str = "index",
        use_flow: bool = False,
        flow_output_resolution: Optional[Sequence[int]] = None,
        flow_broadcast_resolution: Sequence[int] = (16, 16),
        flow_channels: int = 2,
        flow_hidden_dim: int = 128,
        flow_num_layers: int = 3,
        flow_loss_weight: float = 0.1,
        normalize_reconstruction_target: bool = False,
        normalize_reconstruction_prediction: bool = False,
        vision_encoder: Optional[nn.Module] = None,
        slot_model: Optional[VideoSlotAutoEncoder] = None,
    ) -> None:
        super().__init__()
        if encoding_mode not in self.VALID_ENCODING_MODES:
            raise ValueError(
                f"Unknown encoding_mode={encoding_mode!r}; expected one of "
                f"{sorted(self.VALID_ENCODING_MODES)}."
            )
        if tubelet_size != 2 and vision_encoder is None:
            raise ValueError(
                "This wrapper is intended to keep the pretrained tubelet_size=2 by default. "
                "Pass a custom vision_encoder if you intentionally want another temporal kernel."
            )

        self.encoding_mode = encoding_mode
        self.window_batch_size = None if window_batch_size is None else int(window_batch_size)
        if self.window_batch_size is not None and self.window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive when provided.")
        self.use_flow = bool(use_flow)
        if self.use_flow and flow_output_resolution is None:
            flow_output_resolution = (image_size, image_size)
        if not self.use_flow:
            flow_output_resolution = None

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

        if slot_model is None:
            slot_model = VideoSlotAutoEncoder(
                input_dim=input_dim,
                slot_dim=slot_dim,
                num_slots=num_slots,
                num_iterations=num_iterations,
                num_patches=num_patches,
                query_dim=query_dim,
                predictor_type=predictor_type,
                num_predictor_layers=num_predictor_layers,
                num_predictor_heads=num_predictor_heads,
                slot_mlp_hidden_dim=slot_mlp_hidden_dim,
                decoder_hidden_dim=decoder_hidden_dim,
                decoder_num_layers=decoder_num_layers,
                decoder_conditioned=decoder_conditioned,
                embedding_dim=embedding_dim,
                contrastive_temperature=contrastive_temperature,
                recon_loss_weight=recon_loss_weight,
                contrastive_loss_weight=contrastive_loss_weight,
                contrastive_matching=contrastive_matching,
                flow_output_resolution=flow_output_resolution,
                flow_broadcast_resolution=flow_broadcast_resolution,
                flow_channels=flow_channels,
                flow_hidden_dim=flow_hidden_dim,
                flow_num_layers=flow_num_layers,
                flow_loss_weight=flow_loss_weight,
                normalize_reconstruction_target=normalize_reconstruction_target,
                normalize_reconstruction_prediction=normalize_reconstruction_prediction,
            )
        if not self.use_flow and getattr(slot_model, "flow_decoder", None) is not None:
            # Keep the public switch authoritative, even for injected test/custom models.
            slot_model.flow_decoder = None
            slot_model.flow_loss = None
        self.slot_model = slot_model

    @staticmethod
    def _check_video(video: torch.Tensor) -> Tuple[int, int]:
        if video.ndim != 5:
            raise ValueError(
                f"video must be [B,T,H,W,3] or [B,T,3,H,W], got {tuple(video.shape)}."
            )
        batch_size, time = video.shape[:2]
        if video.shape[2] != 3 and video.shape[-1] != 3:
            raise ValueError(f"Could not infer channel axis for video shape {tuple(video.shape)}.")
        return batch_size, time

    @staticmethod
    def _flatten_time_windows(windows: torch.Tensor) -> torch.Tensor:
        # [B, S, 2, ...] -> [B*S, 2, ...], preserving dtype/device.
        batch_size, steps = windows.shape[:2]
        return windows.reshape(batch_size * steps, *windows.shape[2:])

    @classmethod
    def _make_sliding_pair_windows(cls, video: torch.Tensor) -> torch.Tensor:
        batch_size, time = cls._check_video(video)
        if time < 2:
            raise ValueError("sliding_pair mode requires at least 2 frames.")

        # Pair adjacent frames: [B,T-1,2,H,W,3] or [B,T-1,2,3,H,W].
        windows = torch.stack((video[:, :-1], video[:, 1:]), dim=2)
        assert windows.shape[0] == batch_size
        return cls._flatten_time_windows(windows)

    @classmethod
    def _make_single_frame_windows(cls, video: torch.Tensor) -> torch.Tensor:
        cls._check_video(video)

        # Duplicate each frame into a 2-frame clip so tubelet_size=2 returns one
        # feature step per original frame without changing pretrained weights.
        windows = torch.stack((video, video), dim=2)
        return cls._flatten_time_windows(windows)

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
                "Window modes expect one V-JEPA2 token time step per 2-frame window; "
                f"got T'={frame_tokens.shape[1]}."
            )
        _, _, num_patches, input_dim = frame_tokens.shape
        return frame_tokens[:, 0].reshape(batch_size, steps, num_patches, input_dim)

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

    def encode_video(
        self,
        video: torch.Tensor,
        *,
        skip_predictor: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Return frame tokens as ``[B, steps, N, D]`` plus token grid shape."""
        batch_size, time = self._check_video(video)

        if self.encoding_mode == "sliding_pair":
            steps = time - 1
            windows = self._make_sliding_pair_windows(video)
            encoded = self._encode_windows(windows, skip_predictor=skip_predictor)
            frame_tokens = self._reshape_window_tokens(
                encoded,
                batch_size=batch_size,
                steps=steps,
            )
            _, h_prime, w_prime = encoded.grid_shape
            return frame_tokens, (steps, h_prime, w_prime)

        if self.encoding_mode == "single_frame":
            steps = time
            windows = self._make_single_frame_windows(video)
            encoded = self._encode_windows(windows, skip_predictor=skip_predictor)
            frame_tokens = self._reshape_window_tokens(
                encoded,
                batch_size=batch_size,
                steps=steps,
            )
            _, h_prime, w_prime = encoded.grid_shape
            return frame_tokens, (steps, h_prime, w_prime)

        encoded = self.vision_encoder(video, skip_predictor=skip_predictor)
        if encoded.frame_tokens.ndim != 4:
            raise ValueError(
                f"vision_encoder.frame_tokens must be [B,T,N,D], got {tuple(encoded.frame_tokens.shape)}."
            )
        if encoded.frame_tokens.shape[0] != batch_size:
            raise ValueError(f"Expected batch size {batch_size}, got {encoded.frame_tokens.shape[0]}.")
        return encoded.frame_tokens, encoded.grid_shape

    def forward(
        self,
        video: torch.Tensor,
        *,
        video_padding_mask: Optional[torch.Tensor] = None,
        semantic_queries: Optional[torch.Tensor] = None,
        contrastive_loss_mask: Optional[torch.Tensor] = None,
        flow_target: Optional[torch.Tensor] = None,
        initial_slots: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        decode: bool = True,
        skip_predictor: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, input_time = self._check_video(video)
        frame_tokens, grid_shape = self.encode_video(video, skip_predictor=skip_predictor)
        encoded_padding_mask = None
        if video_padding_mask is not None:
            if video_padding_mask.shape != (batch_size, input_time):
                raise ValueError(
                    "video_padding_mask must match input video [B,T], got "
                    f"{tuple(video_padding_mask.shape)} for video {tuple(video.shape)}."
                )
            if video_padding_mask.device != video.device:
                raise ValueError("video_padding_mask and video must be on the same device.")
            video_padding_mask = video_padding_mask.to(dtype=torch.bool)
            if self.encoding_mode == "sliding_pair":
                encoded_padding_mask = video_padding_mask[:, :-1] | video_padding_mask[:, 1:]
            elif self.encoding_mode == "single_frame":
                encoded_padding_mask = video_padding_mask
            else:
                steps = frame_tokens.shape[1]
                tubelet_size = int(getattr(self.vision_encoder, "tubelet_size", 2))
                used = steps * tubelet_size
                encoded_padding_mask = video_padding_mask[:, :used].reshape(
                    batch_size, steps, tubelet_size
                ).any(dim=-1)
            if encoded_padding_mask.shape != frame_tokens.shape[:2]:
                raise RuntimeError(
                    "Encoded padding mask does not match V-JEPA2 token time grid: "
                    f"mask={tuple(encoded_padding_mask.shape)}, tokens={tuple(frame_tokens.shape)}."
                )

        slot_dtype = next(self.slot_model.parameters()).dtype
        if torch.is_floating_point(frame_tokens) and frame_tokens.dtype != slot_dtype:
            frame_tokens = frame_tokens.to(dtype=slot_dtype)

        if flow_target is not None and not self.use_flow:
            raise ValueError("flow_target was provided but use_flow=False.")
        if flow_target is not None and flow_target.ndim >= 2:
            expected_steps = frame_tokens.shape[1]
            if flow_target.shape[1] != expected_steps:
                raise ValueError(
                    f"flow_target time dimension must match encoded slot steps: "
                    f"got {flow_target.shape[1]}, expected {expected_steps} for "
                    f"encoding_mode={self.encoding_mode!r}."
                )

        outputs = self.slot_model(
            frame_tokens=frame_tokens,
            semantic_queries=semantic_queries,
            contrastive_loss_mask=contrastive_loss_mask,
            flow_target=flow_target,
            frame_padding_mask=encoded_padding_mask,
            initial_slots=initial_slots,
            compute_loss=compute_loss,
            decode=decode,
        )
        outputs["frame_tokens"] = frame_tokens
        outputs["spatial_tokens"] = frame_tokens_to_spatial_grid(frame_tokens, grid_shape)
        outputs["grid_shape"] = grid_shape
        if encoded_padding_mask is not None:
            num_patches = frame_tokens.shape[2]
            frame_token_padding_mask = encoded_padding_mask.unsqueeze(-1).expand(
                batch_size, frame_tokens.shape[1], num_patches
            )
            outputs["frame_padding_mask"] = encoded_padding_mask
            outputs["frame_token_padding_mask"] = frame_token_padding_mask
            outputs["video_token_padding_mask"] = frame_token_padding_mask.reshape(
                batch_size, frame_tokens.shape[1] * num_patches
            )
            if "slots" in outputs:
                outputs["slot_padding_mask"] = encoded_padding_mask.unsqueeze(-1).expand(
                    batch_size, frame_tokens.shape[1], outputs["slots"].shape[2]
                )
        return outputs
