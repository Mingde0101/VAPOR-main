"""Multimodal condition token aggregation for action decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn


@dataclass
class ConditionAggregationOutput:
    """Fused condition memory and its Transformer padding mask."""

    fused_cond_tokens: torch.Tensor
    padding_mask: torch.Tensor


class ConditionAggregationTransformer(nn.Module):
    """Fuse already-extracted multimodal condition tokens with self-attention.

    The module only performs projection, additive token metadata embeddings,
    token-wise concatenation, and TransformerEncoder aggregation. It does not
    call V-JEPA2, Slot Attention, or any action decoder.

    Expected inputs:
        hist_image_tokens: [B, T_hist_img, H_img, W_img, D_video]
        hist_states: [B, T_hist_state, D_state]
        hist_actions: [B, T_hist_action, D_action]
        future_video_tokens: [B, T_future, H_fut, W_fut, D_video]
        future_slot_tokens: [B, T_future_slot, K_slot, D_slot]
        task_semantic_tokens: [B, D_text]

    Output:
        fused_cond_tokens: [B, N_cond, d_model]
    """

    MODALITY_HIST_IMAGE = 0
    MODALITY_HIST_STATE = 1
    MODALITY_HIST_ACTION = 2
    MODALITY_FUTURE_VIDEO = 3
    MODALITY_FUTURE_SLOT = 4
    MODALITY_TASK_SEMANTIC = 5
    NUM_MODALITIES = 6

    def __init__(
        self,
        *,
        d_video: int,
        d_state: int,
        d_action: int,
        d_slot: int,
        d_text: int,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_hist_len: int = 16,
        max_future_len: int = 16,
        max_spatial_h: int = 32,
        max_spatial_w: int = 32,
        activation: str = "gelu",
        norm_first: bool = True,
        use_slot_tokens: bool = True,
    ) -> None:
        super().__init__()
        dims = {
            "d_video": d_video,
            "d_state": d_state,
            "d_action": d_action,
            "d_slot": d_slot,
            "d_text": d_text,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "max_hist_len": max_hist_len,
            "max_future_len": max_future_len,
            "max_spatial_h": max_spatial_h,
            "max_spatial_w": max_spatial_w,
        }
        for name, value in dims.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        self.d_video = int(d_video)
        self.d_state = int(d_state)
        self.d_action = int(d_action)
        self.d_slot = int(d_slot)
        self.d_text = int(d_text)
        self.d_model = int(d_model)
        self.max_hist_len = int(max_hist_len)
        self.max_future_len = int(max_future_len)
        self.max_spatial_h = int(max_spatial_h)
        self.max_spatial_w = int(max_spatial_w)
        self.use_slot_tokens = bool(use_slot_tokens)

        self.image_proj = nn.Linear(self.d_video, self.d_model)
        self.video_proj = nn.Linear(self.d_video, self.d_model)
        self.state_proj = nn.Linear(self.d_state, self.d_model)
        self.action_proj = nn.Linear(self.d_action, self.d_model)
        self.slot_proj = nn.Linear(self.d_slot, self.d_model) if self.use_slot_tokens else None
        self.task_proj = nn.Linear(self.d_text, self.d_model)

        # BERT/VisualBERT-style learnable token type embeddings.
        self.modality_embedding = nn.Embedding(self.NUM_MODALITIES, self.d_model)

        self.hist_time_embedding = nn.Embedding(self.max_hist_len, self.d_model)
        self.future_time_embedding = nn.Embedding(self.max_future_len, self.d_model)

        # Factorized absolute 2D spatial embeddings for image/video patch grids.
        self.height_embedding = nn.Embedding(self.max_spatial_h, self.d_model)
        self.width_embedding = nn.Embedding(self.max_spatial_w, self.d_model)

        dim_feedforward = int(round(self.d_model * mlp_ratio))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

    @staticmethod
    def _check_float_tensor(name: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be a floating point tensor, got {tensor.dtype}.")

    @staticmethod
    def _check_same_batch_and_device(
        reference_name: str,
        reference: torch.Tensor,
        *others: Tuple[str, torch.Tensor],
    ) -> None:
        batch_size = reference.shape[0]
        device = reference.device
        for name, tensor in others:
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: {reference_name} B={batch_size}, "
                    f"{name} B={tensor.shape[0]}."
                )
            if tensor.device != device:
                raise ValueError(
                    f"Device mismatch: {reference_name} is on {device}, {name} is on {tensor.device}."
                )

    def _modality_embed(self, modality_id: int, *, device: torch.device) -> torch.Tensor:
        ids = torch.tensor([modality_id], device=device, dtype=torch.long)
        return self.modality_embedding(ids).view(1, 1, self.d_model)

    def _history_time_embed(self, time: int, *, device: torch.device) -> torch.Tensor:
        if time > self.max_hist_len:
            raise ValueError(f"History length {time} exceeds max_hist_len={self.max_hist_len}.")
        ids = torch.arange(time, device=device, dtype=torch.long)
        return self.hist_time_embedding(ids)

    def _future_time_embed(self, time: int, *, device: torch.device) -> torch.Tensor:
        if time > self.max_future_len:
            raise ValueError(f"Future length {time} exceeds max_future_len={self.max_future_len}.")
        ids = torch.arange(time, device=device, dtype=torch.long)
        return self.future_time_embedding(ids)

    def _spatial_embed(self, height: int, width: int, *, device: torch.device) -> torch.Tensor:
        if height > self.max_spatial_h:
            raise ValueError(f"Spatial height {height} exceeds max_spatial_h={self.max_spatial_h}.")
        if width > self.max_spatial_w:
            raise ValueError(f"Spatial width {width} exceeds max_spatial_w={self.max_spatial_w}.")
        h_ids = torch.arange(height, device=device, dtype=torch.long)
        w_ids = torch.arange(width, device=device, dtype=torch.long)
        h_embed = self.height_embedding(h_ids).view(1, height, 1, self.d_model)
        w_embed = self.width_embedding(w_ids).view(1, 1, width, self.d_model)
        return h_embed + w_embed  # [1, H, W, d_model]

    def _encode_video_tokens(
        self,
        tokens: torch.Tensor,
        *,
        projector: nn.Linear,
        time_embedding: nn.Embedding,
        max_time: int,
        modality_id: int,
        name: str,
    ) -> torch.Tensor:
        # [B,T,H,W,D_video] -> [B,T,H,W,d_model]
        if tokens.ndim != 5:
            raise ValueError(f"{name} must have shape [B,T,H,W,D_video], got {tuple(tokens.shape)}.")
        self._check_float_tensor(name, tokens)
        if tokens.shape[-1] != self.d_video:
            raise ValueError(f"{name} expected D_video={self.d_video}, got {tokens.shape[-1]}.")

        batch_size, time, height, width, _ = tokens.shape
        if time > max_time:
            raise ValueError(f"{name} time length {time} exceeds max_time={max_time}.")
        if height > self.max_spatial_h or width > self.max_spatial_w:
            raise ValueError(
                f"{name} spatial size {(height, width)} exceeds "
                f"max_spatial_h/w={(self.max_spatial_h, self.max_spatial_w)}."
            )

        encoded = projector(tokens.to(dtype=projector.weight.dtype))
        time_ids = torch.arange(time, device=tokens.device, dtype=torch.long)
        time_pos = time_embedding(time_ids).view(1, time, 1, 1, self.d_model)
        spatial_pos = self._spatial_embed(height, width, device=tokens.device).view(
            1, 1, height, width, self.d_model
        )
        modality_pos = self._modality_embed(modality_id, device=tokens.device).view(
            1, 1, 1, 1, self.d_model
        )
        encoded = encoded + time_pos + spatial_pos + modality_pos
        return encoded.reshape(batch_size, time * height * width, self.d_model)

    def _encode_sequence_tokens(
        self,
        tokens: torch.Tensor,
        *,
        projector: nn.Linear,
        expected_dim: int,
        time_embedding: nn.Embedding,
        max_time: int,
        modality_id: int,
        name: str,
    ) -> torch.Tensor:
        # [B,T,D] -> [B,T,d_model]
        if tokens.ndim != 3:
            raise ValueError(f"{name} must have shape [B,T,D], got {tuple(tokens.shape)}.")
        self._check_float_tensor(name, tokens)
        if tokens.shape[-1] != expected_dim:
            raise ValueError(f"{name} expected feature dim {expected_dim}, got {tokens.shape[-1]}.")

        time = tokens.shape[1]
        if time > max_time:
            raise ValueError(f"{name} time length {time} exceeds max_time={max_time}.")

        encoded = projector(tokens.to(dtype=projector.weight.dtype))
        time_ids = torch.arange(time, device=tokens.device, dtype=torch.long)
        time_pos = time_embedding(time_ids).view(1, time, self.d_model)
        modality_pos = self._modality_embed(modality_id, device=tokens.device)
        return encoded + time_pos + modality_pos

    def _encode_slot_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B,T,K,D_slot] -> [B,T*K,d_model]. Slots get temporal + modality, no spatial embedding.
        if self.slot_proj is None:
            raise RuntimeError("slot token encoding was requested but use_slot_tokens=False.")
        name = "future_slot_tokens"
        if tokens.ndim != 4:
            raise ValueError(f"{name} must have shape [B,T,K,D_slot], got {tuple(tokens.shape)}.")
        self._check_float_tensor(name, tokens)
        if tokens.shape[-1] != self.d_slot:
            raise ValueError(f"{name} expected D_slot={self.d_slot}, got {tokens.shape[-1]}.")

        batch_size, time, num_slots, _ = tokens.shape
        if time > self.max_future_len:
            raise ValueError(f"{name} time length {time} exceeds max_future_len={self.max_future_len}.")

        encoded = self.slot_proj(tokens.to(dtype=self.slot_proj.weight.dtype))
        time_ids = torch.arange(time, device=tokens.device, dtype=torch.long)
        time_pos = self.future_time_embedding(time_ids).view(1, time, 1, self.d_model)
        modality_pos = self._modality_embed(self.MODALITY_FUTURE_SLOT, device=tokens.device).view(
            1, 1, 1, self.d_model
        )
        encoded = encoded + time_pos + modality_pos
        return encoded.reshape(batch_size, time * num_slots, self.d_model)

    @staticmethod
    def _sequence_padding_mask(
        mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        time: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if mask is None:
            return torch.zeros(batch_size, time, device=device, dtype=torch.bool)
        if mask.shape != (batch_size, time):
            raise ValueError(f"{name} must be [B,T]={(batch_size, time)}, got {tuple(mask.shape)}.")
        if mask.device != device:
            raise ValueError(f"{name} must be on {device}, got {mask.device}.")
        return mask.to(dtype=torch.bool)

    def _condition_padding_mask(
        self,
        *,
        hist_image_tokens: torch.Tensor,
        hist_states: torch.Tensor,
        hist_actions: torch.Tensor,
        future_video_tokens: torch.Tensor,
        future_slot_tokens: Optional[torch.Tensor],
        hist_image_padding_mask: Optional[torch.Tensor],
        hist_state_padding_mask: Optional[torch.Tensor],
        hist_action_padding_mask: Optional[torch.Tensor],
        future_video_padding_mask: Optional[torch.Tensor],
        future_slot_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = hist_image_tokens.shape[0]
        device = hist_image_tokens.device
        hist_image_mask = self._sequence_padding_mask(
            hist_image_padding_mask,
            batch_size=batch_size,
            time=hist_image_tokens.shape[1],
            device=device,
            name="hist_image_padding_mask",
        )
        hist_image_mask = hist_image_mask.unsqueeze(-1).expand(
            batch_size,
            hist_image_tokens.shape[1],
            hist_image_tokens.shape[2] * hist_image_tokens.shape[3],
        ).reshape(batch_size, -1)
        hist_state_mask = self._sequence_padding_mask(
            hist_state_padding_mask,
            batch_size=batch_size,
            time=hist_states.shape[1],
            device=device,
            name="hist_state_padding_mask",
        )
        hist_action_mask = self._sequence_padding_mask(
            hist_action_padding_mask,
            batch_size=batch_size,
            time=hist_actions.shape[1],
            device=device,
            name="hist_action_padding_mask",
        )
        future_video_mask = self._sequence_padding_mask(
            future_video_padding_mask,
            batch_size=batch_size,
            time=future_video_tokens.shape[1],
            device=device,
            name="future_video_padding_mask",
        )
        future_video_mask = future_video_mask.unsqueeze(-1).expand(
            batch_size,
            future_video_tokens.shape[1],
            future_video_tokens.shape[2] * future_video_tokens.shape[3],
        ).reshape(batch_size, -1)

        if self.use_slot_tokens:
            if future_slot_tokens is None:
                raise ValueError("future_slot_tokens is required when use_slot_tokens=True.")
        elif future_slot_tokens is not None:
            raise ValueError("future_slot_tokens must be None when use_slot_tokens=False.")

        if self.use_slot_tokens and future_slot_padding_mask is None:
            future_slot_mask = torch.zeros(
                batch_size,
                future_slot_tokens.shape[1],
                future_slot_tokens.shape[2],
                device=device,
                dtype=torch.bool,
            )
        elif self.use_slot_tokens and future_slot_padding_mask.shape == future_slot_tokens.shape[:2]:
            future_slot_mask = future_slot_padding_mask.to(device=device, dtype=torch.bool)
            future_slot_mask = future_slot_mask.unsqueeze(-1).expand(
                batch_size,
                future_slot_tokens.shape[1],
                future_slot_tokens.shape[2],
            )
        elif self.use_slot_tokens and future_slot_padding_mask.shape == future_slot_tokens.shape[:3]:
            if future_slot_padding_mask.device != device:
                raise ValueError("future_slot_padding_mask must be on the token device.")
            future_slot_mask = future_slot_padding_mask.to(dtype=torch.bool)
        elif self.use_slot_tokens:
            raise ValueError(
                "future_slot_padding_mask must be [B,T] or [B,T,K], got "
                f"{tuple(future_slot_padding_mask.shape)}."
            )
        else:
            future_slot_mask = torch.empty(
                batch_size,
                0,
                device=device,
                dtype=torch.bool,
            )
        future_slot_mask = future_slot_mask.reshape(batch_size, -1)
        task_mask = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
        return torch.cat(
            [
                hist_image_mask,
                hist_state_mask,
                hist_action_mask,
                future_video_mask,
                future_slot_mask,
                task_mask,
            ],
            dim=1,
        )

    def _encode_task_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B,D_text] -> [B,1,d_model]. Task gets modality embedding only.
        name = "task_semantic_tokens"
        if tokens.ndim != 2:
            raise ValueError(f"{name} must have shape [B,D_text], got {tuple(tokens.shape)}.")
        self._check_float_tensor(name, tokens)
        if tokens.shape[-1] != self.d_text:
            raise ValueError(f"{name} expected D_text={self.d_text}, got {tokens.shape[-1]}.")

        encoded = self.task_proj(tokens.to(dtype=self.task_proj.weight.dtype)).unsqueeze(1)
        modality_pos = self._modality_embed(self.MODALITY_TASK_SEMANTIC, device=tokens.device)
        return encoded + modality_pos

    def forward(
        self,
        *,
        hist_image_tokens: torch.Tensor,
        hist_states: torch.Tensor,
        hist_actions: torch.Tensor,
        future_video_tokens: torch.Tensor,
        future_slot_tokens: Optional[torch.Tensor],
        task_semantic_tokens: torch.Tensor,
        hist_image_padding_mask: Optional[torch.Tensor] = None,
        hist_state_padding_mask: Optional[torch.Tensor] = None,
        hist_action_padding_mask: Optional[torch.Tensor] = None,
        future_video_padding_mask: Optional[torch.Tensor] = None,
        future_slot_padding_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
    ) -> Union[torch.Tensor, ConditionAggregationOutput]:
        tensors_to_check = [
            ("hist_states", hist_states),
            ("hist_actions", hist_actions),
            ("future_video_tokens", future_video_tokens),
            ("task_semantic_tokens", task_semantic_tokens),
        ]
        if self.use_slot_tokens:
            if future_slot_tokens is None:
                raise ValueError("future_slot_tokens is required when use_slot_tokens=True.")
            tensors_to_check.append(("future_slot_tokens", future_slot_tokens))
        elif future_slot_tokens is not None:
            raise ValueError("future_slot_tokens must be None when use_slot_tokens=False.")
        self._check_same_batch_and_device("hist_image_tokens", hist_image_tokens, *tensors_to_check)

        hist_image = self._encode_video_tokens(
            hist_image_tokens,
            projector=self.image_proj,
            time_embedding=self.hist_time_embedding,
            max_time=self.max_hist_len,
            modality_id=self.MODALITY_HIST_IMAGE,
            name="hist_image_tokens",
        )
        hist_state = self._encode_sequence_tokens(
            hist_states,
            projector=self.state_proj,
            expected_dim=self.d_state,
            time_embedding=self.hist_time_embedding,
            max_time=self.max_hist_len,
            modality_id=self.MODALITY_HIST_STATE,
            name="hist_states",
        )
        hist_action = self._encode_sequence_tokens(
            hist_actions,
            projector=self.action_proj,
            expected_dim=self.d_action,
            time_embedding=self.hist_time_embedding,
            max_time=self.max_hist_len,
            modality_id=self.MODALITY_HIST_ACTION,
            name="hist_actions",
        )
        future_video = self._encode_video_tokens(
            future_video_tokens,
            projector=self.video_proj,
            time_embedding=self.future_time_embedding,
            max_time=self.max_future_len,
            modality_id=self.MODALITY_FUTURE_VIDEO,
            name="future_video_tokens",
        )
        task = self._encode_task_tokens(task_semantic_tokens)

        # Token-wise concat: [B, sum_i N_i, d_model]. Never channel-wise concat.
        token_parts = [hist_image, hist_state, hist_action, future_video]
        if self.use_slot_tokens:
            token_parts.append(self._encode_slot_tokens(future_slot_tokens))
        token_parts.append(task)
        cond_tokens = torch.cat(token_parts, dim=1)
        padding_mask = self._condition_padding_mask(
            hist_image_tokens=hist_image_tokens,
            hist_states=hist_states,
            hist_actions=hist_actions,
            future_video_tokens=future_video_tokens,
            future_slot_tokens=future_slot_tokens,
            hist_image_padding_mask=hist_image_padding_mask,
            hist_state_padding_mask=hist_state_padding_mask,
            hist_action_padding_mask=hist_action_padding_mask,
            future_video_padding_mask=future_video_padding_mask,
            future_slot_padding_mask=future_slot_padding_mask,
        )
        fused = self.transformer(cond_tokens, src_key_padding_mask=padding_mask)
        if return_output:
            return ConditionAggregationOutput(
                fused_cond_tokens=fused,
                padding_mask=padding_mask,
            )
        return fused
