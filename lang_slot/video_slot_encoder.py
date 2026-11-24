"""SAVi-like recurrent video slot encoder in PyTorch.

The encoder accepts precomputed visual tokens, for example V-JEPA2 patch/tubelet
tokens, and runs Slot Attention independently at each time step while carrying
the previous slots forward as the next prior.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

from .modules import SlotAttention


GridShape = Tuple[int, int, int]
ForwardOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class VideoSlotEncoder(nn.Module):
    """Recurrent video slot encoder over frame/tubelet tokens.

    Args:
        input_dim: Feature dimension of visual tokens.
        slot_dim: Slot feature dimension.
        num_slots: Number of slots.
        num_iterations: Number of Slot Attention iterations per frame.
        use_learned_initial_slots: If true, create learned ``[1, K, D_slot]``
            initial slots. Otherwise ``initial_slots`` must be passed to forward.
        predictor_type: ``"identity"`` or ``"transformer"`` predictor for the
            next time-step prior slots.
        num_predictor_layers: Number of transformer encoder layers.
        num_predictor_heads: Number of self-attention heads over the slot axis.
        query_dim: Optional semantic query feature dimension. If provided,
            semantic queries can bias only the first recurrent slot prior.
        mlp_hidden_dim: Hidden size for Slot Attention and transformer MLPs.

    Inputs:
        frame_tokens: ``[B, T, N, D_in]``.
        video_tokens: ``[B, T_prime * N, D_in]`` plus
            ``grid_shape=(T_prime, H_prime, W_prime)``.

    Outputs:
        slots: ``[B, T, K, D_slot]``.
        If ``return_last_slots=True``, returns ``(slots, last_slots)`` where
        ``last_slots`` is ``[B, K, D_slot]``.
    """

    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        num_slots: int,
        num_iterations: int,
        use_learned_initial_slots: bool = True,
        predictor_type: str = "identity",
        num_predictor_layers: int = 1,
        num_predictor_heads: int = 4,
        query_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or slot_dim <= 0:
            raise ValueError("input_dim and slot_dim must be positive.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if predictor_type not in {"identity", "transformer"}:
            raise ValueError('predictor_type must be "identity" or "transformer".')

        self.input_dim = int(input_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.use_learned_initial_slots = bool(use_learned_initial_slots)
        self.predictor_type = predictor_type
        self.query_dim = None if query_dim is None else int(query_dim)
        if self.query_dim is not None and self.query_dim <= 0:
            raise ValueError("query_dim must be positive when provided.")

        if self.use_learned_initial_slots:
            self.initial_slots = nn.Parameter(0.01 * torch.randn(1, self.num_slots, self.slot_dim))
        else:
            self.register_parameter("initial_slots", None)
        self.semantic_init_proj = (
            nn.Linear(self.query_dim, self.slot_dim)
            if self.query_dim is not None
            else None
        )

        self.corrector = SlotAttention(
            input_dim=self.input_dim,
            slot_dim=self.slot_dim,
            num_iterations=self.num_iterations,
            qkv_dim=self.slot_dim,
            num_heads=1,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        if predictor_type == "identity":
            self.predictor = nn.Identity()
        else:
            if num_predictor_layers <= 0:
                raise ValueError("num_predictor_layers must be positive for transformer predictor.")
            if num_predictor_heads <= 0:
                raise ValueError("num_predictor_heads must be positive.")
            if self.slot_dim % num_predictor_heads != 0:
                raise ValueError("slot_dim must be divisible by num_predictor_heads.")
            predictor_mlp_dim = mlp_hidden_dim or 4 * self.slot_dim
            layer = nn.TransformerEncoderLayer(
                d_model=self.slot_dim,
                nhead=num_predictor_heads,
                dim_feedforward=predictor_mlp_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.predictor = nn.TransformerEncoder(
                layer,
                num_layers=num_predictor_layers,
                enable_nested_tensor=False,
            )

    def _tokens_from_inputs(
        self,
        frame_tokens: Optional[torch.Tensor],
        video_tokens: Optional[torch.Tensor],
        grid_shape: Optional[GridShape],
    ) -> torch.Tensor:
        if (frame_tokens is None) == (video_tokens is None):
            raise ValueError("Provide exactly one of frame_tokens or video_tokens.")

        if frame_tokens is not None:
            if frame_tokens.ndim != 4:
                raise ValueError(
                    "frame_tokens must have shape [B, T, N, D_in], "
                    f"got {tuple(frame_tokens.shape)}."
                )
            if frame_tokens.shape[-1] != self.input_dim:
                raise ValueError(f"Expected input_dim={self.input_dim}, got {frame_tokens.shape[-1]}.")
            return frame_tokens

        assert video_tokens is not None
        if grid_shape is None:
            raise ValueError("grid_shape=(T_prime, H_prime, W_prime) is required for video_tokens.")
        if len(grid_shape) != 3:
            raise ValueError("grid_shape must be a tuple of (T_prime, H_prime, W_prime).")
        t_prime, h_prime, w_prime = grid_shape
        if t_prime <= 0 or h_prime <= 0 or w_prime <= 0:
            raise ValueError("grid_shape entries must be positive.")
        if video_tokens.ndim != 3:
            raise ValueError(
                "video_tokens must have shape [B, T_prime * N, D_in], "
                f"got {tuple(video_tokens.shape)}."
            )
        if video_tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {video_tokens.shape[-1]}.")

        batch_size, token_count, _ = video_tokens.shape
        num_patches = h_prime * w_prime
        expected_tokens = t_prime * num_patches
        if token_count != expected_tokens:
            raise ValueError(
                "video_tokens length does not match grid_shape: "
                f"got {token_count}, expected {expected_tokens}."
            )

        # V-JEPA2-style flattened tokens are grouped back into per-time-step
        # patch tokens before the recurrent slot loop.
        return video_tokens.view(batch_size, t_prime, num_patches, self.input_dim)

    def _initial_slots(
        self,
        batch_size: int,
        tokens: torch.Tensor,
        initial_slots: Optional[torch.Tensor],
        semantic_queries: Optional[torch.Tensor],
        semantic_query_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if initial_slots is not None:
            if initial_slots.ndim != 3:
                raise ValueError(
                    "initial_slots must have shape [B, K, D_slot], "
                    f"got {tuple(initial_slots.shape)}."
                )
            if initial_slots.shape != (batch_size, self.num_slots, self.slot_dim):
                raise ValueError(
                    "initial_slots must have shape "
                    f"[{batch_size}, {self.num_slots}, {self.slot_dim}], got {tuple(initial_slots.shape)}."
                )
            if initial_slots.device != tokens.device:
                raise ValueError("initial_slots must be on the same device as input tokens.")
            slots = initial_slots.to(dtype=tokens.dtype)
        else:
            if self.initial_slots is None:
                raise ValueError(
                    "use_learned_initial_slots=False requires initial_slots=[B,K,D_slot] in forward."
                )
            slots = self.initial_slots.expand(batch_size, -1, -1).to(
                device=tokens.device,
                dtype=tokens.dtype,
            )

        if semantic_queries is None:
            return slots
        if self.semantic_init_proj is None:
            raise ValueError("semantic_queries were provided but query_dim was not configured.")
        if semantic_queries.ndim != 3:
            raise ValueError(
                "semantic_queries must have shape [B, K, query_dim], "
                f"got {tuple(semantic_queries.shape)}."
            )
        if semantic_queries.shape[:2] != (batch_size, self.num_slots):
            raise ValueError(
                "semantic_queries must have shape "
                f"[{batch_size}, {self.num_slots}, query_dim], got {tuple(semantic_queries.shape)}."
            )
        if semantic_queries.shape[-1] != self.query_dim:
            raise ValueError(f"Expected semantic query dim {self.query_dim}, got {semantic_queries.shape[-1]}.")
        if semantic_queries.device != tokens.device:
            raise ValueError("semantic_queries must be on the same device as input tokens.")
        semantic_bias = self.semantic_init_proj(semantic_queries.to(dtype=self.semantic_init_proj.weight.dtype))
        semantic_bias = semantic_bias.to(dtype=slots.dtype)
        if semantic_query_mask is None:
            mask = torch.ones(batch_size, self.num_slots, 1, device=tokens.device, dtype=slots.dtype)
        else:
            if semantic_query_mask.shape != (batch_size, self.num_slots):
                raise ValueError(
                    "semantic_query_mask must have shape "
                    f"[{batch_size}, {self.num_slots}], got {tuple(semantic_query_mask.shape)}."
                )
            if semantic_query_mask.device != tokens.device:
                raise ValueError("semantic_query_mask must be on the same device as input tokens.")
            mask = semantic_query_mask.to(dtype=slots.dtype).unsqueeze(-1)
        return slots + semantic_bias * mask

    def forward(
        self,
        *,
        frame_tokens: Optional[torch.Tensor] = None,
        video_tokens: Optional[torch.Tensor] = None,
        grid_shape: Optional[GridShape] = None,
        initial_slots: Optional[torch.Tensor] = None,
        semantic_queries: Optional[torch.Tensor] = None,
        semantic_query_mask: Optional[torch.Tensor] = None,
        return_last_slots: bool = False,
    ) -> ForwardOutput:
        tokens = self._tokens_from_inputs(frame_tokens, video_tokens, grid_shape)
        batch_size, num_frames, _, _ = tokens.shape
        prev_slots = self._initial_slots(
            batch_size,
            tokens,
            initial_slots,
            semantic_queries,
            semantic_query_mask,
        )

        slots_over_time = []
        for timestep in range(num_frames):
            # Correct current-frame prior slots with the visual patch tokens at t.
            slots_t = self.corrector(inputs=tokens[:, timestep], slots=prev_slots)
            slots_over_time.append(slots_t)

            # Predict the next prior over the slot axis. Identity keeps the
            # corrected slots; transformer lets slots communicate before t+1.
            prev_slots = self.predictor(slots_t)

        slots = torch.stack(slots_over_time, dim=1)
        if return_last_slots:
            return slots, slots[:, -1]
        return slots
