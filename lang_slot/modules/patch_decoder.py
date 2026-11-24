"""Patch-token decoder for slot representations.

This is the token-level analogue of OC-Manip's PatchDecoder: each slot predicts
all visual patch tokens plus an alpha mask, then masks are normalized across
slots and used to compose the final reconstruction.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import nn


class PatchDecoderOutput(NamedTuple):
    reconstruction: torch.Tensor
    masks: torch.Tensor
    object_reconstructions: torch.Tensor
    per_slot_reconstructions: torch.Tensor


class PatchDecoder(nn.Module):
    """Decode slots into patch/tubelet tokens.

    Args:
        slot_dim: Slot feature dimension.
        output_dim: Reconstructed token dimension, e.g. V-JEPA2 hidden size.
        num_patches: Number of patch tokens per time step.
        hidden_dim: Hidden size of the per-slot MLP decoder.
        num_layers: Number of hidden MLP layers.
        decoder_dim: Optional internal decoder dimension. Defaults to slot_dim.
        condition_dim: Optional per-slot semantic conditioning dimension.

    Shapes:
        slots: ``[B, T, K, D_slot]`` or ``[B, K, D_slot]``.
        condition_info: optional ``[B, K, D_cond]`` or ``[B, T, K, D_cond]``.
        reconstruction: ``[B, T, N, output_dim]`` or ``[B, N, output_dim]``.
        masks: ``[B, T, K, N]`` or ``[B, K, N]``.
    """

    def __init__(
        self,
        slot_dim: int,
        output_dim: int,
        num_patches: int,
        hidden_dim: int,
        num_layers: int = 3,
        decoder_dim: Optional[int] = None,
        condition_dim: Optional[int] = None,
        pos_embed_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if slot_dim <= 0 or output_dim <= 0 or num_patches <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim, output_dim, num_patches, and hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.slot_dim = int(slot_dim)
        self.output_dim = int(output_dim)
        self.num_patches = int(num_patches)
        self.decoder_dim = int(decoder_dim or slot_dim)
        self.condition_dim = condition_dim

        self.slot_proj = nn.Identity() if self.decoder_dim == self.slot_dim else nn.Linear(
            self.slot_dim,
            self.decoder_dim,
        )
        if condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, self.decoder_dim, bias=False)
        else:
            self.condition_proj = None

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.decoder_dim) * pos_embed_scale)

        layers = []
        in_dim = self.decoder_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim + 1))
        self.decoder = nn.Sequential(*layers)

    def _broadcast_condition(self, condition_info: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        if condition_info.ndim == 3 and slots.ndim == 4:
            condition_info = condition_info.unsqueeze(1).expand(-1, slots.shape[1], -1, -1)
        if condition_info.shape[:-1] != slots.shape[:-1]:
            raise ValueError(
                "condition_info must match slots batch/time/slot axes: "
                f"condition={tuple(condition_info.shape)}, slots={tuple(slots.shape)}."
            )
        return condition_info

    def forward(
        self,
        slots: torch.Tensor,
        condition_info: Optional[torch.Tensor] = None,
    ) -> PatchDecoderOutput:
        if slots.ndim not in (3, 4):
            raise ValueError(f"slots must have shape [B,K,D] or [B,T,K,D], got {tuple(slots.shape)}.")
        if slots.shape[-1] != self.slot_dim:
            raise ValueError(f"Expected slot_dim={self.slot_dim}, got {slots.shape[-1]}.")
        if (condition_info is None) != (self.condition_proj is None):
            raise ValueError("condition_info must be provided iff condition_dim was configured.")

        original_shape = slots.shape[:-1]
        features = self.slot_proj(slots)
        if condition_info is not None:
            condition_info = self._broadcast_condition(condition_info, slots)
            features = features + self.condition_proj(condition_info.to(dtype=features.dtype))

        flat_features = features.flatten(0, -2)
        patch_features = flat_features.unsqueeze(1).expand(-1, self.num_patches, -1)
        patch_features = patch_features + self.pos_embed.to(device=slots.device, dtype=slots.dtype)

        decoded = self.decoder(patch_features).unflatten(0, original_shape)
        decoded_tokens, alpha_logits = decoded.split([self.output_dim, 1], dim=-1)

        # alpha logits shape: [B, T, K, N, 1] or [B, K, N, 1].
        # Softmax over the slot axis K so each patch is explained by competing slots.
        slot_axis = -3
        masks = alpha_logits.softmax(dim=slot_axis)
        per_slot = decoded_tokens * masks
        reconstruction = per_slot.sum(dim=slot_axis)

        return PatchDecoderOutput(
            reconstruction=reconstruction,
            masks=masks.squeeze(-1),
            object_reconstructions=decoded_tokens,
            per_slot_reconstructions=per_slot,
        )
