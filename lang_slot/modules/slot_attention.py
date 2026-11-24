"""Small PyTorch language-conditioned Slot Attention module.

This module follows the pattern used in OC-Manip: semantic features initialize
or bias a fixed set of slots, then the slots compete for visual patch tokens.
It is intentionally independent of TensorFlow/JAX and does not assume a
particular visual encoder.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn


ForwardOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class SlotAttention(nn.Module):
    """Slot Attention corrector for one frame of patch tokens.

    This module does not create slots by itself. It updates externally supplied
    slots with visual patch tokens, which is the form needed by a recurrent
    video slot encoder.

    Shapes:
        inputs: ``[B, N, input_dim]`` patch tokens.
        slots: ``[B, K, slot_dim]`` previous/prior slots.
        output: ``[B, K, slot_dim]`` updated slots.
    """

    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        num_iterations: int = 3,
        qkv_dim: Optional[int] = None,
        num_heads: int = 1,
        mlp_hidden_dim: Optional[int] = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or slot_dim <= 0:
            raise ValueError("input_dim and slot_dim must be positive.")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        qkv_dim = qkv_dim or slot_dim
        if qkv_dim % num_heads != 0:
            raise ValueError("qkv_dim must be divisible by num_heads.")

        self.input_dim = int(input_dim)
        self.slot_dim = int(slot_dim)
        self.num_iterations = int(num_iterations)
        self.qkv_dim = int(qkv_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.qkv_dim // self.num_heads
        self.epsilon = float(epsilon)

        self.norm_inputs = nn.LayerNorm(self.input_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.qkv_dim, bias=False)
        self.to_k = nn.Linear(self.input_dim, self.qkv_dim, bias=False)
        self.to_v = nn.Linear(self.input_dim, self.qkv_dim, bias=False)

        self.gru = nn.GRUCell(self.qkv_dim, self.slot_dim)
        mlp_hidden_dim = mlp_hidden_dim or 2 * self.slot_dim
        self.mlp_norm = nn.LayerNorm(self.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.slot_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        slots: torch.Tensor,
        return_attention: bool = False,
    ) -> ForwardOutput:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must have shape [B, N, D], got {tuple(inputs.shape)}.")
        if slots.ndim != 3:
            raise ValueError(f"slots must have shape [B, K, D_slot], got {tuple(slots.shape)}.")
        if inputs.shape[0] != slots.shape[0]:
            raise ValueError(f"Batch size mismatch: inputs B={inputs.shape[0]}, slots B={slots.shape[0]}.")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {inputs.shape[-1]}.")
        if slots.shape[-1] != self.slot_dim:
            raise ValueError(f"Expected slot_dim={self.slot_dim}, got {slots.shape[-1]}.")
        if inputs.device != slots.device:
            raise ValueError("inputs and slots must be on the same device.")

        batch_size, num_patches, _ = inputs.shape
        _, num_slots, _ = slots.shape

        inputs = self.norm_inputs(inputs)
        key = self.to_k(inputs).view(batch_size, num_patches, self.num_heads, self.head_dim)
        value = self.to_v(inputs).view(batch_size, num_patches, self.num_heads, self.head_dim)

        final_attention = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            query = self.to_q(self.norm_slots(slots))
            query = query.view(batch_size, num_slots, self.num_heads, self.head_dim)
            query = query * (1.0 / math.sqrt(self.head_dim))

            # logits shape: [B, K, H, N].
            # Patch-level competition: for each patch N, softmax over K*H.
            # With one head this is exactly competition over the slots.
            logits = torch.einsum("bkhd,bnhd->bkhn", query, key)
            attention = logits.flatten(1, 2).softmax(dim=1).view(
                batch_size,
                num_slots,
                self.num_heads,
                num_patches,
            )
            final_attention = attention.mean(dim=2)

            # Mean-style slot updates: normalize each slot over the patches it owns.
            attention = attention / (attention.sum(dim=-1, keepdim=True) + self.epsilon)
            updates = torch.einsum("bnhd,bkhn->bkhd", value, attention).flatten(-2, -1)

            slots = self.gru(
                updates.reshape(batch_size * num_slots, self.qkv_dim),
                slots_prev.reshape(batch_size * num_slots, self.slot_dim),
            ).view(batch_size, num_slots, self.slot_dim)
            slots = slots + self.mlp(self.mlp_norm(slots))

        if return_attention:
            assert final_attention is not None
            return slots, final_attention
        return slots


class SemanticSlotAttention(nn.Module):
    """Slot Attention with optional semantic query conditioning.

    Args:
        image_dim: Feature dimension of visual patch tokens.
        query_dim: Feature dimension of semantic query tokens.
        slot_dim: Output slot dimension.
        num_slots: Number of slots. Defaults to 5 for the Bridge/action setup.
        num_iterations: Number of Slot Attention refinement rounds.
        qkv_dim: Projection size for query/key/value. Defaults to ``slot_dim``.
        num_heads: Number of attention heads. ``qkv_dim`` must divide evenly.
        mlp_hidden_dim: Hidden size for the residual MLP after the GRU update.
        epsilon: Numerical epsilon for attention re-normalization.

    Shapes:
        image_features: ``[B, N, image_dim]`` patch tokens.
        semantic_queries: optional ``[B, K, query_dim]`` semantic queries.
        query_mask: optional ``[B, K]``; 1/True means that slot has semantics.

    Returns:
        slots: ``[B, K, slot_dim]`` by default.
        If ``return_attention=True``, also returns patch competition weights
        ``[B, K, N]`` from the final iteration.
    """

    def __init__(
        self,
        image_dim: int,
        query_dim: int,
        slot_dim: int,
        num_slots: int = 5,
        num_iterations: int = 3,
        qkv_dim: Optional[int] = None,
        num_heads: int = 1,
        mlp_hidden_dim: Optional[int] = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if image_dim <= 0 or query_dim <= 0 or slot_dim <= 0:
            raise ValueError("image_dim, query_dim, and slot_dim must be positive.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        qkv_dim = qkv_dim or slot_dim
        if qkv_dim % num_heads != 0:
            raise ValueError("qkv_dim must be divisible by num_heads.")

        self.image_dim = int(image_dim)
        self.query_dim = int(query_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.qkv_dim = int(qkv_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.qkv_dim // self.num_heads
        self.epsilon = float(epsilon)

        self.base_slots = nn.Parameter(0.01 * torch.randn(1, self.num_slots, self.slot_dim))
        self.query_proj = nn.Linear(self.query_dim, self.slot_dim)

        self.norm_inputs = nn.LayerNorm(self.image_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.qkv_dim, bias=False)
        self.to_k = nn.Linear(self.image_dim, self.qkv_dim, bias=False)
        self.to_v = nn.Linear(self.image_dim, self.qkv_dim, bias=False)

        self.gru = nn.GRUCell(self.qkv_dim, self.slot_dim)
        mlp_hidden_dim = mlp_hidden_dim or 2 * self.slot_dim
        self.mlp_norm = nn.LayerNorm(self.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.slot_dim),
        )

    def _initial_slots(
        self,
        batch_size: int,
        semantic_queries: Optional[torch.Tensor],
        query_mask: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        slots = self.base_slots.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        if semantic_queries is None:
            return slots

        if semantic_queries.ndim != 3:
            raise ValueError(
                "semantic_queries must have shape [B, K, query_dim], "
                f"got {tuple(semantic_queries.shape)}."
            )
        if semantic_queries.shape[:2] != (batch_size, self.num_slots):
            raise ValueError(
                "semantic_queries must have shape [B, num_slots, query_dim], "
                f"got {tuple(semantic_queries.shape)} with num_slots={self.num_slots}."
            )
        if semantic_queries.shape[-1] != self.query_dim:
            raise ValueError(
                f"Expected semantic query dim {self.query_dim}, got {semantic_queries.shape[-1]}."
            )
        if semantic_queries.device != device:
            raise ValueError("semantic_queries must be on the same device as image_features.")

        semantic_bias = self.query_proj(semantic_queries.to(dtype=dtype))
        if query_mask is None:
            mask = torch.ones(
                batch_size,
                self.num_slots,
                1,
                device=device,
                dtype=semantic_bias.dtype,
            )
        else:
            if query_mask.shape != (batch_size, self.num_slots):
                raise ValueError(
                    "query_mask must have shape [B, num_slots], "
                    f"got {tuple(query_mask.shape)} with num_slots={self.num_slots}."
                )
            if query_mask.device != device:
                raise ValueError("query_mask must be on the same device as image_features.")
            mask = query_mask.to(dtype=semantic_bias.dtype).unsqueeze(-1)

        # Same spirit as OC-Manip LangConditioning: all slots keep a learned base;
        # semantic slots receive an additive query-conditioned bias.
        return slots + semantic_bias * mask

    def forward(
        self,
        image_features: torch.Tensor,
        semantic_queries: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> ForwardOutput:
        """Extract semantic slots from visual patch tokens."""
        if image_features.ndim != 3:
            raise ValueError(
                "image_features must have shape [B, N, image_dim], "
                f"got {tuple(image_features.shape)}."
            )
        if image_features.shape[-1] != self.image_dim:
            raise ValueError(f"Expected image_dim={self.image_dim}, got {image_features.shape[-1]}.")

        batch_size, num_patches, _ = image_features.shape
        slots = self._initial_slots(
            batch_size,
            semantic_queries,
            query_mask,
            device=image_features.device,
            dtype=image_features.dtype,
        )

        inputs = self.norm_inputs(image_features)
        key = self.to_k(inputs).view(batch_size, num_patches, self.num_heads, self.head_dim)
        value = self.to_v(inputs).view(batch_size, num_patches, self.num_heads, self.head_dim)

        final_attention = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            query = self.to_q(self.norm_slots(slots))
            query = query.view(batch_size, self.num_slots, self.num_heads, self.head_dim)
            query = query * (1.0 / math.sqrt(self.head_dim))

            # logits shape: [B, K, H, N]
            # OC-Manip-style patch competition: for each visual patch N, softmax
            # over the flattened slot/head axis K*H. With num_heads=1 this is
            # exactly a softmax over the 5 slots for every patch.
            logits = torch.einsum("bkhd,bnhd->bkhn", query, key)
            attention = logits.flatten(1, 2).softmax(dim=1).view(
                batch_size,
                self.num_slots,
                self.num_heads,
                num_patches,
            )
            final_attention = attention.mean(dim=2)

            # Reweight over patches so each slot update is an average of the
            # patches assigned to it instead of growing with image token count.
            attention = attention / (attention.sum(dim=-1, keepdim=True) + self.epsilon)
            updates = torch.einsum("bnhd,bkhn->bkhd", value, attention).flatten(-2, -1)

            slots = self.gru(
                updates.reshape(batch_size * self.num_slots, self.qkv_dim),
                slots_prev.reshape(batch_size * self.num_slots, self.slot_dim),
            ).view(batch_size, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.mlp_norm(slots))

        if return_attention:
            assert final_attention is not None
            return slots, final_attention
        return slots
