"""SAVi-style spatial broadcast decoder for optical flow."""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class FlowDecoderOutput(NamedTuple):
    flow: torch.Tensor
    masks: torch.Tensor
    segmentations: torch.Tensor
    flow_slots: torch.Tensor
    flow_masked: torch.Tensor


def _make_coord_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=0)


class FlowDecoder(nn.Module):
    """Decode slots into dense optical flow with alpha-mask composition.

    This mirrors the SAVi decoder pattern: every slot is spatially broadcast,
    predicts a per-slot flow image and an alpha logit map, then alpha is
    normalized over slots and used to combine the per-slot flow predictions.

    Shapes:
        slots: ``[B, T, K, D_slot]`` or ``[B, K, D_slot]``.
        flow: ``[B, T, H, W, C]`` or ``[B, H, W, C]``.
        masks: ``[B, T, K, H, W]`` or ``[B, K, H, W]``.
    """

    def __init__(
        self,
        slot_dim: int,
        flow_channels: int = 2,
        output_resolution: Sequence[int] = (256, 256),
        broadcast_resolution: Sequence[int] = (16, 16),
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if slot_dim <= 0 or flow_channels <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim, flow_channels, and hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if len(output_resolution) != 2 or len(broadcast_resolution) != 2:
            raise ValueError("output_resolution and broadcast_resolution must be (H, W).")

        self.slot_dim = int(slot_dim)
        self.flow_channels = int(flow_channels)
        self.output_resolution = (int(output_resolution[0]), int(output_resolution[1]))
        self.broadcast_resolution = (int(broadcast_resolution[0]), int(broadcast_resolution[1]))

        layers = []
        in_channels = self.slot_dim + 2
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
            in_channels = hidden_dim
        layers.append(nn.Conv2d(in_channels, self.flow_channels + 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, slots: torch.Tensor) -> FlowDecoderOutput:
        if slots.ndim not in (3, 4):
            raise ValueError(f"slots must have shape [B,K,D] or [B,T,K,D], got {tuple(slots.shape)}.")
        if slots.shape[-1] != self.slot_dim:
            raise ValueError(f"Expected slot_dim={self.slot_dim}, got {slots.shape[-1]}.")

        has_time = slots.ndim == 4
        if has_time:
            batch_size, num_frames, num_slots, _ = slots.shape
            flat_slots = slots.reshape(batch_size * num_frames * num_slots, self.slot_dim)
            leading_shape: Tuple[int, ...] = (batch_size, num_frames, num_slots)
        else:
            batch_size, num_slots, _ = slots.shape
            flat_slots = slots.reshape(batch_size * num_slots, self.slot_dim)
            leading_shape = (batch_size, num_slots)

        height_b, width_b = self.broadcast_resolution
        height, width = self.output_resolution
        x = flat_slots[:, :, None, None].expand(-1, -1, height_b, width_b)
        coords = _make_coord_grid(height_b, width_b, slots.device, slots.dtype)
        coords = coords.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, coords], dim=1)

        decoded = self.decoder(x)
        if (height_b, width_b) != (height, width):
            decoded = F.interpolate(decoded, size=(height, width), mode="bilinear", align_corners=False)

        decoded = decoded.reshape(*leading_shape, self.flow_channels + 1, height, width)
        decoded = decoded.movedim(-3, -1)
        flow_slots, alpha_logits = decoded.split([self.flow_channels, 1], dim=-1)

        slot_axis = 2 if has_time else 1
        masks = alpha_logits.softmax(dim=slot_axis)
        flow_masked = flow_slots * masks
        flow = flow_masked.sum(dim=slot_axis)
        segmentations = alpha_logits.squeeze(-1).argmax(dim=slot_axis)

        return FlowDecoderOutput(
            flow=flow,
            masks=masks.squeeze(-1),
            segmentations=segmentations,
            flow_slots=flow_slots,
            flow_masked=flow_masked,
        )
