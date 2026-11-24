"""Flow-matching action decoder conditioned on fused multimodal tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class FlowMatchingActionOutput:
    """Training tensors for rectified-flow action prediction."""

    loss: torch.Tensor
    velocity_pred: torch.Tensor
    velocity_target: torch.Tensor
    noisy_action: torch.Tensor
    flow_time: torch.Tensor
    noise: torch.Tensor


def sinusoidal_flow_time_embedding(flow_time: torch.Tensor, dim: int) -> torch.Tensor:
    """Embed continuous flow time ``[B]`` into ``[B,dim]``."""
    if flow_time.ndim != 1:
        raise ValueError(f"flow_time must be [B], got {tuple(flow_time.shape)}.")
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"time embedding dim must be a positive even number, got {dim}.")
    half_dim = dim // 2
    work_time = flow_time.float()
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(
        torch.arange(half_dim, device=flow_time.device, dtype=torch.float32) * exponent
    )
    angles = 2.0 * math.pi * work_time[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1).to(dtype=flow_time.dtype)


class FlowMatchingActionDecoder(nn.Module):
    """Predict ``v(x_t,t,condition)`` for a fixed-length future action chunk.

    Rectified-flow convention:
        ``x_t = (1 - t) * action + t * noise``
        ``velocity_target = noise - action``

    Sampling starts from Gaussian noise at ``t=1`` and integrates backward to
    ``t=0``. Action tokens self-attend and cross-attend to fused condition
    tokens through ``nn.TransformerDecoder``.
    """

    def __init__(
        self,
        *,
        action_dim: int = 7,
        action_horizon: int = 4,
        condition_dim: int = 512,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        loss_type: str = "mse",
        flow_time_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        for name, value in {
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "condition_dim": condition_dim,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError('loss_type must be "mse", "l1", or "smooth_l1".')
        if not 0.0 <= flow_time_eps < 0.5:
            raise ValueError("flow_time_eps must be in [0,0.5).")

        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.condition_dim = int(condition_dim)
        self.d_model = int(d_model)
        self.loss_type = loss_type
        self.flow_time_eps = float(flow_time_eps)

        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.condition_proj = (
            nn.Identity()
            if self.condition_dim == self.d_model
            else nn.Linear(self.condition_dim, self.d_model)
        )
        self.action_step_embedding = nn.Embedding(self.action_horizon, self.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(round(self.d_model * mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(self.d_model)
        self.velocity_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.action_dim),
        )

    def _validate_inputs(
        self,
        noisy_action: torch.Tensor,
        flow_time: torch.Tensor,
        fused_cond_tokens: torch.Tensor,
        condition_padding_mask: Optional[torch.Tensor],
        action_padding_mask: Optional[torch.Tensor],
    ) -> None:
        if noisy_action.ndim != 3 or noisy_action.shape[1:] != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "noisy_action must be "
                f"[B,{self.action_horizon},{self.action_dim}], got {tuple(noisy_action.shape)}."
            )
        if flow_time.shape != (noisy_action.shape[0],):
            raise ValueError(f"flow_time must be [B], got {tuple(flow_time.shape)}.")
        if fused_cond_tokens.ndim != 3 or fused_cond_tokens.shape[-1] != self.condition_dim:
            raise ValueError(
                f"fused_cond_tokens must be [B,N,{self.condition_dim}], "
                f"got {tuple(fused_cond_tokens.shape)}."
            )
        if fused_cond_tokens.shape[0] != noisy_action.shape[0]:
            raise ValueError("noisy_action and fused_cond_tokens batch sizes must match.")
        if noisy_action.device != fused_cond_tokens.device or flow_time.device != noisy_action.device:
            raise ValueError("noisy_action, flow_time, and fused_cond_tokens must share a device.")
        if condition_padding_mask is not None:
            if condition_padding_mask.shape != fused_cond_tokens.shape[:2]:
                raise ValueError(
                    "condition_padding_mask must be [B,N_cond], got "
                    f"{tuple(condition_padding_mask.shape)}."
                )
            if condition_padding_mask.device != noisy_action.device:
                raise ValueError("condition_padding_mask must share the input device.")
        if action_padding_mask is not None:
            if action_padding_mask.shape != noisy_action.shape[:2]:
                raise ValueError(
                    f"action_padding_mask must be [B,{self.action_horizon}], "
                    f"got {tuple(action_padding_mask.shape)}."
                )
            if action_padding_mask.device != noisy_action.device:
                raise ValueError("action_padding_mask must share the input device.")

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        flow_time: torch.Tensor,
        fused_cond_tokens: torch.Tensor,
        condition_padding_mask: Optional[torch.Tensor] = None,
        action_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            noisy_action,
            flow_time,
            fused_cond_tokens,
            condition_padding_mask,
            action_padding_mask,
        )
        model_dtype = self.action_proj.weight.dtype
        action_tokens = self.action_proj(noisy_action.to(dtype=model_dtype))
        step_ids = torch.arange(self.action_horizon, device=noisy_action.device)
        action_tokens = action_tokens + self.action_step_embedding(step_ids).unsqueeze(0)
        time_embedding = sinusoidal_flow_time_embedding(
            flow_time.to(dtype=model_dtype),
            self.d_model,
        )
        action_tokens = action_tokens + self.time_mlp(time_embedding).unsqueeze(1)
        memory = self.condition_proj(fused_cond_tokens.to(dtype=model_dtype))
        decoded = self.decoder(
            tgt=action_tokens,
            memory=memory,
            tgt_key_padding_mask=None
            if action_padding_mask is None
            else action_padding_mask.to(dtype=torch.bool),
            memory_key_padding_mask=None
            if condition_padding_mask is None
            else condition_padding_mask.to(dtype=torch.bool),
        )
        return self.velocity_head(self.output_norm(decoded))

    def _masked_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            elementwise = (prediction - target).square()
        elif self.loss_type == "l1":
            elementwise = (prediction - target).abs()
        else:
            elementwise = torch.nn.functional.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
            )
        if action_padding_mask is None:
            return elementwise.mean()
        valid = (~action_padding_mask.to(dtype=torch.bool)).unsqueeze(-1).expand_as(elementwise)
        if not valid.any():
            return prediction.sum() * 0.0
        return elementwise[valid].mean()

    def compute_loss(
        self,
        *,
        action_target: torch.Tensor,
        fused_cond_tokens: torch.Tensor,
        condition_padding_mask: Optional[torch.Tensor] = None,
        action_padding_mask: Optional[torch.Tensor] = None,
        flow_time: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> FlowMatchingActionOutput:
        if action_target.ndim != 3 or action_target.shape[1:] != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "action_target must be "
                f"[B,{self.action_horizon},{self.action_dim}], got {tuple(action_target.shape)}."
            )
        if action_target.device != fused_cond_tokens.device:
            raise ValueError("action_target and fused_cond_tokens must share a device.")
        batch_size = action_target.shape[0]
        if flow_time is None:
            flow_time = torch.rand(batch_size, device=action_target.device, dtype=action_target.dtype)
            eps = self.flow_time_eps
            flow_time = eps + flow_time * (1.0 - 2.0 * eps)
        if noise is None:
            noise = torch.randn_like(action_target)
        if noise.shape != action_target.shape:
            raise ValueError("noise must have the same shape as action_target.")

        time_view = flow_time.view(batch_size, 1, 1)
        noisy_action = (1.0 - time_view) * action_target + time_view * noise
        velocity_target = noise - action_target
        velocity_pred = self(
            noisy_action=noisy_action,
            flow_time=flow_time,
            fused_cond_tokens=fused_cond_tokens,
            condition_padding_mask=condition_padding_mask,
            action_padding_mask=action_padding_mask,
        )
        loss = self._masked_loss(
            velocity_pred,
            velocity_target.to(dtype=velocity_pred.dtype),
            action_padding_mask,
        )
        return FlowMatchingActionOutput(
            loss=loss,
            velocity_pred=velocity_pred,
            velocity_target=velocity_target,
            noisy_action=noisy_action,
            flow_time=flow_time,
            noise=noise,
        )

    @torch.no_grad()
    def sample(
        self,
        *,
        fused_cond_tokens: torch.Tensor,
        condition_padding_mask: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        initial_noise: Optional[torch.Tensor] = None,
        action_padding_mask: Optional[torch.Tensor] = None,
        discretize_gripper: bool = False,
        gripper_index: int = 6,
        gripper_threshold: float = 0.5,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        batch_size = fused_cond_tokens.shape[0]
        if initial_noise is None:
            action = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=fused_cond_tokens.device,
                dtype=fused_cond_tokens.dtype,
            )
        else:
            expected = (batch_size, self.action_horizon, self.action_dim)
            if initial_noise.shape != expected:
                raise ValueError(f"initial_noise must be {expected}, got {tuple(initial_noise.shape)}.")
            action = initial_noise.to(device=fused_cond_tokens.device)

        dt = -1.0 / float(num_steps)
        for step in range(num_steps):
            t_value = 1.0 - step / float(num_steps)
            flow_time = torch.full(
                (batch_size,),
                t_value,
                device=action.device,
                dtype=action.dtype,
            )
            velocity = self(
                noisy_action=action,
                flow_time=flow_time,
                fused_cond_tokens=fused_cond_tokens,
                condition_padding_mask=condition_padding_mask,
                action_padding_mask=action_padding_mask,
            )
            action = action + dt * velocity

        if action_padding_mask is not None:
            action = action.masked_fill(action_padding_mask.unsqueeze(-1), 0.0)
        if discretize_gripper:
            if not 0 <= gripper_index < self.action_dim:
                raise ValueError(
                    f"gripper_index must be in [0,{self.action_dim - 1}], got {gripper_index}."
                )
            action = action.clone()
            action[..., gripper_index] = (
                action[..., gripper_index] >= gripper_threshold
            ).to(action.dtype)
        return action
