"""Policy wrapper that trains/samples actions from fused slot/video conditions."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..modules import FlowMatchingActionDecoder
from .policy_condition_encoder import PolicyConditionEncoder


class FlowMatchingPolicy(nn.Module):
    """Condition encoder + flow-matching future-action decoder."""

    def __init__(
        self,
        *,
        condition_encoder: PolicyConditionEncoder,
        action_decoder: FlowMatchingActionDecoder,
        slot_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if slot_loss_weight < 0:
            raise ValueError("slot_loss_weight must be non-negative.")
        self.condition_encoder = condition_encoder
        self.action_decoder = action_decoder
        self.slot_loss_weight = float(slot_loss_weight)

    def forward(
        self,
        batch: Dict[str, object],
        *,
        compute_slot_loss: bool = False,
        flow_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        condition = self.condition_encoder(
            batch,
            compute_slot_loss=compute_slot_loss,
            flow_target=flow_target,
        )
        action_out = self.action_decoder.compute_loss(
            action_target=condition["future_actions"],
            fused_cond_tokens=condition["fused_cond_tokens"],
            condition_padding_mask=condition["condition_padding_mask"],
            action_padding_mask=condition["future_action_padding_mask"],
        )
        total_loss = action_out.loss
        future_outputs = condition["future_outputs"]
        if compute_slot_loss and isinstance(future_outputs, dict) and "loss" in future_outputs:
            total_loss = total_loss + self.slot_loss_weight * future_outputs["loss"]

        return {
            "loss": total_loss,
            "action_loss": action_out.loss,
            "velocity_pred": action_out.velocity_pred,
            "velocity_target": action_out.velocity_target,
            "noisy_action": action_out.noisy_action,
            "flow_time": action_out.flow_time,
            "condition": condition,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        batch: Dict[str, object],
        *,
        num_steps: int = 20,
        discretize_gripper: bool = False,
        gripper_index: int = 6,
    ) -> Dict[str, object]:
        condition = self.condition_encoder(batch, compute_slot_loss=False)
        actions = self.action_decoder.sample(
            fused_cond_tokens=condition["fused_cond_tokens"],
            condition_padding_mask=condition["condition_padding_mask"],
            action_padding_mask=condition["future_action_padding_mask"],
            num_steps=num_steps,
            discretize_gripper=discretize_gripper,
            gripper_index=gripper_index,
        )
        return {"action_pred": actions, "condition": condition}
