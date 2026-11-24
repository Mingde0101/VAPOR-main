"""End-to-end condition encoder before flow-matching action decoding."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..modules import ConditionAggregationOutput, ConditionAggregationTransformer
from .framewise_history_vjepa2_wrapper import VJEPA2FramewiseHistoryWrapper
from .vjepa2_future_video_wrapper import VJEPA2FutureVideoWrapper
from .vjepa2_slot_flow_wrapper import VJEPA2SlotFlowWrapper


class PolicyConditionEncoder(nn.Module):
    """Build fused condition tokens from a policy-window batch.

    Text embeddings are intentionally passed in as tensors. Run SigLIP2 outside
    DataLoader workers and add:
        task_semantic_tokens: [B,D_text]
        semantic_queries: [B,K,D_text]
        contrastive_loss_mask: [B,K]
    """

    def __init__(
        self,
        *,
        history_encoder: VJEPA2FramewiseHistoryWrapper,
        future_slot_encoder: Optional[VJEPA2SlotFlowWrapper] = None,
        future_video_encoder: Optional[VJEPA2FutureVideoWrapper] = None,
        condition_fusion: ConditionAggregationTransformer,
        freeze_future_slot_encoder: bool = False,
        use_slot_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.use_slot_tokens = bool(use_slot_tokens)
        if self.use_slot_tokens and future_slot_encoder is None:
            raise ValueError("future_slot_encoder is required when use_slot_tokens=True.")
        if not self.use_slot_tokens and future_video_encoder is None:
            raise ValueError("future_video_encoder is required when use_slot_tokens=False.")
        self.history_encoder = history_encoder
        self.future_slot_encoder = future_slot_encoder
        self.future_video_encoder = future_video_encoder
        self.condition_fusion = condition_fusion
        self.freeze_future_slot_encoder = bool(freeze_future_slot_encoder)

    @staticmethod
    def _require_tensor(batch: Dict[str, object], key: str) -> torch.Tensor:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise KeyError(f"batch must contain tensor field {key!r}.")
        return value

    @staticmethod
    def _check_batch_alignment(
        hist_images: torch.Tensor,
        hist_states: torch.Tensor,
        hist_actions: torch.Tensor,
        future_images: torch.Tensor,
        future_actions: torch.Tensor,
        task_semantic_tokens: torch.Tensor,
        semantic_queries: torch.Tensor,
    ) -> None:
        tensors = {
            "hist_states": hist_states,
            "hist_actions": hist_actions,
            "future_images": future_images,
            "future_actions": future_actions,
            "task_semantic_tokens": task_semantic_tokens,
            "semantic_queries": semantic_queries,
        }
        batch_size = hist_images.shape[0]
        device = hist_images.device
        for name, tensor in tensors.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(f"Batch mismatch: hist_images B={batch_size}, {name} B={tensor.shape[0]}.")
            if tensor.device != device:
                raise ValueError(f"Device mismatch: hist_images on {device}, {name} on {tensor.device}.")
        if future_images.shape[1] != future_actions.shape[1] + 1:
            raise ValueError(
                "future_images must have one more raw frame than future_actions: "
                f"future_images T={future_images.shape[1]}, future_actions T={future_actions.shape[1]}."
            )

    def forward(
        self,
        batch: Dict[str, object],
        *,
        compute_slot_loss: bool = False,
        flow_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        hist_images = self._require_tensor(batch, "hist_images")
        hist_states = self._require_tensor(batch, "hist_states").float()
        hist_actions = self._require_tensor(batch, "hist_actions").float()
        future_images = self._require_tensor(batch, "future_images")
        future_actions = self._require_tensor(batch, "future_actions").float()
        task_semantic_tokens = self._require_tensor(batch, "task_semantic_tokens").float()
        semantic_queries = self._require_tensor(batch, "semantic_queries").float()
        contrastive_loss_mask = self._require_tensor(batch, "contrastive_loss_mask")

        self._check_batch_alignment(
            hist_images,
            hist_states,
            hist_actions,
            future_images,
            future_actions,
            task_semantic_tokens,
            semantic_queries,
        )

        hist_image_mask = self._require_tensor(batch, "hist_image_padding_mask").to(dtype=torch.bool)
        hist_state_mask = self._require_tensor(batch, "hist_state_padding_mask").to(dtype=torch.bool)
        hist_action_mask = self._require_tensor(batch, "hist_action_padding_mask").to(dtype=torch.bool)
        future_frame_mask = self._require_tensor(batch, "future_frame_padding_mask").to(dtype=torch.bool)
        future_action_mask = self._require_tensor(batch, "future_action_padding_mask").to(dtype=torch.bool)

        history_outputs = self.history_encoder(
            hist_images,
            history_padding_mask=hist_image_mask,
        )
        effective_slot_loss = bool(compute_slot_loss) and self.use_slot_tokens and not self.freeze_future_slot_encoder
        if self.use_slot_tokens and self.freeze_future_slot_encoder:
            with torch.no_grad():
                future_outputs = self.future_slot_encoder(
                    future_images,
                    video_padding_mask=future_frame_mask,
                    semantic_queries=semantic_queries,
                    contrastive_loss_mask=contrastive_loss_mask,
                    flow_target=flow_target,
                    compute_loss=False,
                    decode=False,
                )
        elif self.use_slot_tokens:
            future_outputs = self.future_slot_encoder(
                future_images,
                video_padding_mask=future_frame_mask,
                semantic_queries=semantic_queries,
                contrastive_loss_mask=contrastive_loss_mask,
                flow_target=flow_target,
                compute_loss=effective_slot_loss,
                decode=effective_slot_loss,
            )
        else:
            future_outputs = self.future_video_encoder(
                future_images,
                video_padding_mask=future_frame_mask,
            )
        fusion_output = self.condition_fusion(
            hist_image_tokens=history_outputs["spatial_tokens"],
            hist_states=hist_states,
            hist_actions=hist_actions,
            future_video_tokens=future_outputs["spatial_tokens"],
            future_slot_tokens=future_outputs.get("slots"),
            task_semantic_tokens=task_semantic_tokens,
            hist_image_padding_mask=history_outputs.get("history_padding_mask", hist_image_mask),
            hist_state_padding_mask=hist_state_mask,
            hist_action_padding_mask=hist_action_mask,
            future_video_padding_mask=future_outputs.get("frame_padding_mask"),
            future_slot_padding_mask=future_outputs.get("slot_padding_mask"),
            return_output=True,
        )
        if not isinstance(fusion_output, ConditionAggregationOutput):
            raise RuntimeError("condition_fusion(return_output=True) did not return ConditionAggregationOutput.")

        return {
            "fused_cond_tokens": fusion_output.fused_cond_tokens,
            "condition_padding_mask": fusion_output.padding_mask,
            "history_outputs": history_outputs,
            "future_outputs": future_outputs,
            "future_actions": future_actions,
            "future_action_padding_mask": future_action_mask,
        }
