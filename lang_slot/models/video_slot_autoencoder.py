"""Video slot autoencoder for token-level and flow pretraining."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn

from ..losses import DiagonalContrastiveLoss, ReconstructionLoss, VideoSlotPretrainingLoss
from ..modules import FlowDecoder, LangEmbeddingHead, PatchDecoder, SlotProjectorHead
from ..video_slot_encoder import GridShape, VideoSlotEncoder


class VideoSlotAutoEncoder(nn.Module):
    """Encode V-JEPA2 tokens into slots and reconstruct token/flow targets.

    The patch reconstruction target is the visual token tensor that entered the
    slot encoder, e.g. ``[B, T, N, 1024]`` from V-JEPA2 ViT-L. The optional flow
    branch mirrors SAVi: every slot predicts dense flow and an alpha mask, then
    the slot masks compose a single flow reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        num_slots: int,
        num_iterations: int,
        num_patches: int,
        query_dim: int,
        predictor_type: str = "identity",
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
        flow_output_resolution: Optional[Sequence[int]] = None,
        flow_broadcast_resolution: Sequence[int] = (16, 16),
        flow_channels: int = 2,
        flow_hidden_dim: int = 128,
        flow_num_layers: int = 3,
        flow_loss_weight: float = 0.1,
        normalize_reconstruction_target: bool = False,
        normalize_reconstruction_prediction: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.num_patches = int(num_patches)
        self.query_dim = int(query_dim)
        self.decoder_conditioned = bool(decoder_conditioned)

        self.encoder = VideoSlotEncoder(
            input_dim=input_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_iterations=num_iterations,
            predictor_type=predictor_type,
            num_predictor_layers=num_predictor_layers,
            num_predictor_heads=num_predictor_heads,
            query_dim=query_dim,
            mlp_hidden_dim=slot_mlp_hidden_dim,
        )
        self.decoder = PatchDecoder(
            slot_dim=slot_dim,
            output_dim=input_dim,
            num_patches=num_patches,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            condition_dim=query_dim if decoder_conditioned else None,
        )
        self.slot_projector = SlotProjectorHead(slot_dim=slot_dim, embedding_dim=embedding_dim)
        self.query_projector = LangEmbeddingHead(query_dim=query_dim, embedding_dim=embedding_dim)

        if flow_output_resolution is not None:
            self.flow_decoder = FlowDecoder(
                slot_dim=slot_dim,
                flow_channels=flow_channels,
                output_resolution=flow_output_resolution,
                broadcast_resolution=flow_broadcast_resolution,
                hidden_dim=flow_hidden_dim,
                num_layers=flow_num_layers,
            )
            self.flow_loss = ReconstructionLoss(loss_type="mse", weight=flow_loss_weight)
        else:
            self.flow_decoder = None
            self.flow_loss = None

        self.loss_fn = VideoSlotPretrainingLoss(
            reconstruction=ReconstructionLoss(
                loss_type="mse",
                weight=recon_loss_weight,
                normalize_prediction=normalize_reconstruction_prediction,
                normalize_target=normalize_reconstruction_target,
            ),
            contrastive=DiagonalContrastiveLoss(
                weight=contrastive_loss_weight,
                temperature=contrastive_temperature,
                batch_contrastive=True,
                symmetric=False,
                matching=contrastive_matching,
            ),
        )

    @staticmethod
    def _target_from_inputs(
        frame_tokens: Optional[torch.Tensor],
        video_tokens: Optional[torch.Tensor],
        grid_shape: Optional[GridShape],
    ) -> torch.Tensor:
        if frame_tokens is not None:
            return frame_tokens
        if video_tokens is None or grid_shape is None:
            raise ValueError("video_tokens and grid_shape are required when frame_tokens is not provided.")
        batch_size, _, input_dim = video_tokens.shape
        t_prime, h_prime, w_prime = grid_shape
        return video_tokens.view(batch_size, t_prime, h_prime * w_prime, input_dim)

    def _add_flow_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        slots: torch.Tensor,
    ) -> None:
        if self.flow_decoder is None:
            return
        flow_decoded = self.flow_decoder(slots)
        outputs.update(
            {
                "flow": flow_decoded.flow,
                "flow_masks": flow_decoded.masks,
                "flow_segmentations": flow_decoded.segmentations,
                "flow_slots": flow_decoded.flow_slots,
                "flow_masked": flow_decoded.flow_masked,
            }
        )

    def _add_flow_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        flow_target: Optional[torch.Tensor],
        reference: torch.Tensor,
        frame_padding_mask: Optional[torch.Tensor],
    ) -> None:
        if flow_target is None:
            outputs["flow_reconstruction_loss"] = reference.new_zeros(())
            return
        if self.flow_loss is None or "flow" not in outputs:
            raise ValueError("flow_target was provided but flow decoder is not configured.")
        flow_loss = self.flow_loss(
            outputs["flow"],
            flow_target,
            padding_mask=frame_padding_mask,
        )
        outputs["flow_reconstruction_loss"] = flow_loss
        outputs["loss"] = outputs["loss"] + flow_loss

    def forward(
        self,
        *,
        frame_tokens: Optional[torch.Tensor] = None,
        video_tokens: Optional[torch.Tensor] = None,
        grid_shape: Optional[GridShape] = None,
        semantic_queries: Optional[torch.Tensor] = None,
        contrastive_loss_mask: Optional[torch.Tensor] = None,
        flow_target: Optional[torch.Tensor] = None,
        frame_padding_mask: Optional[torch.Tensor] = None,
        initial_slots: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        decode: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if compute_loss and not decode:
            raise ValueError("decode=False is only valid when compute_loss=False.")
        target_tokens = self._target_from_inputs(frame_tokens, video_tokens, grid_shape)
        if target_tokens.shape[-2] != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches per time step, got {target_tokens.shape[-2]}."
            )

        slots = self.encoder(
            frame_tokens=frame_tokens,
            video_tokens=video_tokens,
            grid_shape=grid_shape,
            initial_slots=initial_slots,
            semantic_queries=semantic_queries,
            semantic_query_mask=contrastive_loss_mask,
        )

        if frame_padding_mask is not None:
            if frame_padding_mask.shape != slots.shape[:2]:
                raise ValueError(
                    f"frame_padding_mask must be [B,T]={slots.shape[:2]}, "
                    f"got {tuple(frame_padding_mask.shape)}."
                )
            if frame_padding_mask.device != slots.device:
                raise ValueError("frame_padding_mask must be on the same device as slots.")
            frame_padding_mask = frame_padding_mask.to(dtype=torch.bool)

        if not decode:
            return {"slots": slots, "target_tokens": target_tokens}

        condition_info = semantic_queries if self.decoder_conditioned else None
        decoded = self.decoder(slots, condition_info=condition_info)

        outputs: Dict[str, torch.Tensor] = {
            "slots": slots,
            "reconstruction": decoded.reconstruction,
            "masks": decoded.masks,
            "object_reconstructions": decoded.object_reconstructions,
            "per_slot_reconstructions": decoded.per_slot_reconstructions,
            "target_tokens": target_tokens,
        }
        self._add_flow_outputs(outputs, slots)

        if semantic_queries is not None:
            # Contrastive supervision uses only valid recurrent time steps.
            if frame_padding_mask is None:
                slot_summary = slots.mean(dim=1)
            else:
                valid = (~frame_padding_mask).to(slots.dtype).unsqueeze(-1).unsqueeze(-1)
                count = valid.sum(dim=1).clamp_min(1.0)
                slot_summary = (slots * valid).sum(dim=1) / count
            slot_embeddings = self.slot_projector(slot_summary)
            query_embeddings = self.query_projector(semantic_queries)
            outputs["slot_embeddings"] = slot_embeddings
            outputs["query_embeddings"] = query_embeddings

            if compute_loss:
                if contrastive_loss_mask is None:
                    raise ValueError("contrastive_loss_mask is required when compute_loss=True.")
                outputs.update(
                    self.loss_fn(
                        reconstruction=decoded.reconstruction,
                        target_tokens=target_tokens,
                        slot_embeddings=slot_embeddings,
                        query_embeddings=query_embeddings,
                        contrastive_loss_mask=contrastive_loss_mask,
                        frame_padding_mask=frame_padding_mask,
                    )
                )
                self._add_flow_loss(
                    outputs,
                    flow_target,
                    decoded.reconstruction,
                    frame_padding_mask,
                )
        elif compute_loss:
            recon_loss = self.loss_fn.reconstruction(
                decoded.reconstruction,
                target_tokens,
                padding_mask=frame_padding_mask,
            )
            outputs["loss"] = recon_loss
            outputs["reconstruction_loss"] = recon_loss
            outputs["slot_lang_contrastive_loss"] = recon_loss.new_zeros(())
            self._add_flow_loss(
                outputs,
                flow_target,
                decoded.reconstruction,
                frame_padding_mask,
            )

        return outputs
