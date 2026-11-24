"""Losses for language-conditioned video slot pretraining."""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    """Token reconstruction loss for V-JEPA2/RADIO-style patch features."""

    def __init__(
        self,
        loss_type: str = "mse",
        weight: float = 1.0,
        normalize_prediction: bool = False,
        normalize_target: bool = False,
    ) -> None:
        super().__init__()
        if loss_type not in {"mse", "l1", "cosine"}:
            raise ValueError('loss_type must be one of "mse", "l1", or "cosine".')
        self.loss_type = loss_type
        self.weight = float(weight)
        self.normalize_prediction = bool(normalize_prediction)
        self.normalize_target = bool(normalize_target)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Reconstruction shape mismatch: "
                f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}."
            )
        target = target.detach()
        if self.normalize_prediction:
            prediction = F.normalize(prediction, dim=-1, eps=1e-8)
        if self.normalize_target:
            target = F.normalize(target, dim=-1, eps=1e-8)

        if self.loss_type == "mse":
            elementwise = (prediction - target).square()
        elif self.loss_type == "l1":
            elementwise = (prediction - target).abs()
        else:
            elementwise = 1.0 - F.cosine_similarity(prediction, target, dim=-1)

        if padding_mask is None:
            return self.weight * elementwise.mean()
        if padding_mask.shape != prediction.shape[:2]:
            raise ValueError(
                f"padding_mask must be [B,T]={prediction.shape[:2]}, got {tuple(padding_mask.shape)}."
            )
        valid = (~padding_mask.to(device=prediction.device, dtype=torch.bool)).to(elementwise.dtype)
        while valid.ndim < elementwise.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.expand_as(elementwise)
        denominator = valid.sum()
        if denominator.item() == 0:
            return prediction.sum() * 0.0
        return self.weight * (elementwise * valid).sum() / denominator


class DiagonalContrastiveLoss(nn.Module):
    """Contrastive loss between slot and language-query embeddings.

    ``slot_embeddings`` and ``query_embeddings`` are expected to have matching
    ``[B, K, D]`` shape. ``mask`` marks semantic slots that should contribute.
    ``matching="index"`` uses slot/query pairs with the same index.
    ``matching="hungarian"`` first matches active queries to unique slots within
    each sample by maximum cosine similarity, then treats matched pairs as
    positives.
    With ``batch_contrastive=True`` all active slots in the batch become
    negatives for one another, matching the OC-Manip pre-slot setup.
    """

    def __init__(
        self,
        weight: float = 1.0,
        temperature: float = 0.1,
        batch_contrastive: bool = True,
        symmetric: bool = False,
        l2_normalize: bool = True,
        matching: str = "index",
    ) -> None:
        super().__init__()
        if matching not in {"index", "hungarian"}:
            raise ValueError('matching must be "index" or "hungarian".')
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.batch_contrastive = bool(batch_contrastive)
        self.symmetric = bool(symmetric)
        self.l2_normalize = bool(l2_normalize)
        self.matching = matching

    @staticmethod
    def _fallback_assignment(scores: torch.Tensor) -> Tuple[List[int], List[int]]:
        num_slots, num_queries = scores.shape
        if num_queries > num_slots:
            raise ValueError(f"Cannot match {num_queries} queries to only {num_slots} slots.")
        best_score = None
        best_slots: Optional[Tuple[int, ...]] = None
        score_cpu = scores.detach().float().cpu()
        for slot_perm in permutations(range(num_slots), num_queries):
            value = sum(float(score_cpu[slot_index, query_index]) for query_index, slot_index in enumerate(slot_perm))
            if best_score is None or value > best_score:
                best_score = value
                best_slots = slot_perm
        assert best_slots is not None
        return list(best_slots), list(range(num_queries))

    @classmethod
    def _hungarian_assignment(cls, scores: torch.Tensor) -> Tuple[List[int], List[int]]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return cls._fallback_assignment(scores)
        row_ind, col_ind = linear_sum_assignment((-scores.detach().float().cpu()).numpy())
        return [int(index) for index in row_ind], [int(index) for index in col_ind]

    def _matched_embeddings(
        self,
        slot_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        active: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.matching == "index":
            return slot_embeddings[active], query_embeddings[active]

        matched_slots = []
        matched_queries = []
        batch_size, num_slots, _ = slot_embeddings.shape
        for batch_index in range(batch_size):
            query_indices = active[batch_index].nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0:
                continue
            if query_indices.numel() > num_slots:
                raise ValueError(
                    f"Cannot match {int(query_indices.numel())} active queries to {num_slots} slots."
                )
            scores = slot_embeddings[batch_index] @ query_embeddings[batch_index, query_indices].T
            slot_indices, query_columns = self._hungarian_assignment(scores)
            for slot_index, query_column in zip(slot_indices, query_columns):
                matched_slots.append(slot_embeddings[batch_index, slot_index])
                matched_queries.append(query_embeddings[batch_index, query_indices[query_column]])
        if not matched_slots:
            empty = slot_embeddings.new_empty(0, slot_embeddings.shape[-1])
            return empty, empty
        return torch.stack(matched_slots, dim=0), torch.stack(matched_queries, dim=0)

    def forward(
        self,
        slot_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if slot_embeddings.shape != query_embeddings.shape:
            raise ValueError(
                "Contrastive embeddings must have the same shape: "
                f"slot={tuple(slot_embeddings.shape)} query={tuple(query_embeddings.shape)}."
            )
        if mask.shape != slot_embeddings.shape[:2]:
            raise ValueError(f"mask must have shape [B,K], got {tuple(mask.shape)}.")
        if mask.device != slot_embeddings.device:
            raise ValueError("mask must be on the same device as embeddings.")

        if self.l2_normalize:
            slot_embeddings = F.normalize(slot_embeddings, dim=-1, eps=1e-8)
            query_embeddings = F.normalize(query_embeddings, dim=-1, eps=1e-8)

        active = mask.to(torch.bool)
        if active.count_nonzero() == 0:
            return slot_embeddings.new_zeros(())

        if self.batch_contrastive:
            slots, queries = self._matched_embeddings(slot_embeddings, query_embeddings, active)
            if slots.numel() == 0:
                return slot_embeddings.new_zeros(())
            logits = slots @ queries.T / self.temperature
            labels = torch.arange(slots.shape[0], device=slots.device)
            loss = F.cross_entropy(logits, labels)
            if self.symmetric:
                loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
            return self.weight * loss

        if self.matching == "hungarian":
            per_sample_losses = []
            for batch_index in range(slot_embeddings.shape[0]):
                query_indices = active[batch_index].nonzero(as_tuple=False).flatten()
                if query_indices.numel() == 0:
                    continue
                scores = slot_embeddings[batch_index] @ query_embeddings[batch_index, query_indices].T
                slot_indices, query_columns = self._hungarian_assignment(scores)
                logits = scores[slot_indices] / self.temperature
                labels = torch.tensor(query_columns, device=slot_embeddings.device, dtype=torch.long)
                per_sample_losses.append(F.cross_entropy(logits, labels))
            if not per_sample_losses:
                return slot_embeddings.new_zeros(())
            return self.weight * torch.stack(per_sample_losses).mean()

        logits = torch.bmm(slot_embeddings, query_embeddings.transpose(1, 2)) / self.temperature
        labels = torch.arange(slot_embeddings.shape[1], device=slot_embeddings.device)
        labels = labels.unsqueeze(0).expand(slot_embeddings.shape[0], -1).reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, reduction="none")
        loss = loss.reshape_as(mask)
        return self.weight * (loss * active.to(loss.dtype)).sum() / active.count_nonzero().clamp_min(1)


class VideoSlotPretrainingLoss(nn.Module):
    """Combine token reconstruction and slot-language contrastive losses."""

    def __init__(
        self,
        reconstruction: ReconstructionLoss,
        contrastive: DiagonalContrastiveLoss,
    ) -> None:
        super().__init__()
        self.reconstruction = reconstruction
        self.contrastive = contrastive

    def forward(
        self,
        reconstruction: torch.Tensor,
        target_tokens: torch.Tensor,
        slot_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        contrastive_loss_mask: torch.Tensor,
        frame_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        recon_loss = self.reconstruction(
            reconstruction,
            target_tokens,
            padding_mask=frame_padding_mask,
        )
        contrastive_loss = self.contrastive(slot_embeddings, query_embeddings, contrastive_loss_mask)
        total = recon_loss + contrastive_loss
        return {
            "loss": total,
            "reconstruction_loss": recon_loss,
            "slot_lang_contrastive_loss": contrastive_loss,
        }
