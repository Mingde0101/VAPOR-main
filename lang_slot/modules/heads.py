"""Projection heads used by language-slot training losses."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SlotProjectorHead(nn.Module):
    """Project slot features into a semantic embedding space."""

    def __init__(self, slot_dim: int, embedding_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * slot_dim
        self.net = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.net(slots)


class LangEmbeddingHead(nn.Module):
    """Project per-slot language/query embeddings into the same space as slots."""

    def __init__(self, query_dim: int, embedding_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        self.net = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, semantic_queries: torch.Tensor) -> torch.Tensor:
        semantic_queries = F.normalize(semantic_queries.float(), dim=-1, eps=1e-8)
        return self.net(semantic_queries)
