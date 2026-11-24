"""SigLIP2 text utilities for language-conditioned slots."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from lang_slot.paths import resources


DEFAULT_SIGLIP2_MODEL_DIR = str(resources()["SIGLIP2_DIR"])


def _filter_object_candidates(candidates: Sequence[str]) -> List[str]:
    filtered: List[str] = []
    for candidate in candidates:
        text = str(candidate).strip().strip("\"\' ").lower()
        if not text or "<|" in text:
            continue
        if "instruction:" in text or "return a json" in text:
            continue
        if text in {"system", "user", "assistant"}:
            continue
        filtered.append(text)
    return filtered


def parse_object_list(text: str, fallback: bool = True) -> List[str]:
    """Parse an explicit object list, falling back to comma/line splitting."""
    try:
        start = text.index("[")
        end = text.index("]", start)
        payload = text[start : end + 1]
        values = json.loads(payload)
        if isinstance(values, list):
            return _filter_object_candidates([str(v) for v in values])
    except Exception:
        pass

    if not fallback:
        return []

    candidates: List[str] = []
    for line in text.splitlines():
        line = line.strip("-•* 	")
        if line:
            candidates.extend(part.strip() for part in line.split(",") if part.strip())
    return _filter_object_candidates(candidates)


def heuristic_object_phrases(instruction: str, max_objects: int) -> List[str]:
    """Extract simple object phrases from Bridge-style robot instructions.

    This mirrors the lightweight OC-Manip fallback: remove common manipulation
    verbs/articles, split on prepositions/conjunctions, and deduplicate phrases.
    It is intentionally deterministic and does not call an LLM.
    """
    text = instruction.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    parts = re.split(
        r"\b(?:and|then|with|to|into|onto|on|in|at|from|of|near|behind|front|left|right)\b",
        text,
    )
    verbs = {
        "put", "place", "move", "pick", "lift", "close", "open", "push", "pull",
        "insert", "remove", "take", "bring", "set", "drop", "grasp", "hold", "turn",
        "up", "down",
    }
    articles = {"the", "a", "an", "it", "them"}
    objects: List[str] = []
    for part in parts:
        tokens = [token for token in part.strip().split() if token and token not in articles]
        while tokens and tokens[0] in verbs:
            tokens = tokens[1:]
        if tokens:
            objects.append(" ".join(tokens))

    deduped: List[str] = []
    seen = set()
    for obj in objects:
        if obj not in seen:
            seen.add(obj)
            deduped.append(obj)
    return deduped[:max_objects]


def dtype_from_name(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None or str(name).lower() in {"none", "auto"}:
        return None
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unknown dtype name {name!r}; expected one of {sorted(table)}.")
    return table[key]


@dataclass
class PolicyTextBatch:
    """Batched text features for policy conditioning and slot semantics."""

    task_semantic_tokens: torch.Tensor
    semantic_queries: torch.Tensor
    contrastive_loss_mask: torch.Tensor
    object_nouns: List[List[str]]


@dataclass
class ObjectQueryResult:
    objects: List[str]
    semantic_queries: torch.Tensor
    contrastive_loss_mask: torch.Tensor


class SigLIP2TextEmbedder:
    """Direct HF/Transformers SigLIP2 text embedder, no RADIO adaptor."""

    def __init__(
        self,
        model_dir: str = DEFAULT_SIGLIP2_MODEL_DIR,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        normalize: bool = True,
        local_files_only: bool = True,
        max_length: int = 64,
    ) -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("SigLIP2TextEmbedder requires transformers.") from exc

        self.model_dir = model_dir
        self.device = torch.device(device if device else "cpu")
        self.normalize = bool(normalize)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        config = getattr(self.model, "config", None)
        text_config = getattr(config, "text_config", None)
        self.output_dim = int(getattr(text_config, "projection_size", 1536))

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, self.output_dim)
        batch = self.tokenizer(
            list(texts),
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        features = self.model.get_text_features(**batch)
        if self.normalize:
            features = F.normalize(features, dim=-1)
        return features.float().cpu()

    def encode_task_semantics(self, instructions: Sequence[str]) -> torch.Tensor:
        """Encode full task instructions as condition tokens.

        This is separate from object-noun queries used by Slot Attention.
        """
        return self.encode_texts([str(text) for text in instructions])

    def encode_policy_text_batch(
        self,
        instructions: Sequence[str],
        num_slots: int,
        fallback_to_instruction: bool = True,
    ) -> PolicyTextBatch:
        """Encode full-task and object-noun SigLIP2 features for a batch.

        Returns:
            task_semantic_tokens: ``[B,D_text]`` full instruction embeddings.
            semantic_queries: ``[B,K,D_text]`` object noun embeddings for slots.
            contrastive_loss_mask: ``[B,K]`` active object-query mask.
            object_nouns: extracted noun phrases per instruction.
        """
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        instructions = [str(text) for text in instructions]
        batch_size = len(instructions)
        task_semantic_tokens = self.encode_task_semantics(instructions)

        object_nouns: List[List[str]] = []
        flat_objects: List[str] = []
        owners: List[Tuple[int, int]] = []
        for batch_index, instruction in enumerate(instructions):
            objects = heuristic_object_phrases(instruction, max_objects=num_slots)
            if not objects and fallback_to_instruction and instruction:
                objects = [instruction]
            objects = objects[:num_slots]
            object_nouns.append(objects)
            for slot_index, obj in enumerate(objects):
                owners.append((batch_index, slot_index))
                flat_objects.append(obj)

        semantic_queries = torch.zeros(
            batch_size,
            num_slots,
            self.output_dim,
            dtype=torch.float32,
        )
        contrastive_loss_mask = torch.zeros(batch_size, num_slots, dtype=torch.float32)
        if flat_objects:
            object_embeddings = self.encode_texts(flat_objects)
            for row, (batch_index, slot_index) in enumerate(owners):
                semantic_queries[batch_index, slot_index] = object_embeddings[row]
                contrastive_loss_mask[batch_index, slot_index] = 1.0

        return PolicyTextBatch(
            task_semantic_tokens=task_semantic_tokens,
            semantic_queries=semantic_queries,
            contrastive_loss_mask=contrastive_loss_mask,
            object_nouns=object_nouns,
        )

    def add_policy_text_to_batch(
        self,
        batch: Dict[str, object],
        *,
        num_slots: int,
        instruction_key: str = "instruction",
    ) -> Dict[str, object]:
        """Return a shallow batch copy with SigLIP2 policy text fields added.

        Expected input batch field:
            ``batch[instruction_key]``: list of B task strings.
        Added fields:
            ``task_semantic_tokens``, ``semantic_queries``,
            ``contrastive_loss_mask``, ``object_nouns``.
        """
        if instruction_key not in batch:
            raise KeyError(f"batch is missing instruction key {instruction_key!r}.")
        instructions = batch[instruction_key]
        if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
            raise ValueError(f"batch[{instruction_key!r}] must be a sequence of strings.")
        encoded = self.encode_policy_text_batch(instructions, num_slots=num_slots)
        output = dict(batch)
        output["task_semantic_tokens"] = encoded.task_semantic_tokens
        output["semantic_queries"] = encoded.semantic_queries
        output["contrastive_loss_mask"] = encoded.contrastive_loss_mask
        output["object_nouns"] = encoded.object_nouns
        return output

    def encode_object_queries(
        self,
        instruction: str,
        num_slots: int,
        fallback_to_instruction: bool = True,
    ) -> ObjectQueryResult:
        objects = heuristic_object_phrases(instruction, max_objects=num_slots)
        if not objects and fallback_to_instruction and instruction:
            objects = [instruction]

        embeddings = self.encode_texts(objects[:num_slots])
        semantic_queries = torch.zeros(num_slots, self.output_dim, dtype=torch.float32)
        mask = torch.zeros(num_slots, dtype=torch.float32)
        if embeddings.numel() > 0:
            count = min(num_slots, embeddings.shape[0])
            semantic_queries[:count] = embeddings[:count]
            mask[:count] = 1.0
        return ObjectQueryResult(
            objects=objects[:num_slots],
            semantic_queries=semantic_queries,
            contrastive_loss_mask=mask,
        )
