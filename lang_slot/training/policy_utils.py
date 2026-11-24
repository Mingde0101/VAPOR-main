"""Shared utilities for Bridge flow-matching policy training scripts."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml

from lang_slot.paths import resolve_config_paths, resolve_path_string
from lang_slot.data import denormalize_policy_actions, load_policy_normalization_stats
from lang_slot.models import (
    FlowMatchingPolicy,
    PolicyConditionEncoder,
    VJEPA2FramewiseHistoryWrapper,
    VJEPA2FutureVideoWrapper,
    VJEPA2SlotFlowWrapper,
    VideoSlotAutoEncoder,
)
from lang_slot.modules import ConditionAggregationTransformer, FlowMatchingActionDecoder
from lang_slot.text import DEFAULT_SIGLIP2_MODEL_DIR, SigLIP2TextEmbedder, dtype_from_name
from lang_slot.vision import (
    DEFAULT_VJEPA2_MODEL_DIR,
    DEFAULT_VJEPA21_CHECKPOINT,
    DEFAULT_VJEPA21_REPO_DIR,
    VJEPA2VideoEncoder,
    VJEPA21VideoEncoder,
)


def _deep_merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings while replacing scalars and lists."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_config_file(path: Path, stack: Tuple[Path, ...]) -> Dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"Cyclic YAML config inheritance: {chain}")
    with resolved_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg)!r} in {resolved_path}.")

    parent_specs = cfg.pop("extends", None)
    if parent_specs is None:
        return cfg
    if isinstance(parent_specs, str):
        parent_specs = [parent_specs]
    if not isinstance(parent_specs, list) or not all(isinstance(item, str) for item in parent_specs):
        raise ValueError(f"Config 'extends' must be a path or list of paths in {resolved_path}.")

    merged: Dict[str, Any] = {}
    next_stack = (*stack, resolved_path)
    for parent_spec in parent_specs:
        parent_path = Path(parent_spec).expanduser()
        if not parent_path.is_absolute():
            parent_path = resolved_path.parent / parent_path
        merged = _deep_merge_config(merged, _load_yaml_config_file(parent_path, next_stack))
    return _deep_merge_config(merged, cfg)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a policy YAML, including optional relative ``extends`` parents."""
    return resolve_config_paths(_load_yaml_config_file(Path(path), ()))


def save_yaml_config(config: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def get_nested(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    cur = config
    parts = path.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def parse_shard_indices(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None or value == "" or value == "null":
        return None
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(v) for v in value)


def resolve_device(device: str) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_policy_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def attach_text_features(
    batch: Dict[str, Any],
    text_embedder: SigLIP2TextEmbedder,
    *,
    num_slots: int,
    device: torch.device,
) -> Dict[str, Any]:
    encoded = text_embedder.encode_policy_text_batch(batch["instruction"], num_slots=num_slots)
    out = dict(batch)
    out["task_semantic_tokens"] = encoded.task_semantic_tokens.to(device, non_blocking=True)
    out["semantic_queries"] = encoded.semantic_queries.to(device, non_blocking=True)
    out["contrastive_loss_mask"] = encoded.contrastive_loss_mask.to(device, non_blocking=True)
    out["object_nouns"] = encoded.object_nouns
    return out


def build_text_embedder(config: Dict[str, Any], device: torch.device) -> SigLIP2TextEmbedder:
    text_cfg = config.get("text", {})
    return SigLIP2TextEmbedder(
        model_dir=resolve_path_string(text_cfg.get("model_dir", DEFAULT_SIGLIP2_MODEL_DIR)),
        device=str(device),
        torch_dtype=dtype_from_name(text_cfg.get("torch_dtype", None)),
        normalize=True,
        local_files_only=True,
    )


def build_vjepa2_encoder(vjepa_cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build V-JEPA2.0 or V-JEPA2.1 behind the existing token interface."""
    backend = str(vjepa_cfg.get("backend", "vjepa2_0")).strip().lower().replace("-", "_")
    shared_kwargs = {
        "tubelet_size": int(vjepa_cfg.get("tubelet_size", 2)),
        "freeze": bool(vjepa_cfg.get("freeze", True)),
        "torch_dtype": dtype_from_name(vjepa_cfg.get("torch_dtype", None)),
    }
    if backend in {"vjepa2_0", "vjepa2", "hf"}:
        return VJEPA2VideoEncoder(
            model_dir=resolve_path_string(vjepa_cfg.get("model_dir", DEFAULT_VJEPA2_MODEL_DIR)),
            image_size=int(vjepa_cfg.get("image_size", 256)),
            resize_shortest_edge=vjepa_cfg.get("resize_shortest_edge", 292),
            local_files_only=True,
            **shared_kwargs,
        )
    if backend in {"vjepa2_1", "vjepa21"}:
        return VJEPA21VideoEncoder(
            checkpoint_path=resolve_path_string(
                vjepa_cfg.get("checkpoint", DEFAULT_VJEPA21_CHECKPOINT)
            ),
            repo_dir=resolve_path_string(vjepa_cfg.get("repo_dir", DEFAULT_VJEPA21_REPO_DIR)),
            image_size=int(vjepa_cfg.get("image_size", 384)),
            resize_shortest_edge=vjepa_cfg.get("resize_shortest_edge", 438),
            patch_size=int(vjepa_cfg.get("patch_size", 16)),
            num_frames=int(vjepa_cfg.get("num_frames", 64)),
            checkpoint_key=str(vjepa_cfg.get("checkpoint_key", "ema_encoder")),
            pretrained=bool(vjepa_cfg.get("pretrained", True)),
            **shared_kwargs,
        )
    raise ValueError(
        f"Unsupported vjepa2.backend={backend!r}; expected 'vjepa2_0' or 'vjepa2_1'."
    )


def validate_vjepa2_downstream_config(
    vision_encoder: torch.nn.Module,
    *,
    slot_cfg: Dict[str, Any],
    fusion_cfg: Dict[str, Any],
    use_slot_tokens: bool,
) -> Tuple[int, int, int, int]:
    """Validate the V-JEPA feature dimension and spatial grid consumers."""
    image_size = int(getattr(vision_encoder, "image_size"))
    patch_size = int(getattr(vision_encoder, "patch_size"))
    hidden_size = int(getattr(vision_encoder, "hidden_size"))
    if image_size % patch_size != 0:
        raise ValueError(
            f"V-JEPA2 image_size={image_size} must be divisible by patch_size={patch_size}."
        )
    grid_h = grid_w = image_size // patch_size
    num_patches = grid_h * grid_w

    fusion_video_dim = int(fusion_cfg.get("d_video", hidden_size))
    if fusion_video_dim != hidden_size:
        raise ValueError(
            f"condition_fusion.d_video={fusion_video_dim} does not match "
            f"V-JEPA2 hidden_size={hidden_size}."
        )
    max_spatial_h = int(fusion_cfg.get("max_spatial_h", grid_h))
    max_spatial_w = int(fusion_cfg.get("max_spatial_w", grid_w))
    if max_spatial_h < grid_h or max_spatial_w < grid_w:
        raise ValueError(
            "condition_fusion.max_spatial_h/w must cover the V-JEPA2 token grid: "
            f"configured={(max_spatial_h, max_spatial_w)}, required={(grid_h, grid_w)}."
        )

    if use_slot_tokens:
        slot_input_dim = int(slot_cfg.get("input_dim", hidden_size))
        slot_num_patches = int(slot_cfg.get("num_patches", num_patches))
        if slot_input_dim != hidden_size:
            raise ValueError(
                f"slot.input_dim={slot_input_dim} does not match V-JEPA2 hidden_size={hidden_size}."
            )
        if slot_num_patches != num_patches:
            raise ValueError(
                f"slot.num_patches={slot_num_patches} does not match the V-JEPA2 "
                f"grid {(grid_h, grid_w)} ({num_patches} patches)."
            )
    return hidden_size, num_patches, grid_h, grid_w


def build_policy_from_config(config: Dict[str, Any], device: torch.device) -> FlowMatchingPolicy:
    validate_policy_config(config)
    data_cfg = config.get("data", {})
    vjepa_cfg = config.get("vjepa2", {})
    slot_cfg = config.get("slot", {})
    fusion_cfg = config.get("condition_fusion", {})
    action_cfg = config.get("action_decoder", {})
    use_slot_tokens = bool(slot_cfg.get("enabled", True)) and bool(fusion_cfg.get("use_slot_tokens", True))

    shared_vjepa2 = build_vjepa2_encoder(vjepa_cfg)
    vjepa_hidden_size, vjepa_num_patches, vjepa_grid_h, vjepa_grid_w = (
        validate_vjepa2_downstream_config(
            shared_vjepa2,
            slot_cfg=slot_cfg,
            fusion_cfg=fusion_cfg,
            use_slot_tokens=use_slot_tokens,
        )
    )

    history_encoder = VJEPA2FramewiseHistoryWrapper(
        history_frames=int(data_cfg.get("history_frames", 4)),
        vision_encoder=shared_vjepa2,
        window_batch_size=vjepa_cfg.get("window_batch_size", None),
    )

    future_slot_encoder = None
    future_video_encoder = None
    if use_slot_tokens:
        slot_model = VideoSlotAutoEncoder(
            input_dim=int(slot_cfg.get("input_dim", vjepa_hidden_size)),
            slot_dim=int(slot_cfg.get("slot_dim", 256)),
            num_slots=int(slot_cfg.get("num_slots", 5)),
            num_iterations=int(slot_cfg.get("num_iterations", 3)),
            num_patches=int(slot_cfg.get("num_patches", vjepa_num_patches)),
            query_dim=int(slot_cfg.get("query_dim", 1536)),
            predictor_type=slot_cfg.get("predictor_type", "transformer"),
            num_predictor_layers=int(slot_cfg.get("num_predictor_layers", 1)),
            num_predictor_heads=int(slot_cfg.get("num_predictor_heads", 4)),
            slot_mlp_hidden_dim=slot_cfg.get("slot_mlp_hidden_dim", None),
            decoder_hidden_dim=int(slot_cfg.get("decoder_hidden_dim", 2048)),
            decoder_num_layers=int(slot_cfg.get("decoder_num_layers", 3)),
            embedding_dim=int(slot_cfg.get("embedding_dim", 4096)),
            recon_loss_weight=float(slot_cfg.get("recon_loss_weight", 0.3)),
            contrastive_loss_weight=float(slot_cfg.get("contrastive_loss_weight", 0.5)),
            contrastive_matching=slot_cfg.get("contrastive_matching", "index"),
            flow_output_resolution=(
                (int(vjepa_cfg.get("image_size", 256)), int(vjepa_cfg.get("image_size", 256)))
                if bool(slot_cfg.get("use_flow", False))
                else None
            ),
        )

        future_slot_encoder = VJEPA2SlotFlowWrapper(
            input_dim=int(slot_cfg.get("input_dim", vjepa_hidden_size)),
            slot_dim=int(slot_cfg.get("slot_dim", 256)),
            num_slots=int(slot_cfg.get("num_slots", 5)),
            num_iterations=int(slot_cfg.get("num_iterations", 3)),
            num_patches=int(slot_cfg.get("num_patches", vjepa_num_patches)),
            query_dim=int(slot_cfg.get("query_dim", 1536)),
            encoding_mode="sliding_pair",
            use_flow=bool(slot_cfg.get("use_flow", False)),
            vision_encoder=shared_vjepa2,
            slot_model=slot_model,
            window_batch_size=vjepa_cfg.get("window_batch_size", None),
        )
    else:
        future_video_encoder = VJEPA2FutureVideoWrapper(
            future_steps=int(data_cfg.get("future_steps", 4)),
            vision_encoder=shared_vjepa2,
            window_batch_size=vjepa_cfg.get("window_batch_size", None),
        )

    condition_fusion = ConditionAggregationTransformer(
        d_video=int(fusion_cfg.get("d_video", vjepa_hidden_size)),
        d_state=int(fusion_cfg.get("d_state", 7)),
        d_action=int(fusion_cfg.get("d_action", 7)),
        d_slot=int(fusion_cfg.get("d_slot", 256)),
        d_text=int(fusion_cfg.get("d_text", 1536)),
        d_model=int(fusion_cfg.get("d_model", 512)),
        num_layers=int(fusion_cfg.get("num_layers", 4)),
        num_heads=int(fusion_cfg.get("num_heads", 8)),
        mlp_ratio=float(fusion_cfg.get("mlp_ratio", 4.0)),
        dropout=float(fusion_cfg.get("dropout", 0.0)),
        max_hist_len=int(fusion_cfg.get("max_hist_len", data_cfg.get("history_frames", 4))),
        max_future_len=int(fusion_cfg.get("max_future_len", data_cfg.get("future_steps", 4))),
        max_spatial_h=int(fusion_cfg.get("max_spatial_h", vjepa_grid_h)),
        max_spatial_w=int(fusion_cfg.get("max_spatial_w", vjepa_grid_w)),
        use_slot_tokens=use_slot_tokens,
    )

    condition_encoder = PolicyConditionEncoder(
        history_encoder=history_encoder,
        future_slot_encoder=future_slot_encoder,
        future_video_encoder=future_video_encoder,
        condition_fusion=condition_fusion,
        freeze_future_slot_encoder=bool(slot_cfg.get("freeze", False)),
        use_slot_tokens=use_slot_tokens,
    )
    action_decoder = FlowMatchingActionDecoder(
        action_dim=int(action_cfg.get("action_dim", 7)),
        action_horizon=int(action_cfg.get("action_horizon", data_cfg.get("future_steps", 4))),
        condition_dim=int(action_cfg.get("condition_dim", fusion_cfg.get("d_model", 512))),
        d_model=int(action_cfg.get("d_model", 512)),
        num_layers=int(action_cfg.get("num_layers", 4)),
        num_heads=int(action_cfg.get("num_heads", 8)),
        mlp_ratio=float(action_cfg.get("mlp_ratio", 4.0)),
        dropout=float(action_cfg.get("dropout", 0.0)),
        loss_type=action_cfg.get("loss_type", "mse"),
        flow_time_eps=float(action_cfg.get("flow_time_eps", 1e-4)),
    )
    policy = FlowMatchingPolicy(
        condition_encoder=condition_encoder,
        action_decoder=action_decoder,
        slot_loss_weight=float(slot_cfg.get("auxiliary_loss_weight", 1.0)) if use_slot_tokens else 0.0,
    )
    if use_slot_tokens and bool(slot_cfg.get("freeze", False)):
        for param in policy.condition_encoder.future_slot_encoder.slot_model.parameters():
            param.requires_grad_(False)
    return policy.to(device)


def validate_policy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate architecture-defining fields for the original policy code.

    The v2 H10 policies use additional temporal and slot-encoding behavior.
    Rejecting those fields here is intentional: silently ignoring them can
    construct a different model and then partially load the wrong checkpoint.
    """
    data_cfg = config.get("data", {}) or {}
    slot_cfg = config.get("slot", {}) or {}
    fusion_cfg = config.get("condition_fusion", {}) or {}
    action_cfg = config.get("action_decoder", {}) or {}

    history_frames = int(data_cfg.get("history_frames", 4))
    future_steps = int(data_cfg.get("future_steps", 4))
    action_horizon = int(action_cfg.get("action_horizon", future_steps))
    if history_frames <= 0 or future_steps <= 0 or action_horizon <= 0:
        raise ValueError("history_frames, future_steps, and action_horizon must be positive.")
    if action_horizon != future_steps:
        raise ValueError(
            f"action_decoder.action_horizon={action_horizon} must equal data.future_steps={future_steps}."
        )

    max_hist_len = int(fusion_cfg.get("max_hist_len", history_frames))
    max_future_len = int(fusion_cfg.get("max_future_len", future_steps))
    if max_hist_len < history_frames:
        raise ValueError(
            f"condition_fusion.max_hist_len={max_hist_len} is smaller than history_frames={history_frames}."
        )
    if max_future_len < future_steps:
        raise ValueError(
            f"condition_fusion.max_future_len={max_future_len} is smaller than future_steps={future_steps}."
        )

    slot_enabled = bool(slot_cfg.get("enabled", True))
    use_slot_tokens = bool(fusion_cfg.get("use_slot_tokens", True))
    if slot_enabled != use_slot_tokens:
        raise ValueError(
            "slot.enabled and condition_fusion.use_slot_tokens must agree; "
            f"got {slot_enabled} and {use_slot_tokens}."
        )
    if not slot_enabled and bool(slot_cfg.get("compute_auxiliary_loss", False)):
        raise ValueError("slot.compute_auxiliary_loss must be false when slots are disabled.")

    unsupported: List[str] = []
    if int(data_cfg.get("history_image_frames", history_frames)) != history_frames:
        unsupported.append("data.history_image_frames")
    if int(data_cfg.get("history_state_frames", history_frames)) != history_frames:
        unsupported.append("data.history_state_frames")
    if bool(data_cfg.get("history_images_include_current", False)):
        unsupported.append("data.history_images_include_current")
    if bool(data_cfg.get("history_states_include_current", False)):
        unsupported.append("data.history_states_include_current")
    if not bool(data_cfg.get("future_images_include_current", True)):
        unsupported.append("data.future_images_include_current=false")
    if str(data_cfg.get("future_encoding_mode", "sliding_pair")) != "sliding_pair":
        unsupported.append("data.future_encoding_mode")
    if str(slot_cfg.get("encoding_mode", "sliding_pair")) != "sliding_pair":
        unsupported.append("slot.encoding_mode")
    if bool(slot_cfg.get("include_current_frame", False)):
        unsupported.append("slot.include_current_frame")
    if str(slot_cfg.get("semantic_contrastive_source", "slot_summary")) != "slot_summary":
        unsupported.append("slot.semantic_contrastive_source")
    if unsupported:
        raise ValueError(
            "This config requires the slot-attention-video-v2 implementation; "
            "the original implementation cannot honor: " + ", ".join(unsupported)
        )

    return {
        "slot_enabled": slot_enabled,
        "action_semantics": (
            "state_delta" if bool(data_cfg.get("state_delta_action", False)) else "raw_action"
        ),
        "history_frames": history_frames,
        "future_steps": future_steps,
        "action_horizon": action_horizon,
    }


def set_frozen_backbones_eval(policy: FlowMatchingPolicy) -> None:
    # ``model.train()`` would otherwise flip the frozen V-JEPA2 backbone back to train mode.
    vision = policy.condition_encoder.history_encoder.vision_encoder
    if not any(param.requires_grad for param in vision.parameters()):
        vision.eval()
    if bool(getattr(policy.condition_encoder, "use_slot_tokens", True)) and bool(
        getattr(policy.condition_encoder, "freeze_future_slot_encoder", False)
    ):
        policy.condition_encoder.future_slot_encoder.eval()


def _matches_any_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def checkpoint_model_state_dict(
    model: torch.nn.Module,
    *,
    include_frozen_prefixes: Sequence[str] = (),
) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    prefixes = tuple(str(prefix) for prefix in include_frozen_prefixes if str(prefix))
    return {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name in trainable_names or (prefixes and _matches_any_prefix(name, prefixes))
    }


def trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return checkpoint_model_state_dict(model)


def load_trainable_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    current = model.state_dict()
    filtered = {name: tensor for name, tensor in state.items() if name in current}
    current.update(filtered)
    model.load_state_dict(current, strict=True)
    return len(filtered), len(state) - len(filtered)


def _as_prefix_tuple(value: Any) -> Tuple[str, ...]:
    if value is None or value == "" or value == "null":
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


def apply_trainable_prefixes(model: torch.nn.Module, prefixes: Any) -> Dict[str, Any]:
    """Restrict trainable parameters to names matching ``prefixes``.

    An empty prefix list leaves the model unchanged. This is intentionally based
    on full ``named_parameters`` prefixes so experiments can train only a head
    such as ``action_decoder.`` without adding module-specific switches.
    """

    include = _as_prefix_tuple(prefixes)
    if not include:
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        return {
            "enabled": False,
            "include_prefixes": [],
            "trainable_parameter_count": int(trainable_params),
            "total_parameter_count": int(total_params),
            "trainable_names": None,
            "frozen_parameter_count": int(total_params - trainable_params),
        }

    trainable_names: List[str] = []
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        is_trainable = _matches_any_prefix(name, include)
        param.requires_grad_(is_trainable)
        if is_trainable:
            trainable_names.append(name)
            trainable_count += int(param.numel())
        else:
            frozen_count += int(param.numel())
    if not trainable_names:
        raise ValueError(f"No parameters matched training.trainable_prefixes={list(include)!r}.")
    return {
        "enabled": True,
        "include_prefixes": list(include),
        "trainable_parameter_count": int(trainable_count),
        "total_parameter_count": int(trainable_count + frozen_count),
        "frozen_parameter_count": int(frozen_count),
        "trainable_names": trainable_names[:50],
        "matched_parameter_tensors": int(len(trainable_names)),
    }


def load_partial_trainable_state_dict(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    *,
    include_prefixes: Sequence[str],
    exclude_prefixes: Sequence[str] = (),
    strict_shapes: bool = True,
) -> Dict[str, Any]:
    """Load a selected subset of trainable checkpoint tensors into ``model``.

    This is intended for architecture-compatible warm starts, e.g. reusing the
    visual slot stack while reinitializing policy fusion/action heads. It never
    loads optimizer, scheduler, scaler, or RNG state.
    """

    include = tuple(str(prefix) for prefix in include_prefixes if str(prefix))
    exclude = tuple(str(prefix) for prefix in exclude_prefixes if str(prefix))
    if not include:
        raise ValueError("partial checkpoint loading requires at least one include prefix.")

    current = model.state_dict()
    selected: Dict[str, torch.Tensor] = {}
    missing: List[str] = []
    shape_mismatch: List[Dict[str, Any]] = []
    excluded = 0
    outside_include = 0

    for name, tensor in state.items():
        if not _matches_any_prefix(name, include):
            outside_include += 1
            continue
        if exclude and _matches_any_prefix(name, exclude):
            excluded += 1
            continue
        if name not in current:
            missing.append(name)
            continue
        if hasattr(tensor, "shape") and hasattr(current[name], "shape") and tuple(tensor.shape) != tuple(current[name].shape):
            shape_mismatch.append(
                {
                    "name": name,
                    "checkpoint_shape": tuple(tensor.shape),
                    "model_shape": tuple(current[name].shape),
                }
            )
            if strict_shapes:
                continue
        selected[name] = tensor

    if selected:
        updated = dict(current)
        updated.update(selected)
        model.load_state_dict(updated, strict=True)

    return {
        "loaded_tensors": len(selected),
        "checkpoint_tensors": len(state),
        "outside_include_tensors": outside_include,
        "excluded_tensors": excluded,
        "missing_tensors": len(missing),
        "shape_mismatch_tensors": len(shape_mismatch),
        "include_prefixes": list(include),
        "exclude_prefixes": list(exclude),
        "missing_names": missing[:50],
        "shape_mismatches": shape_mismatch[:50],
    }


def load_partial_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    include_prefixes: Sequence[str],
    exclude_prefixes: Sequence[str] = (),
    strict_shapes: bool = True,
    expected_state_delta_action: Optional[bool] = None,
    expected_checkpoint_epoch: Optional[int] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Warm-start selected model tensors from a local training checkpoint."""

    load_kwargs: Dict[str, Any] = {
        "map_location": map_location,
        "weights_only": False,
    }
    # ``mmap=True`` avoids eagerly reading optimizer/storage blobs from large
    # checkpoints. Older PyTorch versions do not expose the argument.
    if "mmap" in inspect.signature(torch.load).parameters:
        load_kwargs["mmap"] = True
    checkpoint = torch.load(path, **load_kwargs)
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_config = checkpoint.get("config", {}) or {}
    checkpoint_data_config = checkpoint_config.get("data", {}) or {}
    # Historical configs without this field predate state-delta relabeling and
    # therefore used raw Bridge actions.
    checkpoint_state_delta_action = bool(
        checkpoint_data_config.get("state_delta_action", False)
    )
    if (
        expected_state_delta_action is not None
        and checkpoint_state_delta_action != bool(expected_state_delta_action)
    ):
        expected_name = "state_delta" if expected_state_delta_action else "raw_action"
        actual_name = "state_delta" if checkpoint_state_delta_action else "raw_action"
        raise ValueError(
            f"Partial checkpoint action semantics mismatch: expected {expected_name}, "
            f"checkpoint is {actual_name}: {path}"
        )
    if expected_checkpoint_epoch is not None and (
        checkpoint_epoch is None
        or int(checkpoint_epoch) != int(expected_checkpoint_epoch)
    ):
        raise ValueError(
            "Partial checkpoint epoch mismatch: "
            f"expected {expected_checkpoint_epoch}, checkpoint has {checkpoint_epoch}: {path}"
        )
    state = checkpoint.get("model_trainable", {})
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a dict field 'model_trainable'.")
    result = load_partial_trainable_state_dict(
        model,
        state,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        strict_shapes=strict_shapes,
    )
    result.update(
        {
            "path": str(path),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_samples_seen": checkpoint.get("samples_seen"),
            "checkpoint_state_delta_action": checkpoint_state_delta_action,
        }
    )
    return result


def maybe_load_partial_checkpoint_from_config(
    model: torch.nn.Module,
    config: Dict[str, Any],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Optional[Dict[str, Any]]:
    partial_cfg = config.get("partial_init", {}) or {}
    path = partial_cfg.get("checkpoint", None)
    if not path:
        return None
    include_prefixes = _as_prefix_tuple(partial_cfg.get("include_prefixes"))
    exclude_prefixes = _as_prefix_tuple(partial_cfg.get("exclude_prefixes"))
    require_semantics_match = bool(
        partial_cfg.get("require_action_semantics_match", False)
    )
    return load_partial_checkpoint(
        str(path),
        model=model,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        strict_shapes=bool(partial_cfg.get("strict_shapes", True)),
        expected_state_delta_action=(
            bool((config.get("data", {}) or {}).get("state_delta_action", False))
            if require_semantics_match
            else None
        ),
        expected_checkpoint_epoch=partial_cfg.get("expected_checkpoint_epoch"),
        map_location=map_location,
    )


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"].cpu())
    if torch.cuda.is_available() and "torch_cuda_random" in state:
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["torch_cuda_random"]])


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    scheduler: Optional[Any] = None,
    config: Dict[str, Any],
    global_step: int,
    samples_seen: int,
    epoch: int,
    data_cursor: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    include_frozen_prefixes = _as_prefix_tuple(
        (config.get("checkpoint", {}) or {}).get("include_frozen_prefixes")
    )
    slot_cfg = config.get("slot", {}) or {}
    fusion_cfg = config.get("condition_fusion", {}) or {}
    use_slot_tokens = bool(slot_cfg.get("enabled", True)) and bool(fusion_cfg.get("use_slot_tokens", True))
    if use_slot_tokens and bool(slot_cfg.get("freeze", False)):
        include_frozen_prefixes = tuple(
            dict.fromkeys(
                (
                    *include_frozen_prefixes,
                    "condition_encoder.future_slot_encoder.slot_model.",
                )
            )
        )
    payload = {
        "model_trainable": checkpoint_model_state_dict(
            model,
            include_frozen_prefixes=include_frozen_prefixes,
        ),
        "model_checkpoint_include_frozen_prefixes": list(include_frozen_prefixes),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": config,
        "global_step": int(global_step),
        "samples_seen": int(samples_seen),
        "epoch": int(epoch),
        "data_cursor": data_cursor,
        "rng_state": rng_state(),
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    # Checkpoints are produced locally and intentionally include optimizer/RNG/config
    # state, so PyTorch 2.6+ must not use weights_only=True here.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    loaded, skipped = load_trainable_state_dict(model, checkpoint.get("model_trainable", {}))
    checkpoint["loaded_trainable_tensors"] = loaded
    checkpoint["skipped_trainable_tensors"] = skipped
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            checkpoint["optimizer_loaded"] = True
        except ValueError as exc:
            checkpoint["optimizer_loaded"] = False
            checkpoint["optimizer_load_error"] = str(exc)
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
            checkpoint["scheduler_loaded"] = True
        except ValueError as exc:
            checkpoint["scheduler_loaded"] = False
            checkpoint["scheduler_load_error"] = str(exc)
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint.get("rng_state", {}))
    return checkpoint


def latest_checkpoint(output_dir: str) -> Optional[str]:
    latest = Path(output_dir) / "checkpoints" / "latest.pt"
    return str(latest) if latest.exists() else None


def build_optimizer(config: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = config.get("optimizer", {})
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found for optimizer.")
    cls = optim_cfg.get("class", "AdamW")
    if cls != "AdamW":
        raise ValueError(f"Only AdamW is currently supported, got {cls!r}.")
    return torch.optim.AdamW(
        params,
        lr=float(optim_cfg.get("lr", 1e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
        betas=tuple(float(v) for v in optim_cfg.get("betas", [0.9, 0.95])),
        eps=float(optim_cfg.get("eps", 1e-8)),
    )


def maybe_denormalize_actions(
    actions: torch.Tensor,
    config: Dict[str, Any],
    *,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    stats = load_policy_normalization_stats(
        data_dir=get_nested(config, "data.path"),
        normalization_stats_path=get_nested(config, "data.normalization_stats_path"),
    )
    out = denormalize_policy_actions(
        actions,
        stats,
        eps=float(get_nested(config, "data.normalization_eps", 1e-6)),
    )
    if padding_mask is not None:
        out = out.masked_fill(padding_mask.to(device=out.device, dtype=torch.bool).unsqueeze(-1), 0.0)
    return out


def json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
