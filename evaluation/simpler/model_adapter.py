"""Adapter from SIMPLER rollout data to the existing Bridge policy forward path."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lang_slot.paths import check_policy_files, resources
from lang_slot.data import (  # noqa: E402
    denormalize_policy_actions,
    load_policy_normalization_stats,
    normalize_policy_state_action,
)
from lang_slot.training.policy_utils import (  # noqa: E402
    attach_text_features,
    build_policy_from_config,
    build_text_embedder,
    get_nested,
    load_checkpoint,
    load_yaml_config,
    move_policy_batch_to_device,
    resolve_device,
    set_seed,
)


def resize_frames_to_hw(frames: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize uint8 RGB frames to a shared NHWC size for batch assembly."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}.")
    if frames.shape[1] == height and frames.shape[2] == width:
        return np.ascontiguousarray(frames)

    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = []
    for frame in frames:
        frame_u8 = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
        image = Image.fromarray(frame_u8, mode="RGB")
        resized.append(np.asarray(image.resize((width, height), resampling), dtype=np.uint8))
    return np.ascontiguousarray(np.stack(resized, axis=0))


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown precision {precision!r}.")


class ModelAdapter:
    def __init__(
        self,
        *,
        config_path: str | Path = REPO_ROOT / "configs/simpler.yaml",
        checkpoint: str | Path = "",
        device: str = "",
        precision: str = "",
        sample_steps: int = 0,
        seed: int = 0,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = load_yaml_config(str(self.config_path))
        stats_path = get_nested(self.config, "data.normalization_stats_path", None)
        if stats_path and not Path(str(stats_path)).expanduser().is_absolute():
            self.config["data"]["normalization_stats_path"] = str(REPO_ROOT / str(stats_path))
        self.device = resolve_device(device or str(get_nested(self.config, "training.device", "cuda")))
        self.precision = precision or str(get_nested(self.config, "training.precision", "bf16"))
        self.sample_steps = int(sample_steps or get_nested(self.config, "action_decoder.sample_steps", 20))
        self.num_slots = int(get_nested(self.config, "text.num_slots", get_nested(self.config, "slot.num_slots", 5)))
        self.discretize_gripper = bool(get_nested(self.config, "action_decoder.discretize_gripper", False))
        self.gripper_index = int(get_nested(self.config, "action_decoder.gripper_index", 6))
        self.normalization_eps = float(get_nested(self.config, "data.normalization_eps", 1e-6))
        # A missing field identifies a legacy raw-action config from before
        # state-delta relabeling was introduced.
        self.state_delta_action = bool(get_nested(self.config, "data.state_delta_action", False))
        self.action_semantics = "state_delta" if self.state_delta_action else "raw_bridge_action"
        self.history_frames = int(get_nested(self.config, "data.history_frames", 4))
        self.history_image_frames = int(
            get_nested(self.config, "data.history_image_frames", self.history_frames)
        )
        self.history_state_frames = int(
            get_nested(self.config, "data.history_state_frames", self.history_frames)
        )
        self.future_steps = int(get_nested(self.config, "data.future_steps", 4))
        self.action_horizon = int(
            get_nested(self.config, "action_decoder.action_horizon", self.future_steps)
        )
        self.future_images_include_current = bool(
            get_nested(self.config, "data.future_images_include_current", True)
        )

        set_seed(int(seed or get_nested(self.config, "experiment.seed", 0)))
        ckpt = str(checkpoint or resources()["POLICY_CHECKPOINT"])
        if not ckpt:
            raise FileNotFoundError("No checkpoint provided and no latest.pt found under experiment.output_dir.")
        self.checkpoint_path = str(Path(ckpt).expanduser().resolve())

        check_policy_files(self.config, checkpoint=self.checkpoint_path)
        self.policy = build_policy_from_config(self.config, self.device)
        self.text_embedder = build_text_embedder(self.config, self.device)
        self.checkpoint = load_checkpoint(self.checkpoint_path, model=self.policy, map_location=self.device)
        checkpoint_config = self.checkpoint.get("config", {}) or {}
        checkpoint_data_config = checkpoint_config.get("data", {}) or {}
        # Checkpoints created before this field was introduced used raw Bridge
        # actions. Shapes cannot distinguish those weights from state-delta ones.
        checkpoint_state_delta_action = bool(
            checkpoint_data_config.get("state_delta_action", False)
        )
        if checkpoint_state_delta_action != self.state_delta_action:
            expected = "state_delta" if self.state_delta_action else "raw_action"
            actual = "state_delta" if checkpoint_state_delta_action else "raw_action"
            raise ValueError(
                f"Checkpoint action semantics mismatch: config expects {expected}, "
                f"checkpoint is {actual}: {self.checkpoint_path}"
            )
        checkpoint_epoch = self.checkpoint.get("epoch")
        self.checkpoint_epoch = None if checkpoint_epoch is None else int(checkpoint_epoch)
        self.checkpoint_global_step = int(self.checkpoint.get("global_step", 0))
        self.policy.eval()
        self.normalization_stats = load_policy_normalization_stats(
            data_dir=get_nested(self.config, "data.path"),
            normalization_stats_path=get_nested(self.config, "data.normalization_stats_path"),
        )
        if self.state_delta_action and "steps/action/state_delta" not in self.normalization_stats.get("normalization", {}):
            raise ValueError(
                "Config has data.state_delta_action=true, but normalization stats do not contain "
                "'normalization.steps/action/state_delta'. Use the matching state-delta stats file."
            )

    def _normalize_state_action(self, states: np.ndarray, actions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.float32)
        return normalize_policy_state_action(
            states_t,
            actions_t,
            self.normalization_stats,
            eps=self.normalization_eps,
        )

    def build_batch(
        self,
        *,
        current_rgb: np.ndarray,
        past_frames: np.ndarray,
        future_frames: np.ndarray,
        past_states: np.ndarray,
        past_actions: np.ndarray,
        past_image_padding_mask: np.ndarray,
        past_state_padding_mask: np.ndarray,
        past_action_padding_mask: np.ndarray,
        instruction: str,
    ) -> Dict[str, Any]:
        if past_frames.shape[0] != self.history_image_frames:
            raise ValueError(
                f"past_frames must contain {self.history_image_frames} frames, got {past_frames.shape}."
            )
        if future_frames.shape[0] != self.future_steps:
            raise ValueError(
                f"future_frames must contain {self.future_steps} frames, got {future_frames.shape}."
            )
        if past_states.shape != (self.history_state_frames, 7):
            raise ValueError(
                f"past_states must be [{self.history_state_frames},7], got {past_states.shape}."
            )
        if past_actions.shape != (self.history_frames, 7):
            raise ValueError(
                f"past_actions must be [{self.history_frames},7], got {past_actions.shape}."
            )
        if current_rgb.ndim != 3 or current_rgb.shape[-1] != 3:
            raise ValueError(f"current_rgb must be [H,W,3], got {current_rgb.shape}.")

        hist_states, hist_actions = self._normalize_state_action(past_states, past_actions)
        image_mask_t = torch.as_tensor(past_image_padding_mask, dtype=torch.bool)
        state_mask_t = torch.as_tensor(past_state_padding_mask, dtype=torch.bool)
        action_mask_t = torch.as_tensor(past_action_padding_mask, dtype=torch.bool)
        hist_actions = hist_actions.clone()
        hist_actions[action_mask_t] = 0.0

        target_h, target_w = int(current_rgb.shape[0]), int(current_rgb.shape[1])
        future_frames = resize_frames_to_hw(future_frames, height=target_h, width=target_w)
        future_images = (
            np.concatenate([current_rgb[None, ...], future_frames], axis=0)
            if self.future_images_include_current
            else future_frames
        )
        future_frame_count = int(future_images.shape[0])
        batch: Dict[str, Any] = {
            "hist_images": torch.from_numpy(np.ascontiguousarray(past_frames))[None, ...],
            "hist_states": hist_states[None, ...],
            "hist_actions": hist_actions[None, ...],
            "future_images": torch.from_numpy(np.ascontiguousarray(future_images))[None, ...],
            "future_states": torch.zeros((1, future_frame_count, 7), dtype=torch.float32),
            "future_actions": torch.zeros((1, self.action_horizon, 7), dtype=torch.float32),
            "hist_image_padding_mask": image_mask_t[None, ...],
            "hist_state_padding_mask": state_mask_t[None, ...],
            "hist_action_padding_mask": action_mask_t[None, ...],
            "future_frame_padding_mask": torch.zeros((1, future_frame_count), dtype=torch.bool),
            "future_state_padding_mask": torch.zeros((1, future_frame_count), dtype=torch.bool),
            "future_action_padding_mask": torch.zeros((1, self.action_horizon), dtype=torch.bool),
            "future_video_padding_mask": torch.zeros((1, self.future_steps), dtype=torch.bool),
            "instruction": [instruction],
        }
        batch = move_policy_batch_to_device(batch, self.device)
        return attach_text_features(batch, self.text_embedder, num_slots=self.num_slots, device=self.device)

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        current_rgb: np.ndarray,
        past_frames: np.ndarray,
        future_frames: np.ndarray,
        past_states: np.ndarray,
        past_actions: np.ndarray,
        past_image_padding_mask: np.ndarray,
        past_state_padding_mask: np.ndarray,
        past_action_padding_mask: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """Return one de-normalized policy action.

        When ``data.state_delta_action`` is true, dimensions 0:6 are the predicted
        ``state[t + 1, :6] - state[t, :6]`` delta used by the Bridge/WidowX
        target-delta pose controller.
        """
        batch = self.build_batch(
            current_rgb=current_rgb,
            past_frames=past_frames,
            future_frames=future_frames,
            past_states=past_states,
            past_actions=past_actions,
            past_image_padding_mask=past_image_padding_mask,
            past_state_padding_mask=past_state_padding_mask,
            past_action_padding_mask=past_action_padding_mask,
            instruction=instruction,
        )
        with autocast_context(self.device, self.precision):
            sample = self.policy.sample_actions(
                batch,
                num_steps=self.sample_steps,
                discretize_gripper=self.discretize_gripper,
                gripper_index=self.gripper_index,
            )
        pred_norm = sample["action_pred"]
        pred = denormalize_policy_actions(
            pred_norm,
            self.normalization_stats,
            eps=self.normalization_eps,
        )
        return pred[0, 0].detach().float().cpu().numpy().astype(np.float32)
