"""Policy action construction helpers shared by datasets and stats scripts."""

from __future__ import annotations

import numpy as np


def make_policy_actions(
    states: np.ndarray,
    raw_actions: np.ndarray,
    *,
    state_delta_action: bool = True,
) -> np.ndarray:
    """Build policy actions from Bridge states and raw action fields.

    With ``state_delta_action=True``, action dimensions 0:6 are relabeled as
    ``state[t + 1, :6] - state[t, :6]``. The gripper command and optional extra
    dimensions stay from the raw Bridge action. The final action is all zeros
    because there is no reached next state for that timestep.
    """
    states = np.asarray(states, dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"states must be [T,D_state], got {states.shape}.")
    if raw_actions.ndim != 2:
        raise ValueError(f"raw_actions must be [T,D_action], got {raw_actions.shape}.")
    if states.shape[0] != raw_actions.shape[0]:
        raise ValueError(f"states/raw_actions length mismatch: {states.shape[0]} vs {raw_actions.shape[0]}.")
    if states.shape[1] < 6:
        raise ValueError(f"states must have at least 6 dims, got {states.shape[1]}.")
    if raw_actions.shape[1] < 7:
        raise ValueError(f"Bridge policy actions must have at least 7 dims, got {raw_actions.shape[1]}.")
    if not state_delta_action:
        return raw_actions.copy()

    actions = np.zeros_like(raw_actions, dtype=np.float32)
    if states.shape[0] > 1:
        actions[:-1, :6] = states[1:, :6] - states[:-1, :6]
        actions[:-1, 6:] = raw_actions[:-1, 6:]
    return actions
