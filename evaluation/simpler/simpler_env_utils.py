"""Small SIMPLER/ManiSkill3 helpers shared by benchmark scripts."""

from __future__ import annotations

import importlib
from functools import lru_cache
import sysconfig
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lang_slot.paths import data_path, location


@lru_cache(maxsize=1)
def configure_simulation():
    roots = {
        "simpler_env": ("SIMPLER_ROOT", REPO_ROOT / "external/SimplerEnv"),
        "mani_skill": ("MANISKILL_ROOT", REPO_ROOT / "external/ManiSkill"),
        "sapien": ("SAPIEN_ROOT", sysconfig.get_path("platlib")),
    }
    assets = location("MS_ASSET_DIR", data_path("robot_bench/maniskill_assets"))
    for relative in ("data/robots/widowx/wx250s.urdf",
                     "data/tasks/bridge_v2_real2sim_dataset/stages",
                     "data/tasks/bridge_v2_real2sim_dataset/custom",
                     "data/tasks/bridge_v2_real2sim_dataset/real_inpainting"):
        if not (assets / relative).exists():
            raise FileNotFoundError(f"Missing {assets / relative}. Set MS_ASSET_DIR to the ManiSkill asset root.")
    os.environ["MS_ASSET_DIR"] = str(assets)
    packages = {}
    for package, (variable, default) in roots.items():
        root = location(variable, default)
        if not (root / package / "__init__.py").is_file():
            raise FileNotFoundError(f"Missing {root / package}. Set {variable} to the Python package parent.")
        packages[package] = root / package
        sys.path.insert(0, str(root))
    for package, directory in packages.items():
        module = importlib.import_module(package)
        if Path(module.__file__).resolve().parent != directory.resolve():
            raise ImportError(f"{package} loaded from {module.__file__}, expected {directory}")
    importlib.import_module("mani_skill.envs.tasks.digital_twins.bridge_dataset_eval")


# SIMPLER's Bridge ground-truth replay applies this fixed local-tool rotation
# when mapping Bridge TCP Euler state into the simulated ee_gripper_link frame.
BRIDGE_TO_SIMPLER_TCP_ROTATION = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
WIDOWX_FINGER_CLOSED_QPOS = 0.015
WIDOWX_FINGER_OPEN_QPOS = 0.037

def parse_seed_list(seeds: str, *, start_seed: int = 0, num_episodes: int = 1) -> List[int]:
    if seeds:
        return [int(part.strip()) for part in seeds.split(",") if part.strip()]
    return [int(start_seed) + i for i in range(int(num_episodes))]


def resolve_task(task_name: str, env_id: str = "") -> Tuple[str, str, Dict[str, Any]]:
    """Return ``(task_name, env_id, kwargs)`` for a prepackaged SIMPLER task or raw env id."""
    configure_simulation()
    if env_id:
        return task_name or env_id, resolve_latest_env_id(env_id), {}

    import simpler_env

    if task_name in getattr(simpler_env, "ENVIRONMENT_MAP", {}):
        resolved_env_id, kwargs = simpler_env.ENVIRONMENT_MAP[task_name]
        out = dict(kwargs)
        out["prepackaged_config"] = True
        latest_env_id = resolve_latest_env_id(resolved_env_id)
        if latest_env_id != resolved_env_id:
            out.pop("prepackaged_config", None)
        return task_name, latest_env_id, out
    return task_name, resolve_latest_env_id(task_name), {}


def resolve_latest_env_id(env_id: str) -> str:
    """Use the latest registered Gymnasium version when a mapped env id is stale."""
    match = re.match(r"^(?P<name>.+)-v(?P<version>\d+)$", str(env_id))
    if match is None:
        return str(env_id)
    try:
        import gymnasium as gym
        import importlib
        import mani_skill.envs  # noqa: F401

        importlib.import_module("mani_skill.envs.tasks.digital_twins.bridge_dataset_eval")
    except Exception:
        return str(env_id)

    base = match.group("name")
    versions = []
    for spec_id in gym.envs.registration.registry.keys():
        spec_match = re.match(rf"^{re.escape(base)}-v(?P<version>\d+)$", str(spec_id))
        if spec_match is not None:
            versions.append(int(spec_match.group("version")))
    if not versions:
        return str(env_id)
    latest = max(versions)
    return f"{base}-v{latest}"


def make_maniskill3_env(
    *,
    task_name: str,
    env_id: str = "",
    shader: str = "default",
    num_envs: int = 1,
    obs_mode: str = "rgb+segmentation",
):
    """Create a ManiSkill3 env using SIMPLER's registered digital-twin tasks."""
    configure_simulation()
    import gymnasium as gym

    _, resolved_env_id, kwargs = resolve_task(task_name, env_id)
    sensor_configs = {"shader_pack": shader}
    return gym.make(
        resolved_env_id,
        obs_mode=obs_mode,
        num_envs=int(num_envs),
        sensor_configs=sensor_configs,
        **kwargs,
    )


def reset_episode(env, seed: int, *, reset_by_episode_id: bool = True):
    """Reset with the same convention used by SIMPLER's ManiSkill3 evaluator."""
    if reset_by_episode_id:
        import torch

        return env.reset(seed=int(seed), options={"episode_id": torch.tensor([int(seed)])})
    return env.reset(seed=int(seed))


def default_camera_name(env) -> str:
    uid = str(env.unwrapped.robot_uids.uid)
    if "google_robot" in uid:
        return "overhead_camera"
    if "widowx" in uid:
        return "3rd_view_camera"
    raise ValueError(f"Cannot infer camera for robot uid {uid!r}; pass --camera-name.")


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def first_env_array(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def extract_rgb(obs: Dict[str, Any], *, env=None, camera_name: Optional[str] = None) -> np.ndarray:
    if camera_name is None:
        if env is None:
            raise ValueError("camera_name is required when env is not provided.")
        camera_name = default_camera_name(env)
    try:
        image = obs["sensor_data"][camera_name]["rgb"]
    except KeyError as exc:
        available = sorted(obs.get("sensor_data", {}).keys())
        raise KeyError(f"Camera {camera_name!r} not found in obs['sensor_data']; available={available}") from exc
    arr = first_env_array(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RGB image must be [H,W,3] or [1,H,W,3], got {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def get_instruction(env) -> str:
    instruction = env.unwrapped.get_language_instruction()
    if isinstance(instruction, (list, tuple)):
        return str(instruction[0])
    arr = _to_numpy(instruction)
    if arr.shape != () and len(arr) > 0:
        return str(arr.reshape(-1)[0])
    return str(instruction)


def simpler_tcp_rotation_to_bridge_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a simulated WidowX TCP rotation into Bridge XYZ Euler state."""
    from transforms3d.euler import mat2euler

    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation_matrix must be [3,3], got {rotation.shape}.")
    bridge_rotation = rotation @ BRIDGE_TO_SIMPLER_TCP_ROTATION.T
    return np.asarray(mat2euler(bridge_rotation, axes="sxyz"), dtype=np.float32)


def widowx_finger_qpos_to_bridge_state(finger_qpos: float) -> float:
    """Map continuous simulated finger position to the Bridge gripper-state scale."""
    normalized = (float(finger_qpos) - WIDOWX_FINGER_CLOSED_QPOS) / (
        WIDOWX_FINGER_OPEN_QPOS - WIDOWX_FINGER_CLOSED_QPOS
    )
    return float(np.clip(normalized, 0.0, 1.0))


def extract_bridge_state(env) -> np.ndarray:
    """Extract Bridge-aligned 7D proprio state from a ManiSkill3 env.

    Bridge policy training uses [tcp_xyz, tcp_euler_xyz, gripper]. SIMPLER's
    Bridge replay maps the logged TCP frame into ``ee_gripper_link`` with a
    fixed tool rotation, so state extraction applies the inverse transform.
    """
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "tcp"):
        tcp_pose = unwrapped.tcp.pose
    elif hasattr(unwrapped.agent, "tcp_pose"):
        tcp_pose = unwrapped.agent.tcp_pose
    elif hasattr(unwrapped.agent, "tcp") and hasattr(unwrapped.agent.tcp, "pose"):
        tcp_pose = unwrapped.agent.tcp.pose
    else:
        links = getattr(unwrapped.agent.robot, "links_map", {})
        tcp_pose = None
        for link_name in ("ee_gripper_link", "ee_arm_link", "gripper_link"):
            link = links.get(link_name)
            if link is not None and hasattr(link, "pose"):
                tcp_pose = link.pose
                break
        if tcp_pose is None:
            raise AttributeError("Could not find TCP/end-effector pose on env.unwrapped.agent.")
    pose = unwrapped.agent.robot.pose.inv() * tcp_pose
    p = first_env_array(pose.p).astype(np.float32).reshape(-1)[:3]
    matrix = first_env_array(pose.to_transformation_matrix()).astype(np.float64)
    if matrix.ndim == 3:
        matrix = matrix[0]
    euler = simpler_tcp_rotation_to_bridge_euler(matrix[:3, :3])

    gripper = 0.0
    try:
        qpos = unwrapped.agent.robot.get_qpos()
        qpos_arr = first_env_array(qpos).astype(np.float32).reshape(-1)
        if qpos_arr.size:
            gripper = widowx_finger_qpos_to_bridge_state(float(qpos_arr[-1]))
    except Exception:
        gripper = 0.0
    state = np.concatenate([p, euler, np.asarray([gripper], dtype=np.float32)], axis=0)
    if state.shape != (7,):
        raise ValueError(f"Expected 7D Bridge-style state, got {state.shape}.")
    return state.astype(np.float32)


def torch_action_for_env(action: np.ndarray, env) -> Any:
    import torch

    arr = np.asarray(action, dtype=np.float32)
    if arr.shape == (7,):
        arr = arr[None, :]
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=env.unwrapped.device)
    return tensor


def step_with_rgb_capture(
    env,
    action: Any,
    *,
    camera_name: str,
    capture_fps: float,
):
    """Step once while capturing real intermediate RGB observations."""
    base = env.unwrapped
    control_freq = float(base.control_freq)
    if capture_fps <= 0:
        raise ValueError(f"capture_fps must be positive, got {capture_fps}.")
    captures_per_control = float(capture_fps) / control_freq
    rounded_captures = int(round(captures_per_control))
    if rounded_captures < 1 or abs(captures_per_control - rounded_captures) > 1e-6:
        raise ValueError(
            f"capture_fps={capture_fps} must be an integer multiple of env.control_freq={control_freq}."
        )
    sim_steps_per_capture = float(base._sim_steps_per_control) / float(rounded_captures)
    rounded_sim_steps = int(round(sim_steps_per_capture))
    if rounded_sim_steps < 1 or abs(sim_steps_per_capture - rounded_sim_steps) > 1e-6:
        raise ValueError(
            f"env sim/control ratio={base._sim_steps_per_control} is not divisible by "
            f"captures_per_control={rounded_captures}."
        )

    from mani_skill.utils import common
    import torch

    set_action = False
    action_is_unbatched = False
    action_tensor = None
    if action is None:
        pass
    elif isinstance(action, (np.ndarray, torch.Tensor)):
        action_tensor = common.to_tensor(action, device=base.device)
        if action_tensor.shape == base._orig_single_action_space.shape:
            action_is_unbatched = True
        set_action = True
    elif isinstance(action, dict):
        if "control_mode" in action:
            if action["control_mode"] != base.agent.control_mode:
                base.agent.set_control_mode(action["control_mode"])
                base.agent.controller.reset()
            action_tensor = common.to_tensor(action["action"], device=base.device)
            if action_tensor.shape == base._orig_single_action_space.shape:
                action_is_unbatched = True
        else:
            action_tensor = common.to_tensor(action, device=base.device)
            for sub_action in action_tensor.values():
                if sub_action.shape == base._orig_single_action_space.shape:
                    action_is_unbatched = True
                    break
        set_action = True
    else:
        raise TypeError(type(action))

    if set_action:
        if base.num_envs == 1 and action_is_unbatched:
            action_tensor = common.batch(action_tensor)
        base.agent.set_action(action_tensor)
        if base._sim_device.is_cuda():
            if isinstance(base.agent.controller, dict):
                base.scene.px.gpu_apply_articulation_target_position()
                base.scene.px.gpu_apply_articulation_target_velocity()
            else:
                if base.agent.controller.sets_target_qpos:
                    base.scene.px.gpu_apply_articulation_target_position()
                if base.agent.controller.sets_target_qvel:
                    base.scene.px.gpu_apply_articulation_target_velocity()

    base._elapsed_steps += 1
    base._before_control_step()
    captured_rgbs: List[np.ndarray] = []
    for sim_step in range(int(base._sim_steps_per_control)):
        if base.agent is not None:
            base.agent.before_simulation_step()
        base._before_simulation_step()
        base.scene.step()
        base._after_simulation_step()
        if (sim_step + 1) % rounded_sim_steps == 0:
            if base.gpu_sim_enabled:
                base.scene._gpu_fetch_all()
            mid_obs = {"sensor_data": base._get_obs_sensor_data()}
            captured_rgbs.append(extract_rgb(mid_obs, env=env, camera_name=camera_name))
    base._after_control_step()
    if base.gpu_sim_enabled:
        base.scene._gpu_fetch_all()

    info = base.get_info()
    raw_obs = base.get_obs(info, unflattened=True)
    reward = base.get_reward(obs=raw_obs, action=action_tensor, info=info)
    obs = base._flatten_raw_obs(raw_obs)
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    elif "fail" in info:
        terminated = info["fail"].clone()
    else:
        terminated = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)

    max_episode_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    wrapped = env
    while max_episode_steps is None and wrapped is not None:
        max_episode_steps = getattr(wrapped, "_max_episode_steps", None)
        wrapped = getattr(wrapped, "env", None)
    if max_episode_steps is not None:
        truncated = base.elapsed_steps >= int(max_episode_steps)
    else:
        truncated = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    base._last_obs = obs
    return obs, reward, terminated, truncated, info, captured_rgbs


def euler_xyz_delta_to_axis_angle(rotation_delta: np.ndarray) -> np.ndarray:
    """Convert Bridge/Octo XYZ Euler rotation deltas to axis-angle vectors."""
    import torch
    from mani_skill.utils.geometry import rotation_conversions

    rot = np.asarray(rotation_delta, dtype=np.float32)
    if rot.shape[-1] != 3:
        raise ValueError(f"Rotation delta last dim must be 3, got {rot.shape}.")
    original_shape = rot.shape
    rot_t = torch.as_tensor(rot.reshape(-1, 3), dtype=torch.float32)
    matrix = rotation_conversions.euler_angles_to_matrix(rot_t, "XYZ")
    axis_angle = rotation_conversions.matrix_to_axis_angle(matrix)
    return axis_angle.detach().cpu().numpy().astype(np.float32).reshape(original_shape)


def bridge_action_to_maniskill_action(
    action: np.ndarray,
    *,
    gripper_index: int = 6,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """Convert Bridge-style actions to the SIMPLER/ManiSkill controller.

    Bridge/Octo stores orientation actions as XYZ Euler deltas. SIMPLER's public
    policy interface uses axis-angle vectors for action dimensions 3:6.
    Bridge uses open_gripper in {0, 1}, where 0 means close and 1 means open.
    The WidowX Bridge ManiSkill controller expects normalized gripper position
    in {-1, 1}, where -1 means close and 1 means open.
    """

    arr = np.asarray(action, dtype=np.float32).copy()
    if arr.shape[-1] <= gripper_index:
        raise ValueError(f"Action last dim must include gripper index {gripper_index}, got {arr.shape}.")
    if arr.shape[-1] < 6:
        raise ValueError(f"Action last dim must include rotation dimensions 3:6, got {arr.shape}.")
    arr[..., 3:6] = euler_xyz_delta_to_axis_angle(arr[..., 3:6])
    arr[..., gripper_index] = np.where(
        arr[..., gripper_index] >= float(gripper_threshold),
        1.0,
        -1.0,
    ).astype(np.float32)
    return arr


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
