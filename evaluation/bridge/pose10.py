"""Convert one Bridge delta action into the Mimic anchor-relative pose convention."""
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from lang_slot.data.bridge_rlds import (
    _example_message_class, _feature_float_array, _flatten_tfrecord_actions,
    _iter_selected_tfrecord_records, _load_policy_episode_index,
)


class Episode(NamedTuple):
    states: np.ndarray
    actions: np.ndarray


class ValidationEpisodes:
    def __init__(self, dataset, index):
        grouped, shards = _load_policy_episode_index(str(index), split="val")
        self.shards = {}
        for shard, (name, size, _) in shards.items():
            path = dataset / name
            if Path(name).name != name or path.stat().st_size != size:
                raise ValueError(f"Shard does not match validation index: {path}")
            self.shards[shard] = path
        self.records = {(shard, record): (record, offset, length)
                        for shard, rows in grouped.items() for record, offset, length in rows}
        self.num_episodes = len(self.records)
        with np.load(index, allow_pickle=False) as arrays:
            self.valid_timesteps = int((arrays["num_steps"] - 1).sum())

    @lru_cache(maxsize=1)
    def load_episode(self, shard, record):
        entry = self.records[(shard, record)]
        _, payload = next(_iter_selected_tfrecord_records(self.shards[shard], [entry]))
        example = _example_message_class().FromString(payload)
        states = _feature_float_array(example, "steps/observation/state").reshape(-1, 7)
        return Episode(states, _flatten_tfrecord_actions(example, length=len(states)))


def states_to_eef_poses(states):
    rotation_offset = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    poses = np.broadcast_to(np.eye(4), (len(states), 4, 4)).copy()
    poses[:, :3, :3] = Rotation.from_euler("xyz", states[:, 3:6]).as_matrix() @ rotation_offset
    poses[:, :3, 3] = states[:, :3]
    return poses.astype(np.float32)


def relative_actions(delta, current):
    following = np.column_stack((current[:, :6] + delta[:, :6], delta[:, 6]))
    relative = np.linalg.inv(states_to_eef_poses(current)) @ states_to_eef_poses(following)
    return np.column_stack((relative[:, :3, 3], relative[:, :2, :3].reshape(-1, 6), delta[:, 6]))
