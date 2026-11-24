import sys
import unittest
from pathlib import Path

import numpy as np
from transforms3d.euler import euler2mat


TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from evaluation.simpler.simpler_env_utils import (  # noqa: E402
    BRIDGE_TO_SIMPLER_TCP_ROTATION,
    simpler_tcp_rotation_to_bridge_euler,
    widowx_finger_qpos_to_bridge_state,
)


class SimplerBridgeStateTest(unittest.TestCase):
    def test_tcp_rotation_inverse_matches_bridge_replay_transform(self):
        expected = np.asarray([-0.12, 0.08, 0.31], dtype=np.float64)
        simulated = euler2mat(*expected, axes="sxyz") @ BRIDGE_TO_SIMPLER_TCP_ROTATION

        actual = simpler_tcp_rotation_to_bridge_euler(simulated)

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_gripper_state_maps_nominal_limits_and_clips(self):
        self.assertEqual(widowx_finger_qpos_to_bridge_state(0.015), 0.0)
        self.assertEqual(widowx_finger_qpos_to_bridge_state(0.026), 0.5)
        self.assertEqual(widowx_finger_qpos_to_bridge_state(0.037), 1.0)
        self.assertEqual(widowx_finger_qpos_to_bridge_state(0.014), 0.0)
        self.assertEqual(widowx_finger_qpos_to_bridge_state(0.038), 1.0)


if __name__ == "__main__":
    unittest.main()
