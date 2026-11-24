import unittest
from collections import deque
from pathlib import Path
import sys

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from evaluation.simpler.run_simpler_with_rolling_cosmos import (  # noqa: E402
    build_argparser,
    padded_conditioning_frames,
    resolve_temporal_settings,
)


class RollingCosmosTimingTest(unittest.TestCase):
    def parse_and_resolve(self, *extra_args: str):
        args = build_argparser().parse_args(list(extra_args))
        resolve_temporal_settings(args)
        return args

    def test_defaults_preserve_one_second_window_and_two_second_generation(self):
        args = self.parse_and_resolve()

        self.assertEqual(args.rollout_chunk_steps, 5)
        self.assertEqual(args.rollout_window_sec, 1.0)
        self.assertEqual(args.future_horizon, 4)
        self.assertEqual(args.required_reference_frames, 9)
        self.assertAlmostEqual(args.required_cosmos_duration_sec, 2.0)
        self.assertAlmostEqual(args.cosmos_duration_sec, 2.0)
        self.assertTrue(args.cosmos_duration_auto)

    def test_window_seconds_set_steps_and_auto_duration(self):
        args = self.parse_and_resolve("--rollout-window-sec", "0.8")

        self.assertEqual(args.rollout_chunk_steps, 4)
        self.assertEqual(args.required_reference_frames, 8)
        self.assertAlmostEqual(args.cosmos_duration_sec, 1.8)

    def test_non_integral_policy_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer number of steps"):
            self.parse_and_resolve("--rollout-window-sec", "0.3")

    def test_explicit_reference_duration_cannot_be_too_short(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            self.parse_and_resolve(
                "--rollout-window-sec",
                "1.4",
                "--cosmos-duration-sec",
                "2.3",
            )

    def test_conditioning_uses_latest_five_frames_and_keeps_current_last(self):
        frames = deque(
            [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(7)],
            maxlen=5,
        )

        selected = padded_conditioning_frames(frames, length=5)

        self.assertEqual([int(frame[0, 0, 0]) for frame in selected], [2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
