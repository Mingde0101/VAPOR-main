import unittest

import numpy as np

from evaluation.simpler.reference_preprocess import (
    drop_reference_prefix_frames,
    resample_frames_nearest,
)


class RollingReferencePreprocessTest(unittest.TestCase):
    def test_video2video_drop_then_resample_uses_expected_original_indices(self):
        frames = np.arange(21, dtype=np.uint8).reshape(21, 1, 1, 1)
        frames = np.repeat(frames, 3, axis=-1)

        trimmed, dropped = drop_reference_prefix_frames(frames, num_prefix_frames=4)
        resampled, indices = resample_frames_nearest(trimmed, source_fps=10.0, target_fps=5.0)

        self.assertEqual(dropped, 4)
        np.testing.assert_array_equal(indices + dropped, np.arange(4, 21, 2))
        np.testing.assert_array_equal(resampled[:, 0, 0, 0], np.arange(4, 21, 2))

    def test_initial_reference_can_be_resampled_without_prefix_drop(self):
        frames = np.zeros((21, 2, 2, 3), dtype=np.uint8)

        resampled, indices = resample_frames_nearest(frames, source_fps=10.0, target_fps=5.0)

        self.assertEqual(resampled.shape, (11, 2, 2, 3))
        np.testing.assert_array_equal(indices, np.arange(0, 21, 2))


if __name__ == "__main__":
    unittest.main()
