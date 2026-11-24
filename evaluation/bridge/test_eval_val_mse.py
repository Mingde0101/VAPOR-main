import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from evaluation.bridge.eval_val_mse import aggregate, action_metrics, apply_runtime_options, prepare_output_directory, ROOT


class AggregationTests(unittest.TestCase):
    def test_overwrite_archives_all_old_chunks_and_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            (output / 'chunks').mkdir(parents=True)
            (output / 'manifest.json').write_text('{}')
            (output / 'step_mse.jsonl').write_text('old-step\n')
            (output / 'chunks/predictions_999999.npz').write_bytes(b'old chunk')
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output)
            backup = prepare_output_directory(output, overwrite=True)
            self.assertFalse(output.exists())
            self.assertEqual((backup / 'step_mse.jsonl').read_text(), 'old-step\n')
            self.assertEqual((backup / 'chunks/predictions_999999.npz').read_bytes(), b'old chunk')
            output.mkdir()
            self.assertIsNone(prepare_output_directory(output, overwrite=True))

    def test_overwrite_rejects_project_root_and_unrelated_directory(self):
        with self.assertRaises(ValueError):
            prepare_output_directory(ROOT, overwrite=True)
        with tempfile.TemporaryDirectory() as temporary:
            unrelated = Path(temporary)
            (unrelated / 'notes.txt').write_text('retain me')
            with self.assertRaises(ValueError):
                prepare_output_directory(unrelated, overwrite=True)
            self.assertEqual((unrelated / 'notes.txt').read_text(), 'retain me')

    def test_throughput_overrides_before_model_construction(self):
        args = SimpleNamespace(batch_size=16, num_workers=8, window_batch_size=8)
        config = apply_runtime_options({'data': {'split': 'val'}, 'vjepa2': {'window_batch_size': 1}}, args)
        self.assertEqual(config['data']['batch_size'], 16)
        self.assertEqual(config['data']['num_workers'], 8)
        self.assertEqual(config['data']['split'], 'val')
        self.assertEqual(config['vjepa2']['window_batch_size'], 8)
        args.window_batch_size = None
        self.assertEqual(apply_runtime_options(config, args)['vjepa2']['window_batch_size'], 8)

    def test_per_action_mse_in_original_units(self):
        result = action_metrics([1, 2, 3, 4, 5, 6, 1], np.zeros(7))
        self.assertEqual(result['squared_error_7d'], [1, 4, 9, 16, 25, 36, 1])
        self.assertAlmostEqual(result['mse_all_7d'], 92 / 7)
        self.assertAlmostEqual(result['mse_first_6d'], 91 / 6)
        self.assertAlmostEqual(result['mse_translation_3d'], 14 / 3)
        self.assertAlmostEqual(result['mse_rotation_3d'], 77 / 3)
        self.assertEqual(result['mse_gripper'], 1)

    def test_padding_and_weighting_across_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'chunks').mkdir()
            # Trajectory 0 has one valid row with error=1; trajectory 1 has
            # three valid rows with error=3. Huge padded errors must not count.
            for index, (errors, traj, valid) in enumerate([
                ([1, 1000, 3], [0, 0, 1], [True, False, True]),
                ([3, 3, 1000], [1, 1, 1], [True, True, False]),
            ]):
                np.savez(root / 'chunks' / f'predictions_{index:06d}.npz',
                         pred_action=np.repeat(np.asarray(errors)[:, None], 7, axis=1),
                         target_action=np.zeros((3, 7)), valid_action=np.asarray(valid),
                         shard_index=np.zeros(3, dtype=int), trajectory_index=np.asarray(traj))
            result = aggregate(root)
            self.assertEqual(result['valid_timesteps'], 4)
            self.assertEqual(result['num_trajectories'], 2)
            self.assertEqual(result['timestep_weighted']['mse_all_7d'], 7)
            self.assertEqual(result['trajectory_equal_weighted']['mse_all_7d'], 5)

    def test_no_valid_rows_is_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                aggregate(Path(temporary))

    def test_four_metrics_and_gripper_passthrough(self):
        stats = {'source': {'state_delta_action': True, 'split': 'train'},
                 'normalization': {'steps/action/state_delta': {'mean': [10] * 6, 'std': [2] * 6}}}
        for save_normalized in (False, True):
            with self.subTest(save_normalized=save_normalized), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'chunks').mkdir()
                arrays = {'pred_action': np.array([[2] * 6 + [1]]), 'target_action': np.zeros((1, 7)),
                          'valid_action': np.ones(1, bool), 'shard_index': np.zeros(1, int),
                          'trajectory_index': np.zeros(1, int)}
                if save_normalized:
                    arrays.update(pred_action_normalized=np.array([[-4] * 6 + [1]]),
                                  target_action_normalized=np.array([[-5] * 6 + [0]]))
                np.savez(root / 'chunks/predictions_000000.npz', **arrays)
                result = aggregate(root, stats=stats)['timestep_weighted']
                self.assertEqual(result['normalized']['mse_first_6d'], 1)
                self.assertEqual(result['normalized']['mse_all_7d'], 1)
                self.assertEqual(result['denormalized']['mse_first_6d'], 4)
                self.assertAlmostEqual(result['denormalized']['mse_all_7d'], 25 / 7)
                self.assertEqual(result['normalized']['mse_gripper'], result['denormalized']['mse_gripper'])


if __name__ == '__main__':
    unittest.main()
