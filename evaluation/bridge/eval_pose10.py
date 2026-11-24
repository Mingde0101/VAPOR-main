"""CPU-only v3 state-delta7 -> mimic relative-pose10 MSE from saved predictions."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from lang_slot.paths import resources
from evaluation.bridge.pose10 import relative_actions, states_to_eef_poses, ValidationEpisodes


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def metrics_from_sse(sse, count):
    values = np.asarray(sse) / count
    return {'count': int(count), 'mse_10d': values.tolist(), 'mse_pose_9d': float(values[:9].mean()),
            'mse_all_10d': float(values.mean()), 'mse_translation_3d': float(values[:3].mean()),
            'mse_rotation_6d': float(values[3:9].mean()), 'mse_gripper': float(values[9])}


class Native10MetricSpace:
    """Mimic stats as metric scales; preserve unbounded model predictions."""
    def __init__(self, stats_path):
        stats = json.loads(stats_path.read_text())
        self.shift = np.asarray(stats["shift"])
        self.scale = np.asarray(stats["scale"])
        self.minimum = np.array([x if x is not None else -np.inf for x in stats["minimum"]])
        self.maximum = np.array([x if x is not None else np.inf for x in stats["maximum"]])

    def units(self, prediction, target):
        pred = (prediction - self.shift) * self.scale
        truth = (target - self.shift) * self.scale
        return {'denormalized': (prediction, target), 'normalized': (pred, truth),
                'mimic_training_target_normalized': (pred, np.clip(truth, self.minimum, self.maximum))}

class Aggregator:
    def __init__(self, space):
        self.space = space
        self.groups = defaultdict(lambda: {'count': 0, **{k: np.zeros(10) for k in
                                         ['denormalized', 'normalized', 'mimic_training_target_normalized']}})

    def add(self, prediction, target, shard, record):
        units = self.space.units(prediction, target)
        for key in set(zip(shard.tolist(), record.tolist())):
            mask = (shard == key[0]) & (record == key[1])
            group = self.groups[key]
            group['count'] += int(mask.sum())
            for name, (pred, truth) in units.items():
                error = pred[mask] - truth[mask]
                if not np.isfinite(error).all():
                    raise ValueError('Nonfinite native10 error')
                group[name] += np.square(error).sum(axis=0)

    def summary(self):
        count = sum(g['count'] for g in self.groups.values())
        if not count:
            return {'valid_timesteps': 0}
        weighted, equal, trajectories = {}, {}, []
        names = ['denormalized', 'normalized', 'mimic_training_target_normalized']
        for name in names:
            weighted[name] = metrics_from_sse(sum(g[name] for g in self.groups.values()), count)
            means = [g[name] / g['count'] for g in self.groups.values()]
            equal[name] = metrics_from_sse(np.mean(means, axis=0), 1)
            equal[name]['count'] = count
        for key, group in sorted(self.groups.items()):
            trajectories.append({'shard_index': key[0], 'trajectory_index': key[1], 'valid_timesteps': group['count'],
                                 **{name: metrics_from_sse(group[name], group['count']) for name in names}})
        return {'valid_timesteps': count, 'num_trajectories': len(self.groups),
                'timestep_weighted': weighted, 'trajectory_equal_weighted': equal, 'trajectories': trajectories}


def load_mimic_predictions(directory):
    saved = {}
    if directory is None:
        return saved
    for path in sorted((directory / 'chunks').glob('predictions_*.npz')):
        with np.load(path, allow_pickle=False) as arrays:
            if not {'prediction_native10', 'target_native10', 'valid_mask'} <= set(arrays.files):
                raise ValueError(f'Expected native10 chunk format: {path}')
            for i in range(len(arrays['timestep'])):
                if not arrays['valid_mask'][i, 0]:
                    continue
                key = tuple(int(arrays[k][i]) for k in ['shard_index', 'trajectory_index', 'timestep'])
                if key in saved:
                    raise ValueError('Duplicate Mimic sample identity')
                saved[key] = (arrays['prediction_native10'][i, 0].astype(np.float64),
                              arrays['target_native10'][i, 0].astype(np.float64))
    if not saved:
        raise ValueError('No committed native10 predictions found in the Mimic run')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=resources()['BRIDGE_DATASET_DIR'])
    parser.add_argument('--episode-index', type=Path, default=resources()['BRIDGE_INDEX_DIR'] / 'bridge_1_0_0_val_semantic_valid_v1.npz')
    parser.add_argument('--mimic-stats', type=Path, default=ROOT / 'configs/mimic_first_action_stats.json')
    parser.add_argument('--mimic-run-dir', type=Path, help='Optional partial/full Mimic run; compare exact common anchors')
    parser.add_argument('--overwrite', action='store_true', help='Back up the previous recognized native10 output')
    args = parser.parse_args()
    output = args.output_dir.expanduser().absolute()
    if output.is_symlink() or output.resolve() == args.v3_run_dir.resolve():
        parser.error('Output must be a separate directory')
    if output.exists() and any(output.iterdir()):
        old_manifest = output / 'manifest.json'
        recognized = (output / 'mse_summary_10d.json').is_file()
        if old_manifest.is_file():
            previous = json.loads(old_manifest.read_text())
            recognized = recognized or (previous.get('model_forward_executed') is False
                                         and previous.get('source_run') == str(args.v3_run_dir.resolve()))
        if not args.overwrite or not recognized:
            parser.error('Existing nonempty output: choose another path or --overwrite a recognized run')
        output.rename(output.with_name(output.name + '.backup-' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    (output / 'chunks').mkdir(parents=True, exist_ok=True)
    (output / 'mimic_normalization_stats.json').write_bytes(args.mimic_stats.read_bytes())
    dataset = ValidationEpisodes(args.data_dir, args.episode_index)
    space = Native10MetricSpace(args.mimic_stats)
    full, common_v3, common_mimic = [Aggregator(space) for _ in range(3)]
    mimic = load_mimic_predictions(args.mimic_run_dir)
    matched = set()
    seen = set()
    max_error = 0.
    sources = []
    manifest = {'status': 'running', 'source_run': str(args.v3_run_dir.resolve()),
                'model_forward_executed': False, 'device': 'cpu'}
    atomic_json(output / 'manifest.json', manifest)
    try:
        files = sorted((args.v3_run_dir / 'chunks').glob('predictions_*.npz'))
        if not files:
            raise ValueError('No saved v3 prediction chunks')
        with (output / 'step_mse_10d.jsonl').open('w') as stream:
            for chunk_number, path in enumerate(files):
                sources.append({'file': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                with np.load(path, allow_pickle=False) as source:
                    valid = source['valid_action'].astype(bool)
                    keys = ['sample_index', 'shard_index', 'trajectory_index', 'timestep']
                    ids = {k: source[k][valid].astype(np.int64) for k in keys}
                    pred7, target7 = [source[k][valid].astype(np.float64) for k in ['pred_action', 'target_action']]
                if not len(pred7):
                    continue
                current = np.zeros_like(pred7)
                target10 = np.zeros((len(pred7), 10), np.float64)
                for shard, record in sorted(set(zip(ids['shard_index'].tolist(), ids['trajectory_index'].tolist()))):
                    selected = (ids['shard_index'] == shard) & (ids['trajectory_index'] == record)
                    times = ids['timestep'][selected]
                    episode = dataset.load_episode(shard, record)
                    if np.any(times < 0) or np.any(times >= len(episode.states) - 1):
                        raise ValueError('Saved prediction has an invalid anchor timestep')
                    raw = np.c_[episode.states[times + 1, :6] - episode.states[times, :6], episode.actions[times, 6]]
                    np.testing.assert_allclose(target7[selected], raw, atol=1e-7, rtol=1e-6)
                    current[selected] = episode.states[times]
                    poses = states_to_eef_poses(episode.states)
                    relative = np.linalg.inv(poses[times]) @ poses[times + 1]
                    target10[selected] = np.c_[relative[:, :3, 3], relative[:, :2, :3].reshape(-1, 6),
                                               episode.actions[times, 6]]
                prediction = relative_actions(pred7, current)
                restored_target = relative_actions(target7, current)
                np.testing.assert_allclose(restored_target, target10, atol=3e-6, rtol=1e-5)
                max_error = max(max_error, float(np.max(np.abs(restored_target - target10))))
                full.add(prediction, target10, ids['shard_index'], ids['trajectory_index'])
                units = space.units(prediction, target10)
                common_indices, native_predictions = [], []
                for i in range(len(prediction)):
                    identity = tuple(int(ids[k][i]) for k in ['shard_index', 'trajectory_index', 'timestep'])
                    if identity in seen:
                        raise ValueError('Duplicate saved v3 anchor')
                    seen.add(identity)
                    row = {k: int(ids[k][i]) for k in keys}
                    row.update(pred_action_10d=prediction[i].tolist(), target_action_10d=target10[i].tolist(),
                               **{name: metrics_from_sse(np.square(pred[i] - truth[i]), 1) for name, (pred, truth) in units.items()})
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    if identity in mimic:
                        native_pred, native_truth = mimic[identity]
                        np.testing.assert_allclose(native_truth, target10[i], atol=3e-6, rtol=1e-5)
                        matched.add(identity)
                        common_indices.append(i)
                        native_predictions.append(native_pred)
                if common_indices:
                    selected = np.asarray(common_indices)
                    common_v3.add(prediction[selected], target10[selected], ids['shard_index'][selected],
                                   ids['trajectory_index'][selected])
                    common_mimic.add(np.stack(native_predictions), target10[selected], ids['shard_index'][selected],
                                      ids['trajectory_index'][selected])
                np.savez_compressed(output / 'chunks' / f'predictions_{chunk_number:06d}.npz', **ids,
                                    pred_action_10d=prediction, target_action_10d=target10,
                                    pred_action_10d_normalized=units['normalized'][0],
                                    target_action_10d_normalized=units['normalized'][1],
                                    target_action_10d_training_normalized=units['mimic_training_target_normalized'][1],
                                    pred_action_7d=pred7, target_action_7d=target7, current_state_7d=current)
                stream.flush()
                manifest.update(converted_samples=len(seen), completed_chunks=chunk_number + 1)
                atomic_json(output / 'manifest.json', manifest)
                print(json.dumps(manifest), flush=True)
        if len(matched) != len(mimic):
            raise ValueError(f'V3 does not cover all Mimic anchors: matched {len(matched)}/{len(mimic)}')
        summary = {'schema_version': '1.0', 'dataset': 'Bridge TFDS 1.0.0', 'split': 'val',
                   'source_run': str(args.v3_run_dir.resolve()),
                   'metric_space': '[anchor-relative translation3, relative rotation matrix first two ROWS6, gripper command1]',
                   'prediction_protocol': 'first predicted action at each nonterminal anchor',
                   'model_forward_executed': False, 'source_chunks': sources,
                   'label_roundtrip': {'passed': True, 'max_abs_error': max_error},
                   'metric_normalization': {'stats_path': str(args.mimic_stats.resolve()),
                     'stats_sha256': hashlib.sha256(args.mimic_stats.read_bytes()).hexdigest(),
                     'checkpoint_action_stats_shape': [15, 10], 'action_horizon_index': 0,
                     'normalized': 'Mimic shift/scale, both prediction and raw target UNCLIPPED; rotation identity',
                     'mimic_training_target_normalized': 'same unbounded prediction; target uses native training clamps',
                     'prediction_clipped': False, 'recomputed_stats_on_val': False},
                   **full.summary(),
                   'coverage': {'expected_valid_timesteps': dataset.valid_timesteps, 'expected_trajectories': dataset.num_episodes,
                                'full_index_val_complete': len(seen) == dataset.valid_timesteps
                                  and len(full.groups) == dataset.num_episodes},
                   'limitations': ['A metric in mimic action representation, not a reproduction of its training flow loss',
                                   'Full15-horizon MSE is not comparable with first-action MSE',
                                   'TFDS 5Hz/controller convention assumptions inherited from the Bridge adapter']}
        if mimic:
            summary['same_anchor_comparison'] = {'mimic_run_dir': str(args.mimic_run_dir.resolve()),
                                                 'matched_samples': len(matched), 'targets_match': True,
                                                 'v3': common_v3.summary(), 'mimic': common_mimic.summary()}
        atomic_json(output / 'mse_summary_10d.json', summary)
        with (output / 'trajectory_mse_10d.jsonl').open('w') as stream:
            for row in summary['trajectories']:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        manifest.update(status='complete', valid_timesteps=len(seen))
        atomic_json(output / 'manifest.json', manifest)
        print(json.dumps({'valid_timesteps': len(seen), 'timestep_weighted': summary['timestep_weighted'],
                          'label_roundtrip': summary['label_roundtrip']}), flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        atomic_json(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
