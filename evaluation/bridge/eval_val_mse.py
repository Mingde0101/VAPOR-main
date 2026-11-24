#!/usr/bin/env python3
"""Validation-only v3 Bridge action MSE in normalized and original units."""
import argparse
from collections import defaultdict
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from lang_slot.paths import check_policy_files, resources

STATS = REPO / "configs/bridge_1_0_0_train_state_delta_action_norm_stats.json"
CONFIG = REPO / "configs/bridge_val.yaml"


def preparation():
    path = resources()["BRIDGE_INDEX_DIR"] / "bridge_1_0_0_val_semantic_valid_v1.meta.json"
    index = json.loads(path.read_text())
    return {"selected_trajectories": index["num_trajectories"],
            "valid_first_action_timesteps": index["num_timesteps"] - index["num_trajectories"],
            "dataset_fingerprint": index["dataset_fingerprint"]}


def action_metrics(pred, target):
    squared = np.square(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    if squared.shape != (7,) or not np.isfinite(squared).all():
        raise ValueError('Expected finite 7D actions')
    return {'squared_error_7d': squared.tolist(), 'mse_first_6d': float(squared[:6].mean()),
            'mse_all_7d': float(squared.mean()),
            'mse_translation_3d': float(squared[:3].mean()),
            'mse_rotation_3d': float(squared[3:6].mean()), 'mse_gripper': float(squared[6])}


def mean_metrics(mse):
    mse = np.asarray(mse, dtype=np.float64)
    return {'mse_7d': mse.tolist(), 'mse_first_6d': float(mse[:6].mean()),
            'mse_all_7d': float(mse.mean()), 'mse_translation_3d': float(mse[:3].mean()),
            'mse_rotation_3d': float(mse[3:6].mean()), 'mse_gripper': float(mse[6])}


def normalization_arrays(stats):
    if stats['source']['state_delta_action'] is not True or stats['source']['split'] != 'train':
        raise ValueError('Expected checkpoint-compatible train-split state-delta stats')
    values = stats['normalization']['steps/action/state_delta']
    mean, std = np.zeros(7, np.float64), np.ones(7, np.float64)
    # Match v3's float32 normalization constants and epsilon rule exactly.
    mean[:6] = np.asarray(values['mean'], dtype=np.float32)[:6]
    source_std = np.asarray(values.get('std_safe', values['std']), dtype=np.float32)[:6]
    std[:6] = np.where(np.abs(source_std) < 1e-6, 1.0, source_std)
    return mean, std


def aggregate(output, stats=None):
    stats_path = output / 'normalization_stats.json'
    if stats is None:
        stats = json.loads((stats_path if stats_path.exists() else STATS).read_text())
    mean, std = normalization_arrays(stats)
    groups = defaultdict(lambda: {'sse': np.zeros(7, np.float64), 'normalized_sse': np.zeros(7, np.float64), 'n': 0})
    normalized_sources = defaultdict(int)
    for file in sorted((output / 'chunks').glob('predictions_*.npz')):
        with np.load(file, allow_pickle=False) as arrays:
            valid = arrays['valid_action'].astype(bool)
            pred, target = arrays['pred_action'].astype(np.float64), arrays['target_action'].astype(np.float64)
            if pred.shape != target.shape or pred.shape[1:] != (7,):
                raise ValueError(f'Unexpected action shape in {file}: {pred.shape}, {target.shape}')
            if not np.isfinite(pred[valid]).all() or not np.isfinite(target[valid]).all():
                raise ValueError(f'Nonfinite valid prediction/target in {file}')
            normalized_keys = {'pred_action_normalized', 'target_action_normalized'} & set(arrays.files)
            if len(normalized_keys) == 2:
                pred_normalized = arrays['pred_action_normalized'].astype(np.float64)
                target_normalized = arrays['target_action_normalized'].astype(np.float64)
                source = 'saved_model_outputs'
            elif normalized_keys:
                raise ValueError(f'Missing one normalized prediction/target array in {file}')
            else:
                # Older smoke files saved original units only. No extra forward pass is necessary.
                pred_normalized, target_normalized = (pred - mean) / std, (target - mean) / std
                source = 'reconstructed_from_original_units_with_train_stats'
            if pred_normalized.shape != pred.shape or target_normalized.shape != target.shape:
                raise ValueError(f'Normalized/original shape mismatch in {file}')
            if not np.isfinite(pred_normalized[valid]).all() or not np.isfinite(target_normalized[valid]).all():
                raise ValueError(f'Nonfinite valid normalized action in {file}')
            normalized_sources[source] += int(valid.sum())
            for shard, trajectory in set(zip(arrays['shard_index'][valid].tolist(), arrays['trajectory_index'][valid].tolist())):
                mask = valid & (arrays['shard_index'] == shard) & (arrays['trajectory_index'] == trajectory)
                g = groups[(int(shard), int(trajectory))]
                g['sse'] += np.square(pred[mask] - target[mask]).sum(axis=0)
                g['normalized_sse'] += np.square(pred_normalized[mask] - target_normalized[mask]).sum(axis=0)
                g['n'] += int(mask.sum())
    if not groups:
        raise ValueError('No valid action timesteps to aggregate')
    total_sse = sum(g['sse'] for g in groups.values())
    total_n = sum(g['n'] for g in groups.values())
    weighted = total_sse / total_n
    equal = np.mean([g['sse'] / g['n'] for g in groups.values()], axis=0)
    normalized_weighted = sum(g['normalized_sse'] for g in groups.values()) / total_n
    normalized_equal = np.mean([g['normalized_sse'] / g['n'] for g in groups.values()], axis=0)
    rows = [{'split': 'val', 'shard_index': k[0], 'trajectory_index': k[1],
             'valid_timesteps': g['n'], 'mse_7d': (g['sse'] / g['n']).tolist(),
             'normalized': mean_metrics(g['normalized_sse'] / g['n']),
             'denormalized': mean_metrics(g['sse'] / g['n'])}
            for k, g in sorted(groups.items())]
    return {'schema_version': '2.0', 'dataset_version': '1.0.0', 'split': 'val',
            'action_units': 'normalized and denormalized; legacy flat fields remain denormalized',
            'normalization': {'stats_split': 'train', 'normalized_dims': [0, 1, 2, 3, 4, 5],
                              'gripper_passthrough': True, 'normalized_metric_sources': dict(normalized_sources),
                              'stats_sha256': hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest(),
                              'stats_sha256_representation': 'canonical JSON with sorted keys'},
            'action_semantics': 'state[t+1,:6]-state[t,:6], followed by raw Bridge gripper command',
            'prediction_protocol': 'first action of each sampled 4-action chunk at every timestep',
            'reference_source': 'ground-truth Bridge future frames',
            'padding': 'exclude invalid_action rows, including the last frame of each state-delta trajectory',
            'num_trajectories': len(groups), 'valid_timesteps': total_n,
            'timestep_weighted': {**mean_metrics(weighted), 'normalized': mean_metrics(normalized_weighted),
                                  'denormalized': mean_metrics(weighted)},
            'trajectory_equal_weighted': {**mean_metrics(equal), 'normalized': mean_metrics(normalized_equal),
                                          'denormalized': mean_metrics(equal)},
            'trajectories': rows}


def apply_runtime_options(config, args):
    config.setdefault('data', {}).update(batch_size=args.batch_size, num_workers=args.num_workers)
    if args.window_batch_size is not None:
        config.setdefault('vjepa2', {})['window_batch_size'] = args.window_batch_size
    return config


def prepare_output_directory(output, overwrite=False):
    """Start clean at the requested path; retain the previous run in a sibling backup."""
    output = Path(output).expanduser().absolute()
    if output.is_symlink():
        raise ValueError(f'Output directory must not be a symlink: {output}')
    resolved = output.resolve()
    if ROOT.is_relative_to(resolved) or REPO.is_relative_to(resolved):
        raise ValueError(f'Refusing to overwrite a project/data ancestor: {resolved}')
    if not output.exists():
        return None
    if not output.is_dir():
        raise NotADirectoryError(str(output))
    if not any(output.iterdir()):
        return None
    if not overwrite:
        raise FileExistsError(f'Output directory is not empty: {output}. Pass --overwrite to restart at this path.')
    markers = ('manifest.json', 'resolved_config.json', 'dataset_preparation.json', 'step_mse.jsonl')
    if not any((output / name).is_file() for name in markers):
        raise ValueError(f'Existing directory is not a recognized evaluation output: {output}')
    backup = output.with_name(f'{output.name}.backup-{datetime.now():%Y%m%d_%H%M%S_%f}')
    output.rename(backup)
    print(json.dumps({'stage': 'overwrite', 'output_dir': str(output),
                      'previous_output_backup': str(backup)}, ensure_ascii=False), flush=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=resources()['POLICY_CHECKPOINT'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true',
                        help='Restart in an existing output directory, backing up the complete previous directory')
    parser.add_argument('--shard-indices', default='', help='Empty: all val shards')
    parser.add_argument('--max-trajectories', type=int, default=0, help='0: all selected val trajectories')
    parser.add_argument('--max-samples', type=int, default=0, help='Limit dataset timesteps for smoke tests; 0 means no limit')
    parser.add_argument('--step-delay-sec', type=float, default=0, help='Synchronize and idle after each inference batch')
    parser.add_argument('--cuda-memory-fraction', type=float, help='Per-process CUDA allocator limit, e.g. 0.3; not a compute quota')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--window-batch-size', type=int, help='V-JEPA two-frame windows per encoder call; default from config')
    parser.add_argument('--num-workers', type=int, default=0, help='CPU data-loading workers')
    parser.add_argument('--sample-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--aggregate-only', type=Path, help='Recompute metrics from an existing prediction directory')
    args = parser.parse_args()
    if min(args.batch_size, args.sample_steps) < 1 or min(args.max_trajectories, args.max_samples, args.step_delay_sec) < 0:
        parser.error('batch-size and sample-steps must be positive; max-trajectories must be nonnegative')
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        parser.error('cuda-memory-fraction must be in (0,1]')
    if args.num_workers < 0 or (args.window_batch_size is not None and args.window_batch_size < 1):
        parser.error('num-workers must be nonnegative; window-batch-size must be positive')
    if args.num_workers and args.max_trajectories:
        parser.error('max-trajectories requires num-workers=0; use max-trajectories=0 for full val')
    if args.aggregate_only and args.overwrite:
        parser.error('--overwrite is for a new forward run; do not combine with --aggregate-only')
    runtime = {}
    if args.aggregate_only:
        output = args.aggregate_only.expanduser().resolve()
    else:
        output = (args.output_dir or resources()['POLICY_OUTPUT_DIR'] / 'bridge_mse' / f'{args.checkpoint.parent.parent.name}_{datetime.now():%Y%m%d_%H%M%S}').expanduser().absolute()
        sys.path.insert(0, str(REPO))
        os.environ.setdefault('USE_TF', '0')
        os.environ.setdefault('USE_FLAX', '0')
        import torch
        torch.set_num_threads(min(torch.get_num_threads(), 8))
        device = torch.device(args.device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device.type == 'cuda' and args.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
        from lang_slot.data import bridge_rlds
        from lang_slot.training import policy_utils
        original_load = policy_utils.load_checkpoint

        def strict_load(path, *, model, **kwargs):
            checkpoint = original_load(path, model=model, **kwargs)
            state = checkpoint.get('model_trainable', {})
            expected = {k for k, v in model.named_parameters() if v.requires_grad}
            missing = sorted(expected - set(state))
            if missing or checkpoint['skipped_trainable_tensors']:
                raise ValueError(f'Checkpoint is not a complete match: missing={missing}, skipped={checkpoint["skipped_trainable_tensors"]}')
            if not checkpoint.get('config', {}).get('data', {}).get('state_delta_action', False):
                raise ValueError('Expected a state-delta checkpoint')
            return checkpoint

        from evaluation.bridge import run_policy_forward as forward
        config = apply_runtime_options(policy_utils.load_yaml_config(str(CONFIG)), args)
        check_policy_files(config, checkpoint=args.checkpoint, split="val")
        preparation()
        previous_output_backup = prepare_output_directory(output, args.overwrite)
        output = output.resolve()
        stats = bridge_rlds.load_policy_normalization_stats(normalization_stats_path=config['data']['normalization_stats_path'])
        original_build = forward.build_policy_from_config

        def instrumented_build(model_config, model_device):
            (output / 'normalization_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            (output / 'resolved_config.json').write_text(json.dumps(model_config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            (output / 'dataset_preparation.json').write_text(json.dumps(preparation(), indent=2))
            model = original_build(model_config, model_device)
            original_sample = model.sample_actions

            def sample_with_step_metrics(batch, **kwargs):
                result = original_sample(batch, **kwargs)
                mask = batch['future_action_padding_mask']
                predicted = bridge_rlds.denormalize_policy_actions(result['action_pred'], stats).detach().float().cpu().numpy()
                target = bridge_rlds.denormalize_policy_actions(batch['future_actions'], stats).detach().float().cpu().numpy()
                predicted_normalized = result['action_pred'].detach().float().cpu().numpy()
                target_normalized = batch['future_actions'].detach().float().cpu().numpy()
                valid = (~mask).detach().cpu().numpy()
                for b in range(len(predicted)):
                    t = int(batch['timestep'][b])
                    horizon_actions = []
                    for h in range(predicted.shape[1]):
                        detail = {'horizon_index': h, 'action_timestep': t + h, 'valid': bool(valid[b, h])}
                        if valid[b, h]:
                            normalized_metric = action_metrics(predicted_normalized[b, h], target_normalized[b, h])
                            denormalized_metric = action_metrics(predicted[b, h], target[b, h])
                            detail.update(pred_action=predicted[b, h].tolist(), target_action=target[b, h].tolist(),
                                          pred_action_normalized=predicted_normalized[b, h].tolist(),
                                          target_action_normalized=target_normalized[b, h].tolist(),
                                          normalized=normalized_metric, denormalized=denormalized_metric,
                                          **denormalized_metric)
                        horizon_actions.append(detail)
                    row = {'split': 'val', 'shard_index': int(batch['shard_index'][b]),
                           'trajectory_index': int(batch['trajectory_index'][b]), 'timestep': t,
                           'valid_first_action': bool(valid[b, 0]), 'horizon_actions': horizon_actions}
                    if valid[b, 0]:
                        row.update(normalized=horizon_actions[0]['normalized'],
                                   denormalized=horizon_actions[0]['denormalized'],
                                   **horizon_actions[0]['denormalized'])
                    with (output / 'step_mse.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    print(json.dumps({'stage': 'step_mse', **{k: row[k] for k in ('shard_index', 'trajectory_index', 'timestep', 'valid_first_action')},
                                      'normalized': row.get('normalized'), 'denormalized': row.get('denormalized')}, ensure_ascii=False), flush=True)
                if args.step_delay_sec:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    time.sleep(args.step_delay_sec)
                return result

            model.sample_actions = sample_with_step_metrics
            return model

        argv = ['--config', str(CONFIG),
                    '--checkpoint', str(args.checkpoint.expanduser().resolve()),
                    '--split', 'val', '--output-dir', str(output), '--device', args.device,
                    '--precision', 'bf16', '--batch-size', str(args.batch_size), '--num-workers', str(args.num_workers),
                    '--sample-steps', str(args.sample_steps), '--seed', str(args.seed),
                    '--max-trajectories', str(args.max_trajectories), '--max-samples', str(args.max_samples),
                    '--no-plots', '--save-instructions', '--save-normalized']
        if args.shard_indices:
            argv += ['--shard-indices', args.shard_indices]
        forward.main(argv, config=config, build_policy=instrumented_build, restore=strict_load)
        runtime = {'max_samples': args.max_samples, 'max_trajectories': args.max_trajectories,
                   'batch_size': args.batch_size, 'step_delay_sec': args.step_delay_sec,
                   'window_batch_size': config['vjepa2'].get('window_batch_size'), 'num_workers': args.num_workers,
                   'overwrite': args.overwrite,
                   'previous_output_backup': str(previous_output_backup) if previous_output_backup else None,
                   'cuda_memory_fraction': args.cuda_memory_fraction}
        if device.type == 'cuda':
            runtime.update(cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                           cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
        del forward
        gc.collect()
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['split'] != 'val':
        raise ValueError('This entry point is validation-only')
    summary = aggregate(output)
    summary['forward_manifest'] = manifest
    previous_summary_path = output / 'mse_summary.json'
    if not runtime and previous_summary_path.exists():
        runtime = json.loads(previous_summary_path.read_text()).get('runtime_settings', {})
    if runtime:
        summary['runtime_settings'] = runtime
    snapshot = output / 'dataset_preparation.json'
    prep = json.loads(snapshot.read_text()) if snapshot.exists() else preparation()
    summary['coverage'] = {'expected_selected_val_trajectories': prep['selected_trajectories'],
                           'expected_valid_timesteps': prep['valid_first_action_timesteps'],
                           'all_selected_val_complete': summary['num_trajectories'] == prep['selected_trajectories']
                               and summary['valid_timesteps'] == prep['valid_first_action_timesteps']}
    (output / 'mse_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output_dir': str(output), 'num_trajectories': summary['num_trajectories'],
                      'valid_timesteps': summary['valid_timesteps'], 'timestep_weighted': summary['timestep_weighted']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
