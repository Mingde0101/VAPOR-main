# Bridge language-conditioned slot policy

Train a flow-matching action policy on Bridge 1.0.0 using frozen V-JEPA 2.1 and SigLIP 2 encoders. The policy learns semantic slots, fuses video, text and proprioception, and predicts four 7D actions per window.

## Installation and paths

Use Python 3.10 and a CUDA GPU with BF16 support. Install a CUDA-compatible PyTorch and torchvision pair, then install the policy dependencies. The pinned versions describe the environment used to verify this release.

```bash
python -m pip install -r requirements.txt
mkdir -p external
git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
git -C external/vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
```

Run commands from the repository root. Place the original Bridge TFDS files, the V-JEPA 2.1 ViT-L/16 distilled 384 checkpoint and the complete SigLIP 2 Giant 384 model directory at the locations below. The Bridge directory must contain `dataset_info.json` and its train and val TFRecord shards. Download both model files and tokenizer files for SigLIP. [Bridge TFDS 1.0.0](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/) · [V-JEPA checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt) · [SigLIP model](https://huggingface.co/google/siglip2-giant-opt-patch16-384)

| Environment variable | Default |
|---|---|
| `SLOT_ATTENTION_VIDEO_DATA_DIR` | `<repo>/data` |
| `BRIDGE_DATASET_DIR` | `<data>/Bridge/bridge_dataset/1.0.0` |
| `BRIDGE_INDEX_DIR` | `<data>/bridge_indices` |
| `VJEPA2_REPO_DIR` | `<repo>/external/vjepa2` |
| `VJEPA2_CHECKPOINT` | `<data>/pretrained_models/vjepa2_1_vitl_dist_vitG_384/vjepa2_1_vitl_dist_vitG_384.pt` |
| `SIGLIP2_DIR` | `<data>/pretrained_models/siglip2-giant-opt-patch16-384` |
| `POLICY_OUTPUT_DIR` | `<data>/outputs` |
| `POLICY_CHECKPOINT` | `<outputs>/stage2/checkpoints/latest.pt` for inference and evaluation |

Set a shared root with e.g. `export SLOT_ATTENTION_VIDEO_DATA_DIR=/mnt/data`. Each individual override takes precedence. Third-party source, models, TFRecords, generated indices and run outputs live outside Git. The same Bridge settings apply to preparation, training and both evaluation representations.

## Prepare Bridge and train

```bash
python scripts/prepare_bridge.py
python scripts/train_policy_flow_matching.py --config configs/stage1.yaml
python scripts/train_policy_flow_matching.py --config configs/stage2.yaml
```

Preparation streams the original train and val splits. Each trajectory must pass the source, language-presence and language-validity checks. The scanner excludes scripted collection paths containing `/scripted_numpy_256/`, requires a true language flag and nonempty normalized text, and screens for known actions, encoding issues and annotation notes. These rules describe instruction quality. Task completion and video–instruction agreement would require separate labels. The scanner writes per-episode decisions and byte-offset indices under `BRIDGE_INDEX_DIR`. [Language rules](scripts/bridge_filter/configs/common_language_rules.yaml) · [Scanner](scripts/prepare_bridge.py)

| Split | Input episodes | Scripted | Missing language after source filter | Review | Invalid | Retained | Frame windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 53,192 | 8,674 | 5,858 | 1,555 | 2 | 37,103 | 1,334,619 |
| val | 6,872 | 1,025 | 700 | 182 | 2 | 4,963 | 177,732 |

The slot modules start from their model initialization. Both stages train the slots, condition fusion and action decoder end to end. The experiment used AdamW, BF16, batch size 32, gradient accumulation 1, weight decay 0.01, gradient norm clipping at 1 and 16 visual windows per encoder call.

| Stage | Epochs | Base learning rate | Cumulative updates | Cumulative windows |
|---|---|---:|---:|---:|
| 1 | 1 | 0.00005, linear warmup over 1,000 updates | 41,707 | 1,334,619 |
| 2 | 2 and 3 | 0.000025 | 125,121 | 4,003,857 |

Stage 2 reads the stage 1 checkpoint and restores the optimizer, random state and global step. Both later epochs use the same learning rate, so they share one continuation configuration. The warmup uses the cumulative step. The epoch limit controls termination for both configs. [Stage 1](configs/stage1.yaml) · [Stage 2](configs/stage2.yaml)

At anchor `t`, the model uses four history frames and a future reference clip from `t` to `t+4`. The first six action labels are adjacent differences in the recorded XYZ position and XYZ Euler state. The seventh comes from the recorded gripper command. All seven state components and the six motion deltas use their respective train z-scores. The gripper action keeps its original scale. End padding is masked. The included normalization snapshot was fitted on the **original train split** of 53,192 trajectories. To recompute it, run `python scripts/compute_bridge_policy_norm_stats.py --reader tfrecord`. [Normalization snapshot](configs/bridge_1_0_0_train_state_delta_action_norm_stats.json)

## Run inference and evaluate Bridge

After stage 2, the evaluator finds the final checkpoint automatically. To use an existing epoch 3 checkpoint, set `POLICY_CHECKPOINT` to its file. The historical checkpoint SHA-256 is `98c355f04c67680550c671a3d834b553f88f7580bf10f8bc15755d52bdd78594`. Training produces the policy checkpoint. The repository contains code, configurations and metric summaries.

```bash
# One batch of four windows, including saved first-action predictions and labels
python evaluation/bridge/eval_val_mse.py --shard-indices 0 --max-samples 4 --output-dir runs/bridge-smoke

# Complete numerical experiment
python evaluation/bridge/eval_val_mse.py --output-dir runs/bridge-val
python evaluation/bridge/eval_pose10.py --v3-run-dir runs/bridge-val --output-dir runs/bridge-val-pose10
```

The offline reference uses recorded future RGB frames. The experiment used seed 0, 20 flow steps, batch size 4, V-JEPA window batch size 2, and zero data-loader workers. It scores the first action at each nonterminal anchor, yielding 172,769 valid actions across 4,963 trajectories. `chunks/` holds predictions and targets. `step_mse.jsonl` holds per-anchor errors and `mse_summary.json` includes per-trajectory results, per-dimension MSE and both time-weighted and trajectory-equal means. The optional CPU conversion scores the same predictions as relative translation (3), relative rotation (6) and gripper (1) using the first-action Mimic statistics.

| Historical epoch 3, weighted by valid timestep | Translation (3) | Rotation (3) | Gripper | Overall (7) |
|---|---:|---:|---:|---:|
| Original Bridge units | 0.00000963895 | 0.00109260 | 0.01164577 | 0.00213607 |
| Train-normalized units | 0.07066890 | 0.40801761 | 0.01164577 | 0.20681504 |

[Full aggregate reference values for 7D and 10D](evaluation/bridge/reference_results.json) · [Optional SIMPLER and Cosmos evaluation](evaluation/simpler/README.md)
