# Evaluate in SIMPLER

The benchmark uses the SIMPLER **maniskill3** branch, the ManiSkill 3 Bridge digital twins, and SAPIEN. The fixed-reference runner reads a saved reference clip. The rolling runner obtains references from a Cosmos3-Nano service while the policy controls WidowX at 5 Hz.

## Simulator and assets

Install the simulation packages in the policy environment with Vulkan rendering available. The tested versions are ManiSkill 3.0.1 and SAPIEN 3.0.3. The following setup uses the installed ManiSkill package and the SIMPLER checkout.

```bash
python -m pip install -r evaluation/simpler/requirements.txt
mkdir -p external
git clone --branch maniskill3 https://github.com/simpler-env/SimplerEnv.git external/SimplerEnv
git -C external/SimplerEnv checkout 52a5088ca4bfc3a7159828af08874a198d6f95c3
export MANISKILL_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
export MS_ASSET_DIR="${MS_ASSET_DIR:-${SLOT_ATTENTION_VIDEO_DATA_DIR:-$PWD/data}/robot_bench/maniskill_assets}"
python -m mani_skill.utils.download_asset widowx250s -y
python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
```

| Variable | Default | Expected contents |
|---|---|---|
| `SIMPLER_ROOT` | `<repo>/external/SimplerEnv` | `simpler_env/` from the pinned branch |
| `MANISKILL_ROOT` | `<repo>/external/ManiSkill` | `mani_skill/` including the Bridge digital-twin environments |
| `SAPIEN_ROOT` | Active Python's platform site-packages | `sapien/` and compatible compiled runtime |
| `MS_ASSET_DIR` | `<data>/robot_bench/maniskill_assets` | `data/robots/widowx/` and `data/tasks/bridge_v2_real2sim_dataset/` |

For fixed references, supply an NPZ containing RGB `frames` in `[T,H,W,3]` order and a scalar `fps`. The policy uses the checkpoint and frozen encoders configured in the [main README](../../README.md).

```bash
python evaluation/simpler/run_simpler_with_reference.py \
  --task-name widowx_carrot_on_plate --seeds 0 --reference /mnt/references/carrot-seed0.npz
```

## Rolling Cosmos references

Use a separate vLLM-Omni environment for the generation service. The local experiment used Python 3.12, vLLM 0.26.0 and vLLM-Omni 0.26.0. Install the backend following its CUDA-specific instructions, and place a complete local Cosmos3-Nano snapshot at the configured model path. [Cosmos3-Nano model and serving instructions](https://huggingface.co/nvidia/Cosmos3-Nano) · [vLLM-Omni](https://github.com/vllm-project/vllm-omni)

| Variable | Default |
|---|---|
| `COSMOS3_HOME` | `<data>/cosmos3` |
| `COSMOS3_VLLM_ENV` | `<COSMOS3_HOME>/envs/vllm_omni` with executable `bin/vllm` |
| `COSMOS3_MODEL_DIR` | `<COSMOS3_HOME>/models/Cosmos3-Nano` |
| `COSMOS3_ALLOWED_LOCAL_MEDIA_PATH` | `<POLICY_OUTPUT_DIR>/simpler` |
| `COSMOS3_SERVE_HOST`, `COSMOS3_SERVE_PORT` | `127.0.0.1`, `8001` |
| `COSMOS3_SERVED_MODEL` | Final directory name of `COSMOS3_MODEL_DIR` |

```bash
# Terminal 1, with the service environment prepared
bash scripts/start_cosmos_service.sh --check
bash scripts/start_cosmos_service.sh

# Terminal 2, in the policy environment
python evaluation/simpler/run_simpler_with_rolling_cosmos.py \
  --task-name widowx_carrot_on_plate --seeds 0 --rollout-window-sec 1.4 --run-name carrot-1p4
```

The vLLM-Omni service reads the captured media by filename. If the service runs on a second machine, mount the references directory at the same absolute path on both machines and allow it through the service media-root setting. The client verifies the configured served-model ID. A 1.4 s window executes seven control steps. The first image drives image-to-video generation, then the last five captured frames at 10 Hz condition video-to-video updates. Other Bridge tasks are `widowx_spoon_on_towel`, `widowx_stack_cube` and `widowx_put_eggplant_in_basket`. Results, videos and generated references for the named run go under `<POLICY_OUTPUT_DIR>/simpler/runs/<run-name>/`.
