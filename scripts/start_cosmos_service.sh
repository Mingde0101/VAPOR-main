#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SLOT_ATTENTION_VIDEO_DATA_DIR:-${REPO_ROOT}/data}"
COSMOS_HOME="${COSMOS3_HOME:-${DATA_ROOT}/cosmos3}"
VLLM_ENV="${COSMOS3_VLLM_ENV:-${COSMOS_HOME}/envs/vllm_omni}"
MODEL_DIR="${COSMOS3_MODEL_DIR:-${COSMOS_HOME}/models/Cosmos3-Nano}"
HOST="${COSMOS3_SERVE_HOST:-127.0.0.1}"
PORT="${COSMOS3_SERVE_PORT:-8001}"
MEDIA_ROOT="${COSMOS3_ALLOWED_LOCAL_MEDIA_PATH:-${POLICY_OUTPUT_DIR:-${DATA_ROOT}/outputs}/simpler}"

if [[ ! -x "${VLLM_ENV}/bin/vllm" ]]; then
  echo "Missing vLLM environment: ${VLLM_ENV}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing Cosmos3-Nano model: ${MODEL_DIR}" >&2
  exit 1
fi

python3 - "${MODEL_DIR}" <<'CHECK_WEIGHTS'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
if not index.is_file():
    sys.exit(f"Missing Cosmos model index {index}. Check COSMOS3_MODEL_DIR.")
shards = set(json.loads(index.read_text())["weight_map"].values())
missing = sorted(name for name in shards if not (root / name).is_file())
if not shards or missing:
    sys.exit(f"Incomplete Cosmos weights at {root}. Missing shards: {missing}")
CHECK_WEIGHTS
if [[ "${1:-}" == "--check" ]]; then
  echo "Found the vLLM executable and all indexed Cosmos weights in ${MODEL_DIR}."
  exit 0
fi
if (( $# )); then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi
mkdir -p -- "${MEDIA_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

if models=$(curl --noproxy '*' --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" 2>/dev/null); then
  python3 -c 'import json,sys
from pathlib import Path
ids = [entry["id"] for entry in json.load(sys.stdin)["data"]]
if not any(i == sys.argv[1] or Path(i).name == sys.argv[1] for i in ids):
    sys.exit(f"Unexpected Cosmos service model: {ids}")' "${COSMOS3_SERVED_MODEL:-${MODEL_DIR##*/}}" <<<"$models"
  echo "Cosmos service ready at http://${HOST}:${PORT}"
  exit 0
fi

exec "${VLLM_ENV}/bin/vllm" serve "${MODEL_DIR}" \
  --served-model-name "${COSMOS3_SERVED_MODEL:-${MODEL_DIR##*/}}" \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --host "${HOST}" \
  --port "${PORT}" \
  --no-guardrails \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --enable-layerwise-offload \
  --vae-use-tiling \
  --vae-patch-parallel-size 1 \
  --disable-multithread-weight-load \
  --num-weight-load-threads 1 \
  --no-diffusion-compile-dynamic \
  --init-timeout 1800
