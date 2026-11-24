"""Locations shared by training, data preparation and evaluation."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def work_dir():
    return ROOT


def work_path(*parts):
    return ROOT.joinpath(*parts)


def location(variable, default):
    return Path(os.environ.get(variable) or default).expanduser().resolve()


def data_dir():
    return location("SLOT_ATTENTION_VIDEO_DATA_DIR", ROOT / "data")


def data_path(*parts):
    return data_dir().joinpath(*parts)


def resources():
    data = data_dir()
    outputs = location("POLICY_OUTPUT_DIR", data / "outputs")
    return {
        "REPO": ROOT,
        "DATA": data,
        "BRIDGE_DATASET_DIR": location("BRIDGE_DATASET_DIR", data / "Bridge/bridge_dataset/1.0.0"),
        "BRIDGE_INDEX_DIR": location("BRIDGE_INDEX_DIR", data / "bridge_indices"),
        "POLICY_OUTPUT_DIR": outputs,
        "POLICY_CHECKPOINT": location("POLICY_CHECKPOINT", outputs / "stage2/checkpoints/latest.pt"),
        "VJEPA2_REPO_DIR": location("VJEPA2_REPO_DIR", ROOT / "external/vjepa2"),
        "VJEPA2_CHECKPOINT": location("VJEPA2_CHECKPOINT", data / "pretrained_models/vjepa2_1_vitl_dist_vitG_384/vjepa2_1_vitl_dist_vitG_384.pt"),
        "SIGLIP2_DIR": location("SIGLIP2_DIR", data / "pretrained_models/siglip2-giant-opt-patch16-384"),
    }


def resolve_path_string(value):
    if not isinstance(value, str):
        return value
    if "${" in value:
        for name, path in resources().items():
            value = value.replace("${" + name + "}", str(path))
    return os.path.expanduser(os.path.expandvars(value))


def resolve_config_paths(value):
    if isinstance(value, dict):
        return {key: resolve_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config_paths(item) for item in value]
    return resolve_path_string(value)


def check_policy_files(config, *, checkpoint=None, split=None):
    """Validate the external resources used by the selected policy entry point."""
    vision, data = config["vjepa2"], config["data"]
    files = []
    if vision.get("backend") == "vjepa2_1":
        files.extend([
            (Path(vision["repo_dir"]) / "app/vjepa_2_1/models/vision_transformer.py", "VJEPA2_REPO_DIR"),
            (vision["checkpoint"], "VJEPA2_CHECKPOINT"),
        ])
    text = Path(config["text"]["model_dir"])
    files.append((text / "config.json", "SIGLIP2_DIR"))
    if data.get("normalize_state_action"):
        files.append((data["normalization_stats_path"], "data.normalization_stats_path"))
    if checkpoint:
        files.append((checkpoint, "policy checkpoint"))
    if split:
        files.append((Path(data["path"]) / "dataset_info.json", "BRIDGE_DATASET_DIR"))
        if data.get("valid_trajectories_only"):
            files.append((data["valid_trajectory_index_paths"][split], "BRIDGE_INDEX_DIR (run scripts/prepare_bridge.py)"))
    for path, setting in files:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {path}. Check {setting}.")
    if not any(text.glob("*.safetensors")) and not any(text.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"Missing SigLIP 2 weights in {text}. Check SIGLIP2_DIR.")
    for index in text.glob("*.index.json"):
        shards = json.loads(index.read_text()).get("weight_map", {}).values()
        for name in set(shards):
            if not (text / name).is_file():
                raise FileNotFoundError(f"Missing {text / name}. Check SIGLIP2_DIR.")
