"""Scan original Bridge 1.0.0 TFRecords and build the policy's byte-offset indices."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lang_slot.data.bridge_rlds import _example_message_class, _iter_tfrecord_records
from lang_slot.paths import resources
from bridge_filter.bridge_language_rules import LanguageAnalyzer
from build_bridge_valid_episode_index import build_split, find_shards, load_manifest, file_sha256


def first_text(features, name):
    values = features[name].bytes_list.value
    decoded = (v.decode("utf-8", errors="replace").strip() for v in values)
    return next((text for text in decoded if text), "")


def scan(dataset, split, audit, rules):
    analyzer = LanguageAnalyzer(rules)
    reasons = Counter()
    example_class = _example_message_class()
    with audit.open("w") as stream:
        for shard, path in sorted(find_shards(dataset, split).items()):
            for record, raw in enumerate(_iter_tfrecord_records(path)):
                features = example_class.FromString(raw).features.feature
                instruction = first_text(features, "steps/language_instruction")
                analysis = analyzer.analyze(instruction)
                source = first_text(features, "episode_metadata/file_path")
                source_ok = not any(s in source for s in rules["exclude_path_substrings"])
                flag = features["episode_metadata/has_language"].int64_list.value
                has_text = bool(flag and flag[0] and analysis["normalized"])
                validity = {"valid": "pass", "invalid": "reject", "review": "review"}[analysis["validity"]]
                stages = {"source_policy": "pass" if source_ok else "reject",
                          "semantic_presence": "pass" if has_text else "reject",
                          "semantic_validity": validity if has_text else "not_run"}
                reason = next((k + ":" + v for k, v in stages.items() if v != "pass"), "selected")
                reasons[reason] += 1
                row = dict(split=split, shard_number=shard, record_index=record,
                           episode_id=features["episode_metadata/episode_id"].int64_list.value[0],
                           num_steps=len(features["steps/language_instruction"].bytes_list.value),
                           filter_stages={k: {"status": v} for k, v in stages.items()})
                stream.write(json.dumps(row) + "\n")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    args = parser.parse_args()
    paths = resources()
    dataset, output = paths["BRIDGE_DATASET_DIR"], paths["BRIDGE_INDEX_DIR"]
    if not (dataset / "dataset_info.json").is_file():
        raise FileNotFoundError(f"Missing dataset_info.json at {dataset}. Check BRIDGE_DATASET_DIR.")
    info = json.loads((dataset / "dataset_info.json").read_text())
    if info["version"] != "1.0.0":
        raise ValueError(f"Expected Bridge 1.0.0 at {dataset}")
    rules = yaml.safe_load(Path(__file__).with_name("bridge_filter").joinpath("configs/common_language_rules.yaml").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for split in (["val", "train"] if args.split == "all" else [args.split]):
        with TemporaryDirectory(dir=output) as tmp:
            temp = Path(tmp)
            audit = temp / f"{split}_audit.jsonl"
            counts = scan(dataset, split, audit, rules)
            selected, totals = load_manifest(audit, {split})
            expected = {"train": (53192, 37103, 1334619), "val": (6872, 4963, 177732)}[split]
            selected_frames = sum(steps for shard in selected[split].values() for steps, _ in shard.values())
            print(split, counts, flush=True)
            if (totals[split], counts["selected"], selected_frames) != expected:
                raise ValueError(f"Unexpected Bridge 1.0.0 coverage; expected {expected}")
            build_split(split, selected[split], totals[split], manifest=output / audit.name,
                        manifest_sha256=file_sha256(audit), data_dir=dataset, output_dir=temp)
            meta_path = temp / f"bridge_1_0_0_{split}_semantic_valid_v1.meta.json"
            meta = json.loads(meta_path.read_text())
            meta["first_failing_stage_counts"] = {k: v for k, v in counts.items() if k != "selected"}
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            for file in temp.iterdir():
                file.replace(output / file.name)


if __name__ == "__main__":
    main()
