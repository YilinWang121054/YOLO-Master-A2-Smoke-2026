"""Independently reconcile saved per-candidate records, GT counts and summaries."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def statistics(values):
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "mean": math.fsum(values) / n,
        "p50": values[math.ceil(n * 0.5) - 1],
        "p90": values[math.ceil(n * 0.9) - 1],
        "zero_fraction": values.count(0) / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    folder = args.directory
    output = folder / "verification.json"
    if output.exists():
        raise FileExistsError("Preserve the existing verification record")
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in summary["artifacts_sha256"].items():
        assert digest(folder / name) == expected, name
    gt = {}
    gt_groups, pair_groups = defaultdict(list), defaultdict(list)
    for line in (folder / "gt-records.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = row["image"], row["mode"], row["gt"]
        assert key not in gt
        gt[key] = row
        size = (
            "small"
            if row["input_area"] < 1024
            else "medium"
            if row["input_area"] < 9216
            else "large"
        )
        assert size == row["size"]
        for stage, value in row["counts"].items():
            gt_groups[f"{row['mode']}/{size}/{stage}"].append(value)
    counts = defaultdict(lambda: defaultdict(float))
    anchor_owners, seen_pairs = set(), set()
    candidate_sets, inside_sets = defaultdict(set), defaultdict(set)
    with gzip.open(
        folder / "candidate-records.csv.gz", "rt", newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            key = row["image"], row["mode"], int(row["gt"])
            assert key in gt and row["size"] == gt[key]["size"]
            pair = (*key, int(row["anchor"]))
            assert pair not in seen_pairs
            seen_pairs.add(pair)
            candidate_sets[key].add(int(row["anchor"]))
            flags = {
                k: int(row[k])
                for k in (
                    "inside_gt",
                    "inside_fixed",
                    "before_conflict",
                    "after_conflict",
                )
            }
            assert all(value in (0, 1) for value in flags.values())
            if flags["inside_gt"]:
                inside_sets[key].add(int(row["anchor"]))
            assert flags["after_conflict"] <= flags["before_conflict"]
            values = {
                k: float(row[k])
                for k in (
                    "iou",
                    "rank_ciou",
                    "class_score",
                    "alignment",
                    "target_weight",
                )
            }
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values.values())
            counts[key]["expanded"] += 1
            counts[key]["raw"] += flags["inside_gt"]
            counts[key]["before_conflict"] += flags["before_conflict"]
            counts[key]["after_conflict"] += flags["after_conflict"]
            counts[key]["weight_sum"] += values["target_weight"]
            if not flags["after_conflict"]:
                assert values["target_weight"] == 0
                continue
            owner = row["image"], row["mode"], int(row["anchor"])
            assert owner not in anchor_owners
            anchor_owners.add(owner)
            categories = [
                "all_selected",
                "selected_inside_gt" if flags["inside_gt"] else "selected_outside_gt",
            ]
            if not flags["inside_fixed"]:
                categories.append("selected_new_vs_fixed_candidates")
            for category in categories:
                for metric in ("iou", "rank_ciou", "target_weight"):
                    pair_groups[
                        f"{row['mode']}/{row['size']}/{category}/{metric}"
                    ].append(values[metric])
    max_weight_sum_error = 0.0
    raw_candidates_not_retained = []
    for key, row in gt.items():
        # The fixed xyxy->xywh->xyxy roundtrip can exclude an original candidate
        # at an FP32 boundary. Raw coverage comes from TAL, not its intersection
        # with this mode's candidates; never assume an exact set superset.
        original = candidate_sets[(key[0], "tal", key[2])]
        assert inside_sets[key] == original & candidate_sets[key]
        counts[key]["raw"] = len(original)
        lost = original - candidate_sets[key]
        if lost:
            raw_candidates_not_retained.append(
                {
                    "image": key[0],
                    "mode": key[1],
                    "gt": key[2],
                    "anchors": sorted(lost),
                    "recorded_precast_bbox": row["input_bbox"],
                }
            )
        for stage, value in row["counts"].items():
            if stage == "weight_sum":
                error = abs(counts[key][stage] - value)
                max_weight_sum_error = max(max_weight_sum_error, error)
                assert math.isclose(
                    counts[key][stage], value, rel_tol=1e-6, abs_tol=1e-5
                )
            else:
                assert counts[key][stage] == value, (key, stage)
    for label, source in (
        ("gt_statistics", gt_groups),
        ("selected_pair_statistics", pair_groups),
    ):
        assert set(source) == set(summary[label])
        for key, values in source.items():
            for metric, value in statistics(values).items():
                assert math.isclose(
                    value, summary[label][key][metric], rel_tol=1e-12, abs_tol=1e-12
                ), (label, key, metric)
    image_names = {row["image"] for row in manifest["inputs"]}
    assert len(image_names) == summary["images"] == manifest["config"]["sample_count"]
    assert {row["image"] for row in gt.values()} == image_names
    modes = manifest["config"]["modes"]
    for name in image_names:
        identities = [
            {
                (r["gt"], r["class"], tuple(r["input_bbox"]))
                for r in gt.values()
                if r["image"] == name and r["mode"] == mode
            }
            for mode in modes
        ]
        assert identities[0] == identities[1] == identities[2]
    result = {
        "status": "pass",
        "images": len(image_names),
        "mode_gt_records": len(gt),
        "candidate_pairs": len(seen_pairs),
        "selected_pairs": len(anchor_owners),
        "summary_groups_verified": len(gt_groups) + len(pair_groups),
        "max_gt_weight_sum_float32_error": max_weight_sum_error,
        "raw_candidates_not_retained_by_mode": raw_candidates_not_retained,
        "coordinate_note": "gt-records input_bbox preserves precast coordinates; cast to FP32 before reproducing assignment geometry. input_area and masks were computed on actual FP32 inputs.",
        "summary_sha256": digest(folder / "summary.json"),
        "verification_script_sha256": digest(Path(__file__)),
        "scope": "Raw saved candidate-to-GT and summary reconciliation, not rerun model inference or causality",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
