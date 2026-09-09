"""Re-evaluate six saved predictions on CPU; preserve all previously published metrics."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
from evaluate_visdrone_coco_style import (
    _filter_prediction_rows_official,
    convert_ground_truth_official_filter,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = ROOT / ".local/area-audit-20260910"
    output = ROOT / "results/area-audit-20260910"
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    annotations = Path("F:/datasets/VisDrone/VisDrone2019-DET-val/annotations")
    images = annotations.parent / "images"
    gt = work / "gt.json"
    counts, mapping = convert_ground_truth_official_filter(annotations, images, gt)
    summaries = {}
    for seed in (20260824, 20260825):
        for mode in ("fixed", "adaptive", "tal"):
            name = f"p1-{mode}-s{seed}"
            previous_path = (
                ROOT / "results" / name / "coco-style-official-filter-metrics.json"
            )
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            predictions = Path(previous["inputs"]["predictions"])
            filtered, changes = _filter_prediction_rows_official(
                json.loads(predictions.read_text()), annotations, images
            )
            pd_path = work / f"{name}.json"
            pd_path.write_text(json.dumps(filtered))
            metrics, metadata = evaluate(gt, pd_path, mapping)
            payload = {
                "run": name,
                "revision": "20260910-matlab-ignore-half-open-area-bins",
                "evaluator_script_sha256": hashlib.sha256(
                    (ROOT / "scripts/evaluate_visdrone_coco_style.py").read_bytes()
                ).hexdigest(),
                "metrics": metrics,
                "evaluator": metadata,
                "counts": {**counts, **changes},
                "previous_metrics_file": str(previous_path.relative_to(ROOT)),
                "input_predictions_sha256": hashlib.sha256(
                    predictions.read_bytes()
                ).hexdigest(),
                "ignore_policy": "official-filter",
                "area_space": "original validation GT pixels",
            }
            (output / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
            summaries[name] = metrics
            print(
                json.dumps(
                    {
                        "run": name,
                        "APs": metrics["APs"],
                        "old_APs": previous["metrics"]["APs"],
                    }
                ),
                flush=True,
            )
    deltas = {
        str(seed): summaries[f"p1-adaptive-s{seed}"]["APs"]
        - summaries[f"p1-fixed-s{seed}"]["APs"]
        for seed in (20260824, 20260825)
    }
    summary = {
        "runs": summaries,
        "adaptive_minus_fixed_APs": deltas,
        "mean_delta": sum(deltas.values()) / 2,
        "positive_seeds": sum(v > 0 for v in deltas.values()),
        "complete_seed_count": 2,
        "official_DET_recomputed_here": False,
        "runtime_parity": "original MATLAB execution pending",
        "primary_three_seed_mean_pending": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}))


if __name__ == "__main__":
    main()
