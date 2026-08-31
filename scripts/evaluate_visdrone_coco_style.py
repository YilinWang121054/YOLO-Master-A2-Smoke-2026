#!/usr/bin/env python3
"""Evaluate VisDrone detections with the A2 COCO-style area protocol and maxDets=500."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
AREA_RANGES = {
    "small": [0.0, 32.0**2],
    "medium": [32.0**2, 96.0**2],
    "large": [96.0**2, 1e10],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Original VisDrone val annotations",
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True, help="Original VisDrone val images"
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Ultralytics predictions.json"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Metric summary JSON"
    )
    parser.add_argument(
        "--ground-truth-json",
        type=Path,
        help="Local generated COCO GT path; defaults next to output and should not be published",
    )
    parser.add_argument(
        "--ignore-regions",
        choices=("crowd-per-class", "drop"),
        default="crowd-per-class",
        help="How to handle VisDrone score=0 regions",
    )
    return parser.parse_args()


def convert_ground_truth(
    annotations_dir: Path, images_dir: Path, output: Path, ignore_regions: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Convert original VisDrone validation annotations into a local COCO-format file."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    valid_count = 0
    ignored_regions = 0
    annotation_id = 1
    image_id_map: dict[str, int] = {}

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under {images_dir}")
    for image_id, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        image_id_map[image_path.stem] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            fields = line.rstrip(",").split(",")
            if len(fields) < 6:
                raise ValueError(f"Malformed row at {annotation_path}:{line_number}")
            x, y, box_width, box_height = map(float, fields[:4])
            score, category_id = int(fields[4]), int(fields[5])
            if box_width <= 0 or box_height <= 0:
                continue
            if score == 0 or not 1 <= category_id <= len(NAMES):
                ignored_regions += 1
                if ignore_regions == "crowd-per-class":
                    for ignored_category in range(1, len(NAMES) + 1):
                        annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": ignored_category,
                                "bbox": [x, y, box_width, box_height],
                                "area": box_width * box_height,
                                "iscrowd": 1,
                                "ignore": 1,
                            }
                        )
                        annotation_id += 1
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "ignore": 0,
                }
            )
            annotation_id += 1
            valid_count += 1

    dataset = {
        "info": {
            "description": "Local VisDrone2019-DET val conversion for A2 evaluation"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name} for index, name in enumerate(NAMES, 1)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    return (
        {
            "images": len(images),
            "valid_gt": valid_count,
            "ignored_regions": ignored_regions,
            "coco_annotations": len(annotations),
        },
        image_id_map,
    )


def _mean_percent(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(valid.mean() * 100.0) if valid.size else None


def evaluate(
    ground_truth: Path, predictions: Path, image_id_map: dict[str, int]
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Run bbox COCO evaluation and extract maxDets=500 metrics in percentage points."""
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(ground_truth)
    prediction_rows = json.loads(predictions.read_text(encoding="utf-8"))
    for row in prediction_rows:
        stem = Path(row.get("file_name", str(row["image_id"]))).stem
        if stem not in image_id_map:
            raise ValueError(
                f"Prediction image does not exist in the validation set: {stem}"
            )
        row["image_id"] = image_id_map[stem]
    coco_dt = coco_gt.loadRes(prediction_rows)
    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox", ranges=AREA_RANGES)
    evaluator.params.maxDets = [1, 10, 100, 500]
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]  # IoU, recall, category, area, max_det
    recall = evaluator.eval["recall"]  # IoU, category, area, max_det
    area_index = {name: index for index, name in enumerate(evaluator.params.areaRngLbl)}
    max_det_index = {
        value: index for index, value in enumerate(evaluator.params.maxDets)
    }
    iou_index = {
        round(float(value), 2): index
        for index, value in enumerate(evaluator.params.iouThrs)
    }

    def ap(area: str, iou: float | None = None) -> float | None:
        values = precision[:, :, :, area_index[area], max_det_index[500]]
        if iou is not None:
            values = values[iou_index[iou] : iou_index[iou] + 1]
        return _mean_percent(values)

    def ar(area: str, max_dets: int) -> float | None:
        return _mean_percent(recall[:, :, area_index[area], max_det_index[max_dets]])

    metrics = {
        "AP": ap("all"),
        "AP50": ap("all", 0.5),
        "AP75": ap("all", 0.75),
        "AR1": ar("all", 1),
        "AR10": ar("all", 10),
        "AR100": ar("all", 100),
        "AR500": ar("all", 500),
        "APs": ap("small"),
        "APm": ap("medium"),
        "APl": ap("large"),
        "ARs@500": ar("small", 500),
        "ARm@500": ar("medium", 500),
        "ARl@500": ar("large", 500),
        "AP50s": ap("small", 0.5),
    }
    metadata = {
        "iou_thresholds": [float(value) for value in evaluator.params.iouThrs],
        "max_dets": evaluator.params.maxDets,
        "area_ranges": dict(zip(evaluator.params.areaRngLbl, evaluator.params.areaRng)),
    }
    return metrics, metadata


def main() -> int:
    args = parse_args()
    for path in (args.annotations_dir, args.images_dir, args.predictions):
        if not path.exists():
            raise FileNotFoundError(path)
    ground_truth = args.ground_truth_json or args.output.with_name(
        "visdrone-val-coco-gt.local.json"
    )
    counts, image_id_map = convert_ground_truth(
        args.annotations_dir, args.images_dir, ground_truth, args.ignore_regions
    )
    metrics, evaluator = evaluate(ground_truth, args.predictions, image_id_map)
    prediction_count = len(json.loads(args.predictions.read_text(encoding="utf-8")))
    result = {
        "protocol": {
            "dataset": "VisDrone2019-DET val",
            "metric_unit": "absolute percentage points",
            "area_definition": {
                "status": "COCO-style definition adopted by this A2 project; not an official VisDrone area split",
                "coordinate_space": "original validation-image GT bbox pixels",
                "small": "area < 32^2",
                "medium": "32^2 <= area < 96^2",
                "large": "area >= 96^2",
            },
            "max_dets": 500,
            "ignore_regions": args.ignore_regions,
            "ignore_note": "crowd-per-class approximates VisDrone class-agnostic ignored-region suppression",
        },
        "counts": {**counts, "predictions": prediction_count},
        "metrics": metrics,
        "evaluator": evaluator,
        "inputs": {
            "annotations_dir": str(args.annotations_dir.resolve()),
            "images_dir": str(args.images_dir.resolve()),
            "predictions": str(args.predictions.resolve()),
            "local_ground_truth_json": str(ground_truth.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
