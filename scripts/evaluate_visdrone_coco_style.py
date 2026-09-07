#!/usr/bin/env python3
"""Evaluate VisDrone detections with the A2 COCO-style area protocol and maxDets=500.

The ``official-filter`` mode applies the same ignored-region filtering rule as
the VisDrone DET toolkit before constructing the COCO view: a valid GT or
detection whose rounded pixel box is at least 50% covered by an ignored-region
mask is removed. ``crowd-per-class`` is retained for backwards-compatible
exploratory results, but is only an approximation and must not be called the
official VisDrone evaluator.
"""

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
        choices=("crowd-per-class", "drop", "official-filter"),
        default="crowd-per-class",
        help=(
            "How to handle VisDrone ignored regions: crowd-per-class is the "
            "legacy approximation, drop removes ignored GT only, and "
            "official-filter applies the VisDrone 50%% pixel-coverage filter."
        ),
    )
    return parser.parse_args()


def _official_filter_rows(
    gt_rows: np.ndarray,
    det_rows: np.ndarray,
    img_height: int,
    img_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``viseval.drop_objects_in_igr`` from the VisDrone toolkit.

    The toolkit uses a rasterized ignore mask and rounded integer boxes. We
    intentionally preserve those details instead of replacing them with a
    continuous IoU approximation, so the COCO supplementary view follows the
    same ignore-region decision for both GT and detections.
    """
    gt_rows = np.asarray(gt_rows)
    det_rows = np.asarray(det_rows)
    if gt_rows.ndim != 2:
        gt_rows = gt_rows.reshape((-1, 8))
    if det_rows.ndim != 2:
        det_rows = det_rows.reshape((-1, 6))

    # The VisDrone format marks an ignored region with category id 0 (column 5).
    gt_ignore_mask = gt_rows[:, 5] == 0 if len(gt_rows) else np.zeros(0, dtype=bool)
    kept_gt = gt_rows[~gt_ignore_mask]
    ignored = gt_rows[gt_ignore_mask, :4].clip(min=1)
    if not len(ignored):
        return kept_gt, det_rows

    igr_map = np.zeros((img_height, img_width), dtype=np.int64)
    for x1, y1, box_width, box_height in ignored:
        x1 = int(x1)
        y1 = int(y1)
        x2 = min(x1 + int(box_width), img_width)
        y2 = min(y1 + int(box_height), img_height)
        if x2 > x1 and y2 > y1:
            igr_map[y1 - 1 : y2, x1 - 1 : x2] = 1
    integral = np.cumsum(np.cumsum(igr_map, axis=0), axis=1)

    def keep_boxes(rows: np.ndarray) -> np.ndarray:
        keep = np.ones(len(rows), dtype=bool)
        for index, row in enumerate(rows):
            pos = np.round(row[:4]).astype(np.int32).clip(min=1)
            x = max(1, min(img_width - 1, int(pos[0])))
            y = max(1, min(img_height - 1, int(pos[1])))
            width = int(pos[2])
            height = int(pos[3])
            if width <= 0 or height <= 0:
                keep[index] = False
                continue
            x_right = max(1, min(img_width, x + width))
            y_bottom = max(1, min(img_height, y + height))
            top_left = integral[y - 1, x - 1]
            top_right = integral[y - 1, x_right - 1]
            bottom_left = integral[y_bottom - 1, x - 1]
            bottom_right = integral[y_bottom - 1, x_right - 1]
            covered = top_left + bottom_right - top_right - bottom_left
            keep[index] = covered / float(width * height) < 0.5
        return keep

    return kept_gt[keep_boxes(kept_gt)], det_rows[keep_boxes(det_rows)]


def _load_raw_annotations(path: Path) -> np.ndarray:
    """Load one original VisDrone annotation file with its 8 columns."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = np.loadtxt(path, delimiter=",", dtype=np.int32, ndmin=2, usecols=range(8))
    if not len(rows):
        return rows.reshape((0, 8))
    return rows


def _filter_prediction_rows_official(
    predictions: list[dict[str, Any]], annotations_dir: Path, images_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter prediction boxes using the official ignored-region rule."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        stem = Path(str(row.get("file_name", row.get("image_id", "")))).stem
        grouped.setdefault(stem, []).append(row)

    filtered: list[dict[str, Any]] = []
    removed = 0
    for image_path in sorted(images_dir.glob("*.jpg")):
        stem = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size
        gt_rows = _load_raw_annotations(annotations_dir / f"{stem}.txt")
        source_rows = grouped.get(stem, [])
        det_rows = np.asarray(
            [
                [
                    float(item["bbox"][0]),
                    float(item["bbox"][1]),
                    float(item["bbox"][2]),
                    float(item["bbox"][3]),
                    float(item["score"]),
                    int(item["category_id"]),
                ]
                for item in source_rows
            ],
            dtype=np.float32,
        )
        if not len(det_rows):
            det_rows = np.empty((0, 6), dtype=np.float32)
        _, kept_det = _official_filter_rows(gt_rows, det_rows, height, width)
        # The helper preserves order, so match kept boxes by numeric tuple while
        # consuming duplicates one at a time.
        keep_mask = np.zeros(len(source_rows), dtype=bool)
        remaining: dict[tuple[float, float, float, float, float, int], int] = {}
        for row in kept_det:
            key = tuple(float(value) for value in row[:5]) + (int(row[5]),)
            remaining[key] = remaining.get(key, 0) + 1
        for index, row in enumerate(det_rows):
            key = tuple(float(value) for value in row[:5]) + (int(row[5]),)
            if remaining.get(key, 0):
                keep_mask[index] = True
                remaining[key] -= 1
        filtered.extend(source_rows[index] for index in np.flatnonzero(keep_mask))
        removed += int((~keep_mask).sum())
    return filtered, {
        "predictions_before_filter": len(predictions),
        "predictions_removed_by_igr": removed,
    }


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


def convert_ground_truth_official_filter(
    annotations_dir: Path, images_dir: Path, output: Path
) -> tuple[dict[str, int], dict[str, int]]:
    """Build a COCO GT file after applying VisDrone's ignore-region filter."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id_map: dict[str, int] = {}
    valid_before = 0
    ignored_regions = 0
    removed_gt_by_igr = 0
    annotation_id = 1

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
        raw = _load_raw_annotations(annotations_dir / f"{image_path.stem}.txt")
        positive = raw[(raw[:, 5] >= 1) & (raw[:, 5] <= len(NAMES))]
        valid_before += len(positive)
        ignored_regions += int((raw[:, 5] == 0).sum())
        filtered, _ = _official_filter_rows(
            raw, np.empty((0, 6), dtype=np.float32), height, width
        )
        filtered = filtered[(filtered[:, 5] >= 1) & (filtered[:, 5] <= len(NAMES))]
        removed_gt_by_igr += len(positive) - len(filtered)
        for row in filtered:
            x, y, box_width, box_height = (float(value) for value in row[:4])
            if box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(row[5]),
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "ignore": 0,
                }
            )
            annotation_id += 1

    dataset = {
        "info": {
            "description": "Local VisDrone2019-DET val conversion with official ignore-region filtering"
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
            "valid_gt": len(annotations),
            "valid_gt_before_igr_filter": valid_before,
            "ignored_regions": ignored_regions,
            "valid_gt_removed_by_igr": removed_gt_by_igr,
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
    filter_counts: dict[str, int] = {}
    evaluation_predictions = args.predictions
    if args.ignore_regions == "official-filter":
        counts, image_id_map = convert_ground_truth_official_filter(
            args.annotations_dir, args.images_dir, ground_truth
        )
        prediction_rows = json.loads(args.predictions.read_text(encoding="utf-8"))
        filtered_rows, filter_counts = _filter_prediction_rows_official(
            prediction_rows, args.annotations_dir, args.images_dir
        )
        filtered_predictions = args.output.with_name(
            f"{args.output.stem}.predictions.official-filter.local.json"
        )
        filtered_predictions.write_text(
            json.dumps(filtered_rows, ensure_ascii=False), encoding="utf-8"
        )
        evaluation_predictions = filtered_predictions
    else:
        counts, image_id_map = convert_ground_truth(
            args.annotations_dir, args.images_dir, ground_truth, args.ignore_regions
        )
    metrics, evaluator = evaluate(ground_truth, evaluation_predictions, image_id_map)
    prediction_count = len(json.loads(evaluation_predictions.read_text(encoding="utf-8")))
    ignore_note = {
        "crowd-per-class": "crowd-per-class approximates VisDrone class-agnostic ignored-region suppression",
        "drop": "ignored GT rows are removed; detections are not filtered against ignored regions",
        "official-filter": "VisDrone official 50% pixel-coverage ignore-region filter applied to GT and detections before COCO evaluation",
    }[args.ignore_regions]
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
            "ignore_note": ignore_note,
        },
        "counts": {
            **counts,
            "predictions": prediction_count,
            **filter_counts,
        },
        "metrics": metrics,
        "evaluator": evaluator,
        "inputs": {
            "annotations_dir": str(args.annotations_dir.resolve()),
            "images_dir": str(args.images_dir.resolve()),
            "predictions": str(args.predictions.resolve()),
            "evaluation_predictions": str(evaluation_predictions.resolve()),
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
