#!/usr/bin/env python3
"""Convert Ultralytics-style prediction JSON to VisDrone DET TXT files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-dets", type=int, default=500)
    return parser.parse_args()


def image_key(row: dict) -> str:
    name = row.get("file_name") or row.get("image_id")
    if not name:
        raise ValueError(f"Prediction row has no file_name/image_id: {row}")
    return Path(str(name)).stem


def main() -> None:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    if not args.annotations_dir.is_dir():
        raise FileNotFoundError(args.annotations_dir)

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        category = int(row["category_id"])
        if not 1 <= category <= 10:
            continue
        grouped[image_key(row)].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_files = sorted(args.annotations_dir.glob("*.txt"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotation TXT files in {args.annotations_dir}")

    total_rows = 0
    image_counts: dict[str, int] = {}
    for annotation_file in annotation_files:
        key = annotation_file.stem
        detections = sorted(
            grouped.get(key, []), key=lambda row: float(row["score"]), reverse=True
        )[: args.max_dets]
        output_file = args.output_dir / annotation_file.name
        lines = []
        for row in detections:
            x, y, w, h = (float(value) for value in row["bbox"])
            score = float(row["score"])
            category = int(row["category_id"])
            lines.append(f"{x:.6f},{y:.6f},{w:.6f},{h:.6f},{score:.8f},{category},-1,-1")
        output_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
        image_counts[key] = len(detections)
        total_rows += len(detections)

    manifest = {
        "predictions": str(args.predictions.resolve()),
        "annotations_dir": str(args.annotations_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "images": len(annotation_files),
        "detections": total_rows,
        "max_dets_per_image": args.max_dets,
        "format": "bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion",
        "category_range": "1..10",
        "image_detection_counts": image_counts,
    }
    (args.output_dir / "conversion-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
