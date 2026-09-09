"""Detect overlap at 32^2/96^2 through the actual COCO evaluator."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_visdrone_coco_style import evaluate


def test_exact_area_thresholds_only_belong_to_next_bin(tmp_path):
    gt = {
        "info": {},
        "images": [{"id": 1, "file_name": "one.jpg", "width": 128, "height": 128}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 32, 32],
                "area": 1024,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [32, 32, 96, 96],
                "area": 9216,
                "iscrowd": 0,
            },
        ],
    }
    pred = [
        {"image_id": "one", "category_id": 1, "bbox": [0, 0, 32, 32], "score": 0.9},
        {"image_id": "one", "category_id": 1, "bbox": [32, 32, 96, 96], "score": 0.8},
    ]
    gt_path, pd_path = tmp_path / "gt.json", tmp_path / "pd.json"
    gt_path.write_text(json.dumps(gt))
    pd_path.write_text(json.dumps(pred))
    metrics, metadata = evaluate(gt_path, pd_path, {"one": 1})
    assert metrics["APs"] is None
    assert abs(metrics["APm"] - 100) < 1e-6
    assert abs(metrics["APl"] - 100) < 1e-6
    assert metadata["max_dets"] == [1, 10, 100, 500]
