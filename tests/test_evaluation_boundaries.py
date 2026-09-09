"""Detect overlap at 32^2/96^2 through the actual COCO evaluator."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_visdrone_coco_style import _official_filter_rows, evaluate


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


def test_matlab_half_rounding_and_border_pixel_filter():
    # Ignore region occupies only the last column. A box rounded to x=8
    # includes column 10; x=7 would miss it. MATLAB rounds 7.5 up to 8.
    gt = np.array([[10, 1, 1, 9, 0, 0, 0, 0]], dtype=np.int32)
    det = np.array([[7.5, 1, 2, 2, 0.9, 1], [7.49, 1, 2, 2, 0.8, 1]], dtype=np.float64)
    _, kept = _official_filter_rows(gt, det, 10, 10)
    assert kept[:, 4].tolist() == [0.8]
