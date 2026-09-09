"""Strict Python port of the VisDrone DET MATLAB evaluation utilities.

The port keeps MATLAB's positive half-up rounding for ignore-region boxes,
uses stable score ordering, and computes ignored-region overlap against the
ignored boxes themselves. It is intended for CPU audit runs; the original
MATLAB runner remains the reference when a licensed MATLAB is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def comp_oas(dt, gt, ignored):
    """Translate compOas.m in float64, using xywh input without epsilon clipping."""
    output = np.zeros((len(dt), len(gt)), dtype=np.float64)
    for i, box in enumerate(dt):
        extent = np.minimum(box[:2] + box[2:], gt[:, :2] + gt[:, 2:]) - np.maximum(
            box[:2], gt[:, :2]
        )
        valid = (extent > 0).all(-1)
        intersection = extent[valid, 0] * extent[valid, 1]
        det_area = box[2] * box[3]
        gt_area = gt[valid, 2] * gt[valid, 3]
        union = np.where(ignored[valid], det_area, det_area + gt_area - intersection)
        output[i, valid] = intersection / union
    return output


def matlab_round_positive(values: np.ndarray) -> np.ndarray:
    """MATLAB round for the positive image coordinates used by VisDrone."""
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int32)


def _filter_rows(gt: np.ndarray, det: np.ndarray, height: int, width: int):
    ignored = gt[gt[:, 5] == 0, :4].clip(min=1)
    kept = gt[gt[:, 5] != 0].copy()
    # eval_det.py performs this conversion after dropObjectsInIgr: valid
    # VisDrone rows have score=1 and become ignore=0 for evalRes.m.
    if len(kept):
        kept[:, 4] = 1 - kept[:, 4]
    if not len(ignored):
        return kept, det
    mask = np.zeros((height, width), dtype=np.int64)
    for x1, y1, w, h in ignored:
        x1, y1 = int(x1), int(y1)
        x2, y2 = min(x1 + int(w), width), min(y1 + int(h), height)
        if x2 > x1 and y2 > y1:
            mask[y1 - 1 : y2, x1 - 1 : x2] = 1

    def keep(rows):
        result = []
        for row in rows:
            x, y, w, h = matlab_round_positive(row[:4]).clip(min=1)
            x = max(1, min(width, int(x)))
            y = max(1, min(height, int(y)))
            w, h = int(w), int(h)
            if w <= 0 or h <= 0:
                continue
            # I(y,x)+I(y+h,x+w)-I(y,x+w)-I(y+h,x) excludes top/left pixels.
            # This zero-based slice counts exactly that MATLAB integral query.
            covered = mask[y : min(height, y + h), x : min(width, x + w)].sum()
            if covered / float(w * h) < 0.5:
                result.append(row)
        return np.asarray(result, dtype=rows.dtype).reshape((-1, rows.shape[1]))

    return keep(kept), keep(det)


def _eval_res(gt0, det0, threshold):
    gt0 = np.asarray(gt0, dtype=np.float64).reshape((-1, 5))
    det0 = np.asarray(det0, dtype=np.float64).reshape((-1, 5))
    det = det0[np.argsort(-det0[:, 4], kind="mergesort")] if len(det0) else det0
    gt_order = np.argsort(gt0[:, 4], kind="mergesort") if len(gt0) else []
    gt = gt0[gt_order] if len(gt0) else gt0
    gt[:, 4] *= -1
    dt = np.concatenate((det, np.zeros((len(det), 1))), axis=1)
    overlap = comp_oas(det[:, :4], gt[:, :4], gt[:, 4] == -1)
    for d in range(len(det)):
        best = threshold
        best_g = -1
        best_type = 0
        for g in range(len(gt)):
            marker = gt[g, 4]
            if marker == 1:
                continue
            if best_type and marker == -1:
                break
            if overlap[d, g] < best:
                continue
            best = overlap[d, g]
            best_g = g
            best_type = 1 if marker == 0 else -1
        dt[d, 5] = best_type
        if best_type == 1:
            gt[best_g, 4] = 1
    return gt, dt


def _voc_ap(rec, prec):
    mrec = np.concatenate(([0], rec, [1]))
    mprec = np.concatenate(([0], prec, [0]))
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[changed] - mrec[changed - 1]) * mprec[changed]))


def evaluate(all_gt, all_det, heights, widths):
    filtered = [
        _filter_rows(gt, dt, h, w)
        for gt, dt, h, w in zip(all_gt, all_det, heights, widths)
    ]
    ap = np.zeros((10, 10), dtype=np.float64)
    ar = np.zeros((10, 10, 4), dtype=np.float64)
    eval_classes = []
    for category in range(1, 11):
        # The original calcAccuracy.m appends once for EACH image containing
        # this class. Preserve that repeated indexing, even though it weights
        # categories by image prevalence rather than averaging unique classes.
        eval_classes.extend(
            [category - 1 for gt, _ in filtered if np.any(gt[:, 5] == category)]
        )
        for ti, threshold in enumerate(np.linspace(0.5, 0.95, 10)):
            last_gt, last_dt, last_id = None, None, None
            for mi, max_det in enumerate((1, 10, 100, 500)):
                gt_matches, dt_matches = [], []
                for gt, det in filtered:
                    gt1, dt1 = _eval_res(
                        gt[gt[:, 5] == category, :5],
                        det[:max_det][det[:max_det, 5] == category, :5],
                        threshold,
                    )
                    gt_matches.append(gt1[:, 4])
                    dt_matches.append(dt1[:, 4:6])
                gt_match = np.concatenate(gt_matches) if gt_matches else np.zeros(0)
                dt_match = (
                    np.concatenate(dt_matches) if dt_matches else np.zeros((0, 2))
                )
                order = (
                    np.argsort(-dt_match[:, 0], kind="mergesort")
                    if len(dt_match)
                    else []
                )
                tp = (
                    np.cumsum(dt_match[order, 1] == 1) if len(dt_match) else np.zeros(0)
                )
                rec = tp / max(1, len(gt_match))
                ar[category - 1, ti, mi] = np.max(rec) * 100 if len(rec) else 0
                if mi == 3:
                    last_gt, last_dt, last_id = gt_match, dt_match, order
            if last_dt is not None and len(last_dt):
                tp = np.cumsum(last_dt[last_id, 1] == 1)
                fp = np.cumsum(last_dt[last_id, 1] == 0)
                rec = tp / max(1, len(last_gt))
                ap[category - 1, ti] = _voc_ap(rec, tp / np.maximum(1, tp + fp)) * 100
    return {
        "AP": float(np.mean(ap[eval_classes, :])),
        "AP50": float(np.mean(ap[eval_classes, 0])),
        "AP75": float(np.mean(ap[eval_classes, 5])),
        "AR1": float(np.mean(ar[eval_classes, :, 0])),
        "AR10": float(np.mean(ar[eval_classes, :, 1])),
        "AR100": float(np.mean(ar[eval_classes, :, 2])),
        "AR500": float(np.mean(ar[eval_classes, :, 3])),
        "eval_classes": eval_classes,
    }


def read(path, dtype):
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 8), dtype=dtype)
    return np.loadtxt(path, delimiter=",", dtype=dtype, ndmin=2, usecols=range(8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_gt, all_det, heights, widths = [], [], [], []
    for gt_path in sorted((args.dataset_dir / "annotations").glob("*.txt")):
        image = cv2.imread(str(args.dataset_dir / "images" / f"{gt_path.stem}.jpg"))
        if image is None:
            raise FileNotFoundError(gt_path)
        heights.append(image.shape[0])
        widths.append(image.shape[1])
        all_gt.append(read(gt_path, np.int32))
        all_det.append(read(args.results_dir / gt_path.name, np.float64))
    result = {
        "evaluator": "VisDrone DET MATLAB-source Python audit port",
        "runtime_parity": "pending original MATLAB execution",
        "images": len(all_gt),
        "max_dets": [1, 10, 100, 500],
        "iou_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
        "metrics": evaluate(all_gt, all_det, heights, widths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                **result,
                "metrics": {
                    key: value
                    for key, value in result["metrics"].items()
                    if key != "eval_classes"
                },
            }
        )
    )


if __name__ == "__main__":
    main()
