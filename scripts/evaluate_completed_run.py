"""Evaluate a completed A2 120-epoch checkpoint on CPU and archive evidence.

Predictions and optional online statistics are reused only when their provenance
matches. Dataset and checkpoint remain local; reports require author review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("F:/YOLO-Master-A2-P1")
WORK = PROJECT / "closure-evaluation"
DATA = ROOT / "configs/VisDrone-full.yaml"
ORIGINAL = Path("F:/datasets/VisDrone/VisDrone2019-DET-val")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        choices=(
            "p1-fixed-s20260826",
            "p1-adaptive-s20260826",
            "p1-tal-s20260826",
            "p0-locked-s20260824-stats120",
        ),
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run = PROJECT / args.run
    is_p0 = args.run.startswith("p0-")
    source = ROOT.parent / ("YOLO-Master-baseline" if is_p0 else "YOLO-Master")
    expected = (
        "acce839c7e895d6b179de7f7093fa879e237cc7b"
        if is_p0
        else "52c2befa50706b9dff13b6e0813b19413d9f532d"
    )
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(r["epoch"]) for r in rows] != list(range(1, 121)):
        raise ValueError("Expected exactly 120 sequential completed epochs")
    if is_p0:
        from assignment_coverage import missing_assignment_epochs

        missing = missing_assignment_epochs(run, 120)
        if missing:
            raise ValueError(f"Missing online assignment epochs: {missing}")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
        timeout=30,
    ).strip()
    if commit != expected:
        raise ValueError("Frozen source commit drift")
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "ultralytics",
        ],
        text=True,
        encoding="utf-8",
        timeout=30,
    ).strip()
    if dirty:
        raise ValueError("Frozen implementation has local edits")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.chdir(source)
    sys.path.insert(0, str(source))
    import torch
    from evaluate_visdrone_coco_style import (
        _filter_prediction_rows_official,
        convert_ground_truth_official_filter,
    )
    from evaluate_visdrone_coco_style import (
        evaluate as coco_evaluate,
    )
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = run / "weights/last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Completed Ultralytics checkpoints are stripped to epoch=-1. Require the
    # healthy checkpoint to independently attest epoch 119 and retain state.
    healthy = torch.load(
        run / "weights/last_healthy.pt", map_location="cpu", weights_only=False
    )
    if healthy.get("epoch") != 119:
        raise ValueError("Healthy checkpoint does not attest epoch 120")
    if payload.get("train_args", {}).get("epochs") != 120:
        raise ValueError("Checkpoint training target differs from 120 epochs")
    del healthy, payload
    work = WORK / args.run
    out = ROOT / "results/closure-evaluation" / args.run
    log_dir = ROOT / "logs/closure-evaluation" / args.run
    work.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"

    def status(phase, **extra):
        save(
            state_path,
            {
                "run": args.run,
                "phase": phase,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
        )
        print(f"{args.run}: {phase}", flush=True)

    protocol = {
        "source_commit": commit,
        "checkpoint_sha256": sha256(checkpoint),
        "data_sha256": sha256(DATA),
        "device": "cpu",
        "threads": args.threads,
        "imgsz": 800,
        "batch": 4,
        "quantize": 32,
        "max_det": 500,
        "conf": 0.001,
        "iou": 0.7,
        "rect": True,
        "augment": False,
        "workers": 0,
    }
    validation = work / "val"
    predictions = validation / "predictions.json"
    marker = work / "prediction-provenance.json"
    existing = (
        json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    )
    if not (
        existing
        and existing.get("protocol") == protocol
        and predictions.exists()
        and existing.get("sha256") == sha256(predictions)
    ):
        status("cpu-inference")
        model = YOLO(str(checkpoint))
        observed = {}
        model.add_callback(
            "on_val_end", lambda validator: observed.update(images=validator.seen)
        )
        result = model.val(
            data=str(DATA),
            imgsz=800,
            batch=4,
            workers=0,
            device="cpu",
            quantize=32,
            conf=0.001,
            iou=0.7,
            max_det=500,
            rect=True,
            augment=False,
            save_json=True,
            plots=False,
            project=str(work),
            name="val",
            exist_ok=True,
            verbose=False,
        )
        if observed.get("images") != 548:
            raise ValueError(f"Incomplete validation: {observed.get('images')}/548")
        if not predictions.exists():
            raise FileNotFoundError(predictions)
        save(
            marker,
            {
                "protocol": protocol,
                "sha256": sha256(predictions),
                "images": 548,
                "training_validator_metrics": result.results_dict,
            },
        )
        del model, result
    status("det-conversion")
    conversion_command = [
        sys.executable,
        str(ROOT / "scripts/export_visdrone_det_txt.py"),
        "--predictions",
        str(predictions),
        "--annotations-dir",
        str(ORIGINAL / "annotations"),
        "--output-dir",
        str(work / "det"),
    ]
    with (log_dir / "conversion.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            conversion_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    status("det-source-audit")
    det_output = work / "det-source-metrics.json"
    with (log_dir / "det-source-audit.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_visdrone_det_strict.py"),
                "--dataset-dir",
                str(ORIGINAL),
                "--results-dir",
                str(work / "det"),
                "--output",
                str(det_output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
        )
    status("coco-area-evaluation")
    gt = work / "gt.local.json"
    counts, mapping = convert_ground_truth_official_filter(
        ORIGINAL / "annotations", ORIGINAL / "images", gt
    )
    filtered, changes = _filter_prediction_rows_official(
        json.loads(predictions.read_text(encoding="utf-8")),
        ORIGINAL / "annotations",
        ORIGINAL / "images",
    )
    filtered_path = work / "filtered-predictions.local.json"
    save(filtered_path, filtered)
    metrics, metadata = coco_evaluate(gt, filtered_path, mapping)
    save(
        out / "coco-area-metrics.json",
        {
            "revision": "20260910-matlab-ignore-half-open-area-bins",
            "metrics": metrics,
            "evaluator": metadata,
            "counts": {**counts, **changes},
            "area_definition": "original validation GT: <1024; [1024,9216); >=9216",
            "ignore_rule": "MATLAB dropObjectsInIgr source translation",
            "runtime_parity": "original MATLAB run pending",
        },
    )
    for path in (run / "args.yaml", run / "results.csv", det_output, marker):
        shutil.copy2(path, out / path.name)
    if is_p0:
        for path in sorted((run / "assignment").glob("epoch-???.json")):
            destination = out / "assignment" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/plot_assignment_evolution.py"),
                "--run-dir",
                str(run),
                "--output-dir",
                str(out / "figures"),
            ],
            check=True,
            timeout=300,
        )
    save(
        out / "evaluation-manifest.json",
        {
            "run": args.run,
            "protocol": protocol,
            "primary": "epoch-120 last.pt",
            "images": 548,
            "completed_epochs": 120,
            "predictions_sha256": sha256(predictions),
            "status": "evaluated_pending_official_runtime_parity_and_author_review",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_sha256": {
                p.name: sha256(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/evaluate_visdrone_coco_style.py",
                    ROOT / "scripts/evaluate_visdrone_det_strict.py",
                )
            },
        },
    )
    status("complete-local-review", output=str(out))


if __name__ == "__main__":
    main()
