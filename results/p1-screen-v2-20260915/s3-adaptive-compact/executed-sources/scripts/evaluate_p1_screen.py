"""CPU evaluation of one declared screening epoch; never use built-in mAP for selection."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from p1_screen_contract import (
    CONFIG,
    DATA,
    PROJECT,
    ROOT,
    SOURCE,
    configuration,
    exclusive_lock,
    now,
    read,
    save,
    sha,
    train_args,
    verify_completion,
    verify_frozen,
)
from run_p1_v2_fixed import validate_train_args

ORIGINAL = Path("F:/datasets/VisDrone/VisDrone2019-DET-val")


def verify_result(case_id, epoch):
    folder = PROJECT / "evaluation" / case_id / f"epoch-{epoch:03d}"
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    result = read(marker)
    checkpoint = PROJECT / case_id / "weights" / f"epoch{epoch - 1}.pt"
    if result["config_sha256"] != sha(CONFIG) or result["checkpoint_sha256"] != sha(
        checkpoint
    ):
        raise ValueError("Evaluation provenance drift")
    for path, digest in result["files_sha256"].items():
        if sha(folder / path) != digest:
            raise ValueError("Evaluation evidence changed")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args()
    verify_frozen()
    if args.epoch not in configuration()["evaluation_epochs"] or not verify_completion(
        args.case
    ):
        raise ValueError(
            "Only the five declared epochs of a completed case can be evaluated"
        )
    with exclusive_lock(PROJECT / "locks" / f"eval-{args.case}-{args.epoch}.lock"):
        if verify_result(args.case, args.epoch):
            return
        folder = PROJECT / "evaluation" / args.case / f"epoch-{args.epoch:03d}"
        attempt = folder / ("attempt-" + now().replace(":", "").replace(".", ""))
        attempt.mkdir(parents=True, exist_ok=False)
        checkpoint = PROJECT / args.case / "weights" / f"epoch{args.epoch - 1}.pt"
        checkpoint_sha = sha(checkpoint)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.chdir(SOURCE)
        sys.path.insert(0, str(SOURCE))
        import torch
        from evaluate_visdrone_coco_style import (
            _filter_prediction_rows_official,
            convert_ground_truth_official_filter,
        )
        from evaluate_visdrone_coco_style import evaluate as coco_evaluate
        from ultralytics import YOLO

        torch.set_num_threads(configuration()["resources"]["cpu_threads"])
        torch.set_num_interop_threads(1)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["epoch"] != args.epoch - 1:
            raise ValueError("Checkpoint epoch mismatch")
        validate_train_args(
            {k: v for k, v in train_args(args.case).items() if k != "exist_ok"},
            payload["train_args"],
        )
        del payload
        protocol = {
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": sha(CONFIG),
            "case": args.case,
            "epoch": args.epoch,
            "device": "cpu",
            "quantize": 32,
            "threads": 2,
            "batch": 4,
            "imgsz": 800,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 500,
            "rect": True,
            "augment": False,
            "workers": 0,
            "started_at": now(),
        }
        save(attempt / "launch.json", protocol, exclusive=True)
        model = YOLO(str(checkpoint))
        observed = {}
        model.add_callback(
            "on_val_end", lambda validator: observed.update(images=validator.seen)
        )
        model.val(
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
            project=str(attempt),
            name="val",
            exist_ok=False,
            verbose=False,
        )
        if observed.get("images") != 548:
            raise ValueError("Expected complete 548-image validation")
        del model
        predictions = attempt / "val/predictions.json"
        commands = [
            [
                sys.executable,
                "-X",
                "utf8",
                str(ROOT / "scripts/export_visdrone_det_txt.py"),
                "--predictions",
                str(predictions),
                "--annotations-dir",
                str(ORIGINAL / "annotations"),
                "--output-dir",
                str(attempt / "det"),
            ],
            [
                sys.executable,
                "-X",
                "utf8",
                str(ROOT / "scripts/evaluate_visdrone_det_strict.py"),
                "--dataset-dir",
                str(ORIGINAL),
                "--results-dir",
                str(attempt / "det"),
                "--output",
                str(attempt / "det-metrics.json"),
            ],
        ]
        for name, command in zip(("conversion.log", "det-evaluation.log"), commands):
            with (attempt / name).open("xb") as stream:
                subprocess.run(
                    command,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
        gt = attempt / "gt.local.json"
        counts, mapping = convert_ground_truth_official_filter(
            ORIGINAL / "annotations", ORIGINAL / "images", gt
        )
        rows, changes = _filter_prediction_rows_official(
            read(predictions), ORIGINAL / "annotations", ORIGINAL / "images"
        )
        filtered = attempt / "filtered-predictions.local.json"
        save(filtered, rows, exclusive=True)
        area, metadata = coco_evaluate(gt, filtered, mapping)
        save(
            attempt / "area-metrics.json",
            {"metrics": area, "evaluator": metadata, "counts": {**counts, **changes}},
            exclusive=True,
        )
        det = read(attempt / "det-metrics.json")["metrics"]
        if sha(checkpoint) != checkpoint_sha:
            raise ValueError("Checkpoint changed during evaluation")
        files = [
            attempt / name
            for name in (
                "launch.json",
                "det-metrics.json",
                "area-metrics.json",
                "conversion.log",
                "det-evaluation.log",
            )
        ]
        files.append(predictions)
        save(
            folder / "complete.json",
            {
                **protocol,
                "completed_at": now(),
                "images": 548,
                "metrics": {
                    **{key: det[key] for key in ("AP", "AP50", "AP75", "AR500")},
                    **{key: area[key] for key in ("APs", "APm", "APl", "ARs@500")},
                },
                "commands": commands,
                "files_sha256": {
                    p.relative_to(folder).as_posix(): sha(p) for p in files
                },
                "scope": "20-epoch screening, not a P1 improvement claim",
            },
            exclusive=True,
        )


if __name__ == "__main__":
    main()
