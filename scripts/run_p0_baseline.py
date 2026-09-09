"""Locked acce839 baseline with complete online epoch assignment statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from multiprocessing import freeze_support
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "YOLO-Master-baseline"
BASE = "acce839c7e895d6b179de7f7093fa879e237cc7b"
NAME = "p0-locked-s20260824-stats120"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="One epoch on bundled mini data, never a P0 result",
    )
    args = parser.parse_args()
    print("P0 preflight: checking locked source and training configuration", flush=True)
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if commit != BASE:
        raise RuntimeError(f"Locked baseline drift: {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no", "--", "ultralytics"],
        text=True,
        timeout=30,
    )
    if dirty.strip():
        raise RuntimeError(f"Baseline implementation has local edits: {dirty}")
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    import torch
    from assignment_observer import AssignmentObserver
    from ultralytics import YOLO
    from ultralytics.utils.tal import TaskAlignedAssigner

    torch.set_num_threads(2)
    model_path = SOURCE / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml"
    train = {
        "data": str(ROOT / "configs/VisDrone-full.yaml"),
        "epochs": 120,
        "imgsz": 800,
        "batch": 4,
        "workers": 0,
        "device": 0,
        "seed": 20260824,
        "deterministic": True,
        "amp": False,
        "patience": 0,
        "pretrained": False,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "lora_r": 0,
        "nbs": 64,
        "val": True,
        "max_det": 500,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 10,
        "save": True,
        "save_period": 10,
        "project": "F:/YOLO-Master-A2-P1",
        "name": NAME,
        "exist_ok": False,
        "plots": True,
        "verbose": True,
    }
    if args.cpu_smoke:
        train.update(
            data=str(SOURCE / "agent/assets/mini-detect/mini_detect.yaml"),
            epochs=1,
            imgsz=64,
            batch=1,
            device="cpu",
            mosaic=0.0,
            close_mosaic=0,
            project=str(ROOT / ".local/cpu-smoke"),
            name="p0-observer",
            plots=False,
        )
    if args.check_only:
        print(
            json.dumps(
                {
                    "base": commit,
                    "model": str(model_path),
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "train": train,
                    "resume": str(args.resume) if args.resume else None,
                },
                indent=2,
            )
        )
        return
    if not args.resume and (Path(train["project"]) / train["name"]).exists():
        raise RuntimeError("Output directory exists: use validated checkpoint resume")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        recorded = checkpoint.get("train_args") or {}
        for key in (
            "epochs",
            "seed",
            "batch",
            "imgsz",
            "optimizer",
            "warmup_bias_lr",
            "nbs",
        ):
            if recorded.get(key) != train[key]:
                raise RuntimeError(f"Resume protocol drift: {key}")
        if (
            not -1 <= checkpoint.get("epoch", -2) < 120
            or checkpoint.get("optimizer") is None
        ):
            raise RuntimeError("Checkpoint is not resumable")
        bootstrap = checkpoint["epoch"] == -1
        del checkpoint
    observer = AssignmentObserver(
        TaskAlignedAssigner, expected_batch=train["batch"]
    ).install()
    try:
        model = YOLO(
            str(model_path if args.resume and bootstrap else args.resume or model_path)
        )
        observer.attach(model)
        if args.resume and not bootstrap:
            model.train(resume=str(args.resume), device=0, workers=0)
        else:
            if args.resume:
                print(
                    "Bootstrap checkpoint: replaying first epoch from the same YAML/seed; no completed epoch is lost"
                )
                train["exist_ok"] = True
            model.train(**train)
    finally:
        observer.close()


if __name__ == "__main__":
    freeze_support()
    main()
