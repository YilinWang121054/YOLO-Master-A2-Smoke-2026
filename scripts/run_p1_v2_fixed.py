"""First protocol-complete P1 v2 fixed control; online statistics are mandatory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "YOLO-Master"
CONFIG = ROOT / "configs/p1-v2-fixed-s20260825.train.json"


def validate_train_args(expected, recorded):
    for key, value in expected.items():
        actual = recorded.get(key)
        # Ultralytics serializes device=0 as "0". This is not a training change.
        if key == "device" and str(actual) == str(value):
            continue
        if actual != value:
            raise ValueError(f"Frozen training parameter changed: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.resume and args.cpu_smoke:
        raise ValueError("The smoke run is not a formal resume target")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    git = ["git", "-C", str(SOURCE)]
    assert subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == config["source_commit"]
    assert not subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"]).strip()
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    import torch
    from assignment_coverage import missing_assignment_epochs
    from assignment_observer import AssignmentObserver
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.tal import TaskAlignedAssigner

    torch.set_num_threads(2)
    model_path = SOURCE / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml"
    train = {**config["train"], "data": str(ROOT / "configs/VisDrone-full.yaml"),
             "project": "F:/YOLO-Master-A2-P1", "name": config["name"], "exist_ok": False}
    if args.cpu_smoke:
        train.update(data=str(SOURCE / "agent/assets/mini-detect/mini_detect.yaml"), epochs=1,
                     imgsz=64, batch=1, device="cpu", mosaic=0.0, close_mosaic=0, plots=False,
                     project=str(ROOT / ".local/cpu-smoke"), name="p1-v2-fixed-observer-20260913")
    resolved = vars(get_cfg(overrides=train))
    provenance = {"source_commit": config["source_commit"], "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "observer_sha256": hashlib.sha256((ROOT / "scripts/assignment_observer.py").read_bytes()).hexdigest(),
                  "model_yaml_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  "resolved_cfg": resolved, "command": sys.argv,
                  "started_at": datetime.now(timezone.utc).isoformat()}
    if args.check_only:
        print(json.dumps(provenance, ensure_ascii=True, default=str), flush=True)
        return
    run_dir = Path(train["project"]) / train["name"]
    if not args.resume and run_dir.exists():
        raise RuntimeError("Run exists. Use validated checkpoint recovery; never create a suffixed duplicate.")
    bootstrap = False
    if args.resume:
        if args.resume.resolve().parent != (run_dir / "weights").resolve():
            raise ValueError("Resume checkpoint is outside this experiment")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_train_args(config["train"], ckpt["train_args"])
        epoch = ckpt.get("epoch", -2)
        if not -1 <= epoch < 119 or ckpt.get("optimizer") is None:
            raise ValueError("Not an incomplete resumable checkpoint")
        if missing_assignment_epochs(run_dir, epoch + 1):
            raise ValueError("Checkpoint is ahead of online statistics")
        bootstrap = epoch == -1
        del ckpt
    event_dir = Path("F:/YOLO-Master-A2-P1/recovery-logs") / train["name"]
    event_dir.mkdir(parents=True, exist_ok=True)
    event = event_dir / ("runner-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    with event.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=True, indent=2, default=str)
    observer = AssignmentObserver(TaskAlignedAssigner, expected_batch=train["batch"]).install()
    try:
        model = YOLO(str(args.resume if args.resume and not bootstrap else model_path))
        observer.attach(model)
        if args.resume and not bootstrap:
            model.train(resume=str(args.resume), device=0, workers=0)
        else:
            if bootstrap:
                train["exist_ok"] = True
                print("Replaying incomplete first epoch from the frozen YAML and seed", flush=True)
            model.train(**train)
        if missing_assignment_epochs(run_dir, train["epochs"]):
            raise RuntimeError("Training ended without complete online assignment evidence")
    finally:
        observer.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
