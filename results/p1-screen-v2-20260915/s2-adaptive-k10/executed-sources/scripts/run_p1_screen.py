"""Train one frozen 20-epoch case, retaining real statistics and recovery state."""

from __future__ import annotations

import argparse
import os
import sys

from assignment_coverage import missing_assignment_epochs
from p1_screen_contract import (
    CONFIG,
    MODEL,
    PROJECT,
    SOURCE,
    configuration,
    csv_rows,
    exclusive_lock,
    now,
    save,
    sha,
    train_args,
    verify_completion,
    verify_frozen,
    verify_source,
)
from run_p1_v2_fixed import validate_train_args


def recovery(run, train):
    from resume_p1_training import read_results, rewind_csv, select_checkpoint

    expected = {
        key: value for key, value in train.items() if key not in {"device", "exist_ok"}
    }
    config = {
        "expected_train_args": expected,
        "allow_bootstrap_checkpoint": True,
        "expected_epochs": 20,
        "expected_stal_mode": train["stal_mode"],
    }
    chosen, errors = select_checkpoint(run, config)
    if chosen is None:
        raise RuntimeError(
            f"No valid checkpoint; keep existing directory for review: {errors}"
        )
    completed = chosen["completed_epochs"]
    rows = csv_rows(run)
    if len(rows) < completed or missing_assignment_epochs(run, completed):
        raise RuntimeError("Checkpoint is ahead of CSV or real training statistics")
    if completed >= 20:
        return chosen, errors, None
    backup = None
    if len(rows) > completed:
        header, original_rows = read_results(run / "results.csv")
        backup = str(rewind_csv(run / "results.csv", header, original_rows, completed))
    return chosen, errors, backup


def record_completion(case_id, train):
    import torch

    run = PROJECT / case_id
    if len(csv_rows(run)) != 20 or missing_assignment_epochs(run, 20):
        raise ValueError("Full CSV/statistics are required")
    healthy = torch.load(
        run / "weights/last_healthy.pt", map_location="cpu", weights_only=False
    )
    if healthy["epoch"] != 19 or healthy["optimizer"] is None:
        raise ValueError("Healthy checkpoint does not attest epoch20")
    validate_train_args(
        {k: v for k, v in train.items() if k != "exist_ok"}, healthy["train_args"]
    )
    momenta = [
        g["momentum"] for g in healthy["optimizer"]["param_groups"] if "momentum" in g
    ]
    if not momenta or any(value != 0.9 for value in momenta):
        raise ValueError("Actual optimizer momentum changed")
    files = [
        run / "args.yaml",
        run / "results.csv",
        run / "weights/last.pt",
        run / "weights/last_healthy.pt",
    ]
    for epoch in range(1, 21):
        path = run / "assignment" / f"epoch-{epoch:03d}.json"
        from p1_screen_contract import read

        record = read(path)
        if record["batches"] != record["expected_batches"] or record["batches"] != 1618:
            raise ValueError("Incomplete full-train coverage")
        files.append(path)
    for epoch in configuration()["evaluation_epochs"]:
        checkpoint = run / "weights" / f"epoch{epoch - 1}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["epoch"] != epoch - 1:
            raise ValueError("Periodic checkpoint epoch mismatch")
        validate_train_args(
            {k: v for k, v in train.items() if k != "exist_ok"}, payload["train_args"]
        )
        files.append(checkpoint)
        del payload
    save(
        run / "training-complete.json",
        {
            "completed_at": now(),
            "case": case_id,
            "epochs": 20,
            "config_sha256": sha(CONFIG),
            "training_batches": 32360,
            "actual_momenta": momenta,
            "files_sha256": {p.relative_to(run).as_posix(): sha(p) for p in files},
            "scope": "Screening only, not P1 final evidence",
        },
        exclusive=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    verify_source() if args.check_only or args.cpu_smoke else verify_frozen()
    train = train_args(args.case)
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    import torch
    from assignment_observer import AssignmentObserver
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.tal import TaskAlignedAssigner

    torch.set_num_threads(2)
    resolved = vars(get_cfg(overrides=train))
    if args.check_only:
        print(
            {
                "case": args.case,
                "validated": True,
                "mode": resolved["stal_mode"],
                "epochs": resolved["epochs"],
            },
            flush=True,
        )
        return
    if args.cpu_smoke:
        train.update(
            data=str(SOURCE / "agent/assets/mini-detect/mini_detect.yaml"),
            epochs=1,
            imgsz=64,
            batch=1,
            device="cpu",
            mosaic=0.0,
            close_mosaic=0,
            project="F:/YOLO-Master-A2-P1/diagnostics",
            name=f"screen-v2-smoke-{args.case}",
        )
        observer = AssignmentObserver(TaskAlignedAssigner, expected_batch=1).install()
        try:
            model = YOLO(str(MODEL))
            observer.attach(model)
            model.train(**train)
        finally:
            observer.close()
        return
    with exclusive_lock(PROJECT / "locks" / f"train-{args.case}.lock"):
        if verify_completion(args.case):
            print("Already complete; no training started", flush=True)
            return
        run = PROJECT / args.case
        choice, errors, backup = None, [], None
        if run.exists():
            choice, errors, backup = recovery(run, train)
        if choice and choice["completed_epochs"] == 20:
            record_completion(args.case, train)
            return
        event = {
            "started_at": now(),
            "case": args.case,
            "resolved_args": resolved,
            "config_sha256": sha(CONFIG),
            "checkpoint": choice,
            "checkpoint_errors": errors,
            "csv_backup": backup,
            "command": sys.argv,
        }
        stamp = now().replace(":", "").replace(".", "")
        save(
            PROJECT / "logs" / args.case / f"runner-{stamp}.json", event, exclusive=True
        )
        observer = AssignmentObserver(TaskAlignedAssigner, expected_batch=4).install()
        try:
            resume = choice and choice["completed_epochs"] > 0
            model = YOLO(choice["path"] if resume else str(MODEL))
            observer.attach(model)

            def pause_at_checkpoint(trainer):
                if (PROJECT / "PAUSE").exists():
                    print(
                        "User pause requested; completed-epoch checkpoint preserved",
                        flush=True,
                    )
                    raise SystemExit(75)

            model.add_callback("on_fit_epoch_end", pause_at_checkpoint)
            if resume:
                model.train(resume=choice["path"], device=0, workers=0)
            else:
                if choice:
                    train["exist_ok"] = True
                    print(
                        "Replaying incomplete first epoch from frozen YAML and seed",
                        flush=True,
                    )
                model.train(**train)
            record_completion(args.case, train)
        finally:
            observer.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
