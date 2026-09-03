#!/usr/bin/env python3
"""Validate and resume an interrupted YOLO-Master P1 training run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_DIR: Path | None = None


def emit(status: str, **details: Any) -> int:
    payload = {
        "status": status,
        "observed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        **details,
    }
    if EVENT_DIR is not None:
        write_state(EVENT_DIR / "latest-check.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status in {"already_running", "complete", "ready", "resumed"} else 2


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_results(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def running_processes(experiment_id: str, run_dir: Path) -> list[int]:
    try:
        import psutil
    except ImportError:
        return []

    needles = {experiment_id.lower(), str(run_dir).lower()}
    found: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = [str(item) for item in (process.info.get("cmdline") or [])]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        lowered = [item.lower() for item in cmdline]
        joined = " ".join(lowered)
        is_yolo_train = "train" in lowered and any(Path(item.strip('"')).name.lower() == "yolo.exe" for item in cmdline)
        is_explicit_runner = any("run_p1_tal_seed1.py" in item for item in lowered)
        if (is_yolo_train or is_explicit_runner) and any(needle in joined for needle in needles):
            found.append(int(process.info["pid"]))
    return sorted(set(found))


def current_commit(workspace: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def wait_until_stable(path: Path, seconds: float = 2.0) -> None:
    first = path.stat()
    time.sleep(seconds)
    second = path.stat()
    if (first.st_size, first.st_mtime_ns) != (second.st_size, second.st_mtime_ns):
        raise RuntimeError(f"checkpoint is still being written: {path}")


def inspect_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    wait_until_stable(path)
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root is not a dictionary")
    epoch = checkpoint.get("epoch")
    train_args = checkpoint.get("train_args") or {}
    if not isinstance(epoch, int) or not 0 <= epoch < int(config["expected_epochs"]):
        raise ValueError(f"invalid checkpoint epoch: {epoch!r}")
    if checkpoint.get("optimizer") is None:
        raise ValueError("optimizer state is missing")
    if checkpoint.get("scaler") is None:
        raise ValueError("gradient-scaler state is missing")
    if checkpoint.get("model") is None and checkpoint.get("ema") is None:
        raise ValueError("model and EMA states are both missing")
    if int(train_args.get("epochs", -1)) != int(config["expected_epochs"]):
        raise ValueError(f"epoch target mismatch: {train_args.get('epochs')!r}")
    if train_args.get("stal_mode") != config["expected_stal_mode"]:
        raise ValueError(f"STAL mode mismatch: {train_args.get('stal_mode')!r}")
    data_path = Path(str(train_args.get("data", "")))
    if not data_path.exists():
        raise FileNotFoundError(f"dataset config recorded by checkpoint is missing: {data_path}")
    return {
        "path": str(path),
        "epoch_zero_based": epoch,
        "completed_epochs": epoch + 1,
        "optimizer_present": True,
        "scaler_present": True,
        "stal_mode": train_args.get("stal_mode"),
        "data": str(data_path),
    }


def select_checkpoint(run_dir: Path, config: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    weights = run_dir / "weights"
    periodic = sorted(weights.glob("epoch*.pt"), key=lambda item: item.stat().st_mtime_ns, reverse=True)
    candidates = [weights / "last.pt", weights / "last_healthy.pt", *periodic]
    errors: list[dict[str, str]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            return inspect_checkpoint(candidate, config), errors
        except (AttributeError, EOFError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            errors.append({"path": str(candidate), "error": f"{type(error).__name__}: {error}"})
    return None, errors


def rewind_csv(path: Path, header: list[str], rows: list[dict[str, str]], keep_rows: int) -> Path:
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"results.pre-resume-{timestamp}.csv")
    shutil.copy2(path, backup)
    temporary = path.with_suffix(".resume-tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows[:keep_rows])
    os.replace(temporary, path)
    return backup


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    global EVENT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    EVENT_DIR = Path(config["recovery_log_dir"]).resolve()
    workspace = Path(config["workspace"]).resolve()
    run_dir = Path(config["run_dir"]).resolve()
    results_path = run_dir / "results.csv"
    expected_epochs = int(config["expected_epochs"])
    header, rows = read_results(results_path)
    if len(rows) >= expected_epochs:
        return emit("complete", completed_epochs=len(rows), run_dir=str(run_dir))

    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    checkpoint, checkpoint_errors = select_checkpoint(run_dir, config)
    if checkpoint is None:
        return emit("refused", reason="no valid resumable checkpoint", checkpoint_errors=checkpoint_errors)

    active_pids = running_processes(config["experiment_id"], run_dir)
    if active_pids:
        return emit(
            "already_running",
            pids=active_pids,
            csv_epochs=len(rows),
            checkpoint=checkpoint,
            checkpoint_errors=checkpoint_errors,
        )

    actual_commit = current_commit(workspace)
    if actual_commit != config["expected_git_commit"]:
        return emit(
            "refused",
            reason="code commit drift",
            expected_commit=config["expected_git_commit"],
            actual_commit=actual_commit,
        )

    checkpoint_epochs = int(checkpoint["completed_epochs"])
    csv_backup = None
    if len(rows) < checkpoint_epochs:
        return emit(
            "refused",
            reason="results.csv is behind the checkpoint; manual integrity review is required",
            csv_epochs=len(rows),
            checkpoint_epochs=checkpoint_epochs,
        )
    if len(rows) > checkpoint_epochs:
        if config.get("csv_rewind_policy") != "backup_and_truncate":
            return emit(
                "refused",
                reason="results.csv is ahead of the checkpoint",
                csv_epochs=len(rows),
                checkpoint_epochs=checkpoint_epochs,
            )
        if args.check_only:
            return emit(
                "ready",
                experiment_id=config["experiment_id"],
                csv_epochs=len(rows),
                checkpoint=checkpoint,
                checkpoint_errors=checkpoint_errors,
                git_commit=actual_commit,
                would_rewind_csv_to_epochs=checkpoint_epochs,
            )
        csv_backup = rewind_csv(results_path, header, rows, checkpoint_epochs)

    readiness = {
        "experiment_id": config["experiment_id"],
        "csv_epochs": checkpoint_epochs,
        "checkpoint": checkpoint,
        "checkpoint_errors": checkpoint_errors,
        "git_commit": actual_commit,
        "csv_backup": str(csv_backup) if csv_backup else None,
    }
    if args.check_only:
        return emit("ready", **readiness)

    yolo_executable = Path(config["yolo_executable"]).resolve()
    if not yolo_executable.exists():
        return emit("refused", reason=f"YOLO executable is missing: {yolo_executable}")
    overrides = config.get("resume_overrides", {})
    command = [
        str(yolo_executable),
        "train",
        f"model={checkpoint['path']}",
        "resume=True",
        *(f"{key}={value}" for key, value in overrides.items()),
    ]
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(config["recovery_log_dir"]).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"resume-{timestamp}.stdout.log"
    stderr_path = log_dir / f"resume-{timestamp}.stderr.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(workspace) + os.pathsep + environment.get("PYTHONPATH", "")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=creationflags,
        )

    state = {
        "schema_version": 1,
        "status": "resumed",
        "resumed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "pid": process.pid,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        **readiness,
    }
    write_state(log_dir / "latest-resume.json", state)
    return emit("resumed", **state)


if __name__ == "__main__":
    raise SystemExit(main())
