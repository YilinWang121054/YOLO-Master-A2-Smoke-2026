#!/usr/bin/env python3
"""Run adaptive seed 2 first, then pure TAL seed 2, with reboot-safe recovery.

The process is intentionally small and conservative: it never starts the TAL
run while adaptive is active or incomplete, and delegates checkpoint recovery
to resume_p1_training.py.  It can be started at logon by Task Scheduler and
also run detached during the current session.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master-A2-Smoke-2026")
WORKSPACE = Path(r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master")
PYTHON = Path(r"F:\conda-envs\yolo-master\python.exe")
RUNNER = ROOT / "scripts" / "run_p1_seed.py"
RESUMER = ROOT / "scripts" / "resume_p1_training.py"
ADAPTIVE_CONFIG = ROOT / "configs" / "p1-adaptive-s20260825.resume.json"
TAL_CONFIG = ROOT / "configs" / "p1-tal-s20260825.resume.json"
ADAPTIVE_RUN = Path(r"F:\YOLO-Master-A2-P1\p1-adaptive-s20260825")
TAL_RUN = Path(r"F:\YOLO-Master-A2-P1\p1-tal-s20260825")
CHAIN_LOG_DIR = Path(r"F:\YOLO-Master-A2-P1\recovery-logs\p1-seed2-chain")
TAL_TASK = "YOLO-Master-A2-P1-TAL-S20260825-Resume"
EXPECTED_EPOCHS = 120


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rows_in_results(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return sum(1 for row in csv.DictReader(handle) if any(str(value).strip() for value in row.values()))
    except (OSError, UnicodeError, csv.Error):
        return 0


def process_matches(token: str, run_dir: Path) -> list[int]:
    try:
        import psutil
    except ImportError:
        return []
    token = token.lower()
    run_token = str(run_dir).lower()
    pids: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = [str(item) for item in (process.info.get("cmdline") or [])]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        joined = " ".join(cmdline).lower()
        if token in joined or run_token in joined:
            if "run_p1_seed.py" in joined or "yolo.exe" in joined or "resume_p1_training.py" in joined:
                pids.append(int(process.info["pid"]))
    return sorted(set(pids))


def launch_resume(config: Path, state_dir: Path) -> dict[str, Any]:
    command = [str(PYTHON), str(RESUMER), "--config", str(config)]
    result = subprocess.run(
        command,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    payload = {
        "at": now(),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    write_state(state_dir / "last-resume-attempt.json", payload)
    return payload


def launch_fresh_tal(state_dir: Path) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = state_dir / "tal-initial.stdout.log"
    stderr_path = state_dir / "tal-initial.stderr.log"
    command = [str(PYTHON), str(RUNNER), "--mode", "tal", "--seed", "20260825", "--name", "p1-tal-s20260825"]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            env={**os.environ, "PYTHONPATH": str(WORKSPACE) + os.pathsep + os.environ.get("PYTHONPATH", "")},
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=creationflags,
        )
    return {
        "at": now(),
        "pid": process.pid,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def enable_tal_task(state_dir: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["schtasks.exe", "/Change", "/TN", TAL_TASK, "/ENABLE"],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "at": now(),
        "task": TAL_TASK,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }
    write_state(state_dir / "tal-task-enable.json", payload)
    return payload


def complete(run_dir: Path) -> bool:
    return (
        rows_in_results(run_dir) >= EXPECTED_EPOCHS
        and (run_dir / "weights" / "last.pt").exists()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not PYTHON.exists() or not RUNNER.exists() or not RESUMER.exists():
        raise SystemExit("required training files are missing")
    CHAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = CHAIN_LOG_DIR / "chain-state.json"
    last_resume_attempt = 0.0
    tal_enabled = False
    launched_tal = False

    while True:
        adaptive_rows = rows_in_results(ADAPTIVE_RUN)
        adaptive_pids = process_matches("p1-adaptive-s20260825", ADAPTIVE_RUN)
        adaptive_done = complete(ADAPTIVE_RUN) and not adaptive_pids

        if not adaptive_done and not adaptive_pids and ADAPTIVE_RUN.exists():
            if time.time() - last_resume_attempt >= 300:
                launch_resume(ADAPTIVE_CONFIG, CHAIN_LOG_DIR)
                last_resume_attempt = time.time()

        if adaptive_done:
            if not tal_enabled:
                enable_tal_task(CHAIN_LOG_DIR)
                tal_enabled = True
            tal_rows = rows_in_results(TAL_RUN)
            tal_pids = process_matches("p1-tal-s20260825", TAL_RUN)
            tal_done = complete(TAL_RUN) and not tal_pids
            if not tal_done and not tal_pids:
                if TAL_RUN.exists() and (TAL_RUN / "weights").exists():
                    if time.time() - last_resume_attempt >= 300:
                        launch_resume(TAL_CONFIG, CHAIN_LOG_DIR)
                        last_resume_attempt = time.time()
                elif not TAL_RUN.exists() and not launched_tal:
                    launch_info = launch_fresh_tal(CHAIN_LOG_DIR)
                    write_state(CHAIN_LOG_DIR / "tal-launch.json", launch_info)
                    launched_tal = True
            state = {
                "updated_at": now(),
                "phase": "tal",
                "adaptive_rows": adaptive_rows,
                "adaptive_pids": adaptive_pids,
                "adaptive_done": adaptive_done,
                "tal_rows": tal_rows,
                "tal_pids": tal_pids,
                "tal_done": tal_done,
                "tal_task_enabled": tal_enabled,
            }
            write_state(state_path, state)
            if tal_done or args.once:
                return 0
        else:
            write_state(
                state_path,
                {
                    "updated_at": now(),
                    "phase": "adaptive",
                    "adaptive_rows": adaptive_rows,
                    "adaptive_pids": adaptive_pids,
                    "adaptive_done": adaptive_done,
                    "tal_task_enabled": tal_enabled,
                },
            )
            if args.once:
                return 0
        time.sleep(max(15, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
