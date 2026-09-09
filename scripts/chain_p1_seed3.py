#!/usr/bin/env python3
"""Run the three A2 P1 seed-3 modes serially with reboot-safe recovery."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assignment_coverage import missing_assignment_epochs

ROOT = Path(r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master-A2-Smoke-2026")
WORKSPACE = Path(r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master")
PYTHON = Path(r"F:\conda-envs\yolo-master\python.exe")
RUNNER = ROOT / "scripts" / "run_p1_seed.py"
RESUMER = ROOT / "scripts" / "resume_p1_training.py"
PROJECT = Path(r"F:\YOLO-Master-A2-P1")
SEED = 20260826
EXPECTED_EPOCHS = 120
CHAIN_LOG_DIR = PROJECT / "recovery-logs" / "p1-seed3-chain"

RUNS = (
    {
        "mode": "fixed",
        "token": "p1-fixed-s20260826",
        "config": ROOT / "configs" / "p1-fixed-s20260826.resume.json",
        "run_dir": PROJECT / "p1-fixed-s20260826",
    },
    {
        "mode": "baseline",
        "token": "p0-locked-s20260824-stats120",
        "config": ROOT / "configs" / "p0-locked-s20260824-stats120.resume.json",
        "run_dir": PROJECT / "p0-locked-s20260824-stats120",
        "command": [str(PYTHON), str(ROOT / "scripts" / "run_p0_baseline.py")],
    },
    {
        "mode": "adaptive",
        "token": "p1-adaptive-s20260826",
        "config": ROOT / "configs" / "p1-adaptive-s20260826.resume.json",
        "run_dir": PROJECT / "p1-adaptive-s20260826",
    },
    {
        "mode": "tal",
        "token": "p1-tal-s20260826",
        "config": ROOT / "configs" / "p1-tal-s20260826.resume.json",
        "run_dir": PROJECT / "p1-tal-s20260826",
    },
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def rows_in_results(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return sum(
                1
                for row in csv.DictReader(handle)
                if any(str(value).strip() for value in row.values())
            )
    except (OSError, UnicodeError, csv.Error):
        return 0


def complete(run_dir: Path) -> bool:
    if run_dir.name == "p0-locked-s20260824-stats120" and missing_assignment_epochs(
        run_dir, EXPECTED_EPOCHS
    ):
        return False
    return (
        rows_in_results(run_dir) >= EXPECTED_EPOCHS
        and (run_dir / "weights" / "last.pt").exists()
    )


def process_matches(token: str, run_dir: Path) -> list[int]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError(
            "psutil is required to prevent duplicate training"
        ) from error
    needle = token.lower()
    run_needle = str(run_dir).lower()
    pids: list[int] = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (process.info.get("name") or "").lower() not in {
                "python.exe",
                "pythonw.exe",
                "yolo.exe",
                "python",
                "python3",
            }:
                continue
            command = [str(item) for item in (process.info.get("cmdline") or [])]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        joined = " ".join(command).lower()
        is_runner = any(
            name in joined
            for name in (
                "run_p1_seed.py",
                "run_p0_baseline.py",
                "yolo.exe",
                "resume_p1_training.py",
            )
        )
        is_p0_runner = (
            token == "p0-locked-s20260824-stats120" and "run_p0_baseline.py" in joined
        )
        if is_runner and (needle in joined or run_needle in joined or is_p0_runner):
            pids.append(int(process.info["pid"]))
    return sorted(set(pids))


def launch_fresh(run: dict[str, Any]) -> dict[str, Any]:
    log_dir = CHAIN_LOG_DIR / run["token"]
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "initial.stdout.log"
    stderr_path = log_dir / "initial.stderr.log"
    command = run.get("command") or [
        str(PYTHON),
        str(RUNNER),
        "--mode",
        run["mode"],
        "--seed",
        str(SEED),
        "--name",
        run["token"],
    ]
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with (
        stdout_path.open("a", encoding="utf-8") as stdout,
        stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            env={
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(WORKSPACE)
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            },
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=flags,
        )
    payload = {
        "at": now(),
        "action": "fresh",
        "pid": process.pid,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    write_state(log_dir / "last-launch.json", payload)
    return payload


def launch_resume(run: dict[str, Any]) -> dict[str, Any]:
    log_dir = CHAIN_LOG_DIR / run["token"]
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(RESUMER), "--config", str(run["config"])]
    result = subprocess.run(
        command, cwd=WORKSPACE, capture_output=True, text=True, timeout=180, check=False
    )
    payload = {
        "at": now(),
        "action": "resume",
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    write_state(log_dir / "last-resume-attempt.json", payload)
    return payload


def run_once() -> int:
    CHAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        rows = rows_in_results(run["run_dir"])
        pids = process_matches(run["token"], run["run_dir"])
        if complete(run["run_dir"]):
            write_state(
                CHAIN_LOG_DIR / "last-complete.json",
                {"at": now(), "run": run["token"], "rows": rows, "pids": pids},
            )
            if pids:
                write_state(
                    CHAIN_LOG_DIR / "chain-state.json",
                    {
                        "at": now(),
                        "phase": "waiting-for-exit",
                        "run": run["token"],
                        "rows": rows,
                        "pids": pids,
                    },
                )
                return 0
            continue
        if pids:
            write_state(
                CHAIN_LOG_DIR / "chain-state.json",
                {
                    "at": now(),
                    "phase": "active",
                    "run": run["token"],
                    "rows": rows,
                    "pids": pids,
                },
            )
            return 0
        if (
            run["mode"] == "baseline"
            and rows >= EXPECTED_EPOCHS
            and not complete(run["run_dir"])
        ):
            write_state(
                CHAIN_LOG_DIR / "chain-state.json",
                {
                    "at": now(),
                    "phase": "blocked",
                    "run": run["token"],
                    "reason": "P0 training finished but checkpoint or online assignment evidence is incomplete",
                    "missing_epochs": missing_assignment_epochs(
                        run["run_dir"], EXPECTED_EPOCHS
                    ),
                },
            )
            return 2
        if not run["run_dir"].exists():
            launch = launch_fresh(run)
            write_state(
                CHAIN_LOG_DIR / "chain-state.json",
                {
                    "at": now(),
                    "phase": "launched-fresh",
                    "run": run["token"],
                    "rows": rows,
                    "launch": launch,
                },
            )
            return 0
        resume = launch_resume(run)
        status = "resumed" if resume["returncode"] == 0 else "blocked"
        write_state(
            CHAIN_LOG_DIR / "chain-state.json",
            {
                "at": now(),
                "phase": status,
                "run": run["token"],
                "rows": rows,
                "resume": resume,
            },
        )
        return 0 if status == "resumed" else 2
    write_state(
        CHAIN_LOG_DIR / "chain-state.json",
        {"at": now(), "phase": "complete", "runs": [run["token"] for run in RUNS]},
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not PYTHON.exists() or not RUNNER.exists() or not RESUMER.exists():
        raise SystemExit("required training files are missing")
    CHAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # OS-held byte lock is released even if the supervisor is killed/rebooted.
    # It also covers manually started supervisors outside Task Scheduler.
    with (CHAIN_LOG_DIR / "supervisor.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return 0
        return supervise(args)


def supervise(args: argparse.Namespace) -> int:
    while True:
        code = run_once()
        if args.once or code == 2:
            return code
        if (
            json.loads(
                (CHAIN_LOG_DIR / "chain-state.json").read_text(encoding="utf-8")
            ).get("phase")
            == "complete"
        ):
            return 0
        time.sleep(max(15, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
