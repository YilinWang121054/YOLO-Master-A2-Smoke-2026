"""Evaluate completed A2 runs on CPU, independently of the single GPU queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from assignment_coverage import missing_assignment_epochs
from resume_p1_training import read_results, running_processes

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("F:/YOLO-Master-A2-P1")
STATE_DIR = PROJECT / "closure-evaluation"
RUNS = (
    "p1-fixed-s20260826",
    "p0-locked-s20260824-stats120",
    "p1-adaptive-s20260826",
    "p1-tal-s20260826",
)
RUNNER = ROOT / "scripts/evaluate_completed_run.py"


def state(phase, **extra):
    data = {
        "phase": phase,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    temporary = STATE_DIR / "cpu-queue-state.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_DIR / "cpu-queue-state.json")


def evaluation_pids():
    pids = []
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if process.info["name"].lower() in ("python.exe", "pythonw.exe") and any(
                Path(arg).name == "evaluate_completed_run.py"
                for arg in process.info["cmdline"] or []
            ):
                pids.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "cpu-supervisor.lock").open("a+b") as lock:
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
        while True:
            active = evaluation_pids()
            if active:
                state("active", pids=active)
                time.sleep(45)
                continue
            pending = []
            for name in RUNS:
                marker = STATE_DIR / name / "state.json"
                previous = (
                    json.loads(marker.read_text(encoding="utf-8"))
                    if marker.exists()
                    else {}
                )
                if previous.get("phase") == "complete-local-review":
                    continue
                pending.append(name)
                run = PROJECT / name
                _, rows = read_results(run / "results.csv")
                if (
                    len(rows) != 120
                    or not (run / "weights/last.pt").exists()
                    or running_processes(name, run)
                ):
                    continue
                if name.startswith("p0-"):
                    missing = missing_assignment_epochs(run, 120)
                    if missing:
                        state(
                            "blocked",
                            run=name,
                            reason="P0 assignment coverage incomplete",
                            missing=missing,
                        )
                        return 2
                log_dir = ROOT / "logs/closure-evaluation" / name
                log_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                command = [sys.executable, str(RUNNER), "--run", name, "--threads", "2"]
                state("evaluating", run=name, command=command)
                environment = {
                    **os.environ,
                    "PYTHONUTF8": "1",
                    "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": "-1",
                    "OMP_NUM_THREADS": "2",
                }
                with (log_dir / f"cpu-{stamp}.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if result.returncode:
                    state(
                        "blocked",
                        run=name,
                        returncode=result.returncode,
                        log=str(log.name),
                    )
                    return 2
                break
            else:
                if not pending:
                    state("complete-local-review", runs=list(RUNS))
                    return 0
                state("waiting-for-training", pending=pending)
                time.sleep(45)


if __name__ == "__main__":
    raise SystemExit(main())
