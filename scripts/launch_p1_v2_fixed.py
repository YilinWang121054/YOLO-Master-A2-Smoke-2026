"""Launch the first v2 fixed control once; refuse existing output or an occupied training process."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
NAME = "p1-v2-fixed-s20260825-stats120"


def main():
    run_dir = Path("F:/YOLO-Master-A2-P1") / NAME
    if run_dir.exists():
        raise RuntimeError("Output exists; use resume_p1_training.py with the v2 recovery config")
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or []).lower()
            if process.pid != psutil.Process().pid and any(token in command for token in (
                "run_p1_seed.py", "run_p0_baseline.py", "run_p1_v2_fixed.py", "run_p1_tal_seed1.py")):
                raise RuntimeError(f"Training process already active: {process.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    runner = ROOT / "scripts/run_p1_v2_fixed.py"
    subprocess.run([sys.executable, "-X", "utf8", str(runner), "--check-only"], check=True, capture_output=True)
    log_dir = Path("F:/YOLO-Master-A2-P1/recovery-logs") / NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-X", "utf8", str(runner)]
    with (log_dir / "initial.stdout.log").open("xb") as stdout, (log_dir / "initial.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    record = {"status": "launched_not_yet_verified", "pid": process.pid, "command": command,
              "started_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
              "log_dir": str(log_dir), "epochs": 120, "online_statistics_required": True}
    with (log_dir / "launch.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=True, indent=2)
    print(json.dumps(record, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
