"""Run a bounded MATLAB license check, preserving raw output in a new folder."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not 30 <= args.timeout <= 600:
        raise ValueError("Runtime timeout must be within 30..600 seconds")
    if not args.matlab.is_file():
        raise FileNotFoundError(args.matlab)
    args.output.mkdir(parents=True, exist_ok=False)
    code = "disp(version); assert(license('test','MATLAB') == 1); disp('A2_MATLAB_RUNTIME_OK');"
    command = [str(args.matlab), "-wait", "-batch", code]
    info = {"started_at": datetime.now(timezone.utc).isoformat(), "command": command,
            "timeout_seconds": args.timeout, "scope": "Runtime check only; no DET evaluation or training"}
    start = time.monotonic()
    with (args.output / "stdout.log").open("wb") as stdout, (args.output / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        info["pid"] = process.pid
        (args.output / "launch.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        try:
            info["exit_code"] = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"MATLAB runtime check reached its {args.timeout}-second hard timeout; stopping this check only", flush=True)
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            process.wait(timeout=15)
            info["exit_code"] = process.returncode
            info["timed_out"] = True
    info["elapsed_seconds"] = time.monotonic() - start
    info["marker_present"] = b"A2_MATLAB_RUNTIME_OK" in (args.output / "stdout.log").read_bytes()
    info["success"] = info["exit_code"] == 0 and info["marker_present"]
    info["artifacts_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()}
    (args.output / "result.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info), flush=True)
    raise SystemExit(0 if info["success"] else 1)


if __name__ == "__main__":
    main()
