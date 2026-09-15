"""Run original DET MATLAB on the completed P1 v2 fixed predictions.

The P0 runtime evidence and its hash-locked runner are left unchanged. This
wrapper verifies the new predictions, original GT, and official MATLAB source
before and after execution, and preserves the actual process exit status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from run_p0_matlab_alignment import DATA, ROOT, TOOLKIT, dump, quote, sha

RUN = "p1-v2-fixed-s20260825-stats120"
LOCAL = Path("F:/YOLO-Master-A2-P1/closure-evaluation") / RUN
PUBLIC = ROOT / "results/closure-evaluation" / RUN
COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"


def preflight():
    manifest = json.loads(
        (PUBLIC / "evaluation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["completed_epochs"] == 120 and manifest["images"] == 548
    checkpoint = Path("F:/YOLO-Master-A2-P1") / RUN / "weights/last.pt"
    assert sha(checkpoint) == manifest["protocol"]["checkpoint_sha256"]
    assert sha(LOCAL / "val/predictions.json") == manifest["predictions_sha256"]
    git = ["git", "-C", str(TOOLKIT)]
    assert (
        subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        == COMMIT
    )
    original = json.loads(
        (ROOT / "results/det-source-alignment-20260911/input-sha256.json").read_text(
            encoding="utf-8"
        )
    )
    names = {row["name"] for row in original}
    assert len(original) == len(names) == 548
    assert {p.name for p in (DATA / "annotations").iterdir()} == names
    assert {p.name for p in (LOCAL / "det").glob("*.txt")} == names
    inputs = []
    for row in original:
        assert sha(DATA / "annotations" / row["name"]) == row["gt_sha256"]
        inputs.append(
            {
                "name": row["name"],
                "gt_sha256": row["gt_sha256"],
                "det_sha256": sha(LOCAL / "det" / row["name"]),
            }
        )
    sources = {}
    for name in subprocess.check_output(
        git + ["ls-files", "*.m"], text=True
    ).splitlines():
        blob = subprocess.check_output(git + ["show", f"{COMMIT}:{name}"])
        actual = (TOOLKIT / name).read_bytes()
        assert actual.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
        sources[name] = sha(TOOLKIT / name)
    return {
        "run": RUN,
        "images": 548,
        "toolkit_commit": COMMIT,
        "source_sha256": sources,
        "checkpoint_sha256": sha(checkpoint),
        "inputs": inputs,
        "prediction_json_sha256": manifest["predictions_sha256"],
        "python_metrics_sha256": sha(PUBLIC / "det-source-metrics.json"),
        "evaluation_manifest_sha256": sha(PUBLIC / "evaluation-manifest.json"),
        "input_records_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--matlab",
        type=Path,
        default=Path("E:/Program Files/MATLAB/R2024a/bin/matlab.exe"),
    )
    args = parser.parse_args()
    info = preflight()
    code = (
        f"addpath({quote(ROOT / 'scripts')}); disp(version); "
        f"run_visdrone_official_matlab({quote(TOOLKIT)},{quote(DATA)},"
        f"{quote(LOCAL / 'det')},{quote(args.output / 'matlab-metrics.json')});"
    )
    command = [str(args.matlab), "-wait", "-batch", code]
    info.update(
        {
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha(Path(__file__)),
            "wrapper_sha256": sha(ROOT / "scripts/run_visdrone_official_matlab.m"),
            "timeout_seconds": 3600,
        }
    )
    if args.check_only:
        print(json.dumps({k: v for k, v in info.items() if k != "inputs"}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (
        (args.output / "stdout.log").open("wb") as stdout,
        (args.output / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        info["pid"] = process.pid
        dump(args.output / "launch.json", info)
        try:
            info["exit_code"] = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            print(
                "MATLAB evaluation timed out; stopping only this child process tree",
                flush=True,
            )
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            info["exit_code"] = process.wait(timeout=15)
            info["timed_out"] = True
    info["elapsed_seconds"] = time.monotonic() - start
    info["matlab_executed_successfully"] = (
        info["exit_code"] == 0 and (args.output / "matlab-metrics.json").is_file()
    )
    if info["matlab_executed_successfully"]:
        actual = json.loads(
            (args.output / "matlab-metrics.json").read_text(encoding="utf-8")
        )
        reference = json.loads(
            (PUBLIC / "det-source-metrics.json").read_text(encoding="utf-8")
        )["metrics"]
        assert actual["images"] == 548
        info["comparison_pp"] = {}
        for key in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500"):
            assert math.isfinite(actual[key]) and math.isfinite(reference[key])
            info["comparison_pp"][key] = {
                "matlab": actual[key],
                "python": reference[key],
                "absolute_difference": abs(actual[key] - reference[key]),
            }
        info["within_0_01_pp"] = all(
            row["absolute_difference"] < 0.01 for row in info["comparison_pp"].values()
        )
        after = preflight()
        assert after == {key: info[key] for key in after}, (
            "Input/source changed during MATLAB execution"
        )
    info["artifacts_sha256"] = {
        p.name: sha(p) for p in args.output.iterdir() if p.is_file()
    }
    dump(args.output / "comparison.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "inputs"}), flush=True)
    raise SystemExit(0 if info.get("within_0_01_pp") else 1)


if __name__ == "__main__":
    main()
