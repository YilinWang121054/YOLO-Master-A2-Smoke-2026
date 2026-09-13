"""Run original DET MATLAB on hash-locked P0 inputs, then compare actual metrics."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLKIT = Path("F:/tools/VisDrone2018-DET-toolkit")
DATA = Path("F:/datasets/VisDrone/VisDrone2019-DET-val")
DET = Path("F:/YOLO-Master-A2-P1/closure-evaluation/p0-locked-s20260824-stats120/det")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quote(path):
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def preflight():
    audit = json.loads((ROOT / "results/det-source-alignment-20260911/audit.json").read_text(encoding="utf-8"))
    expected_commit = "005445782213e20cb91bc50a597db3dd949e749a"
    assert subprocess.check_output(["git", "-C", str(TOOLKIT), "rev-parse", "HEAD"], text=True).strip() == expected_commit
    inputs = json.loads((ROOT / "results/det-source-alignment-20260911/input-sha256.json").read_text(encoding="utf-8"))
    assert len(inputs) == 548
    names = {row["name"] for row in inputs}
    assert {p.name for p in (DATA / "annotations").iterdir()} == names
    assert {p.name for p in DET.glob('*.txt')} == names
    for row in inputs:
        assert sha(DATA / "annotations" / row["name"]) == row["gt_sha256"]
        assert sha(DET / row["name"]) == row["det_sha256"]
    # Verify all tracked MATLAB source directly against its immutable Git blob.
    files = subprocess.check_output(["git", "-C", str(TOOLKIT), "ls-files", "*.m"], text=True).splitlines()
    hashes = {}
    for name in files:
        blob = subprocess.check_output(["git", "-C", str(TOOLKIT), "show", f"{expected_commit}:{name}"])
        local = (TOOLKIT / name).read_bytes()
        assert local.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
        hashes[name] = sha(TOOLKIT / name)
    return {"toolkit_commit": expected_commit, "source_sha256": hashes,
            "input_manifest_sha256": sha(ROOT / "results/det-source-alignment-20260911/input-sha256.json"),
            "images": 548, "checkpoint_sha256": audit["checkpoint_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", type=Path, default=Path("E:/Program Files/MATLAB/R2024a/bin/matlab.exe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    info = preflight()
    code = (f"addpath({quote(ROOT / 'scripts')}); disp(version); "
            f"run_visdrone_official_matlab({quote(TOOLKIT)},{quote(DATA)},{quote(DET)},{quote(args.output / 'matlab-metrics.json')});")
    command = [str(args.matlab), "-wait", "-batch", code]
    info.update({"command": command, "started_at": datetime.now(timezone.utc).isoformat(),
                 "wrapper_sha256": sha(ROOT / "scripts/run_visdrone_official_matlab.m"),
                 "runner_sha256": sha(Path(__file__)), "timeout_seconds": 3600})
    if args.check_only:
        print(json.dumps(info, ensure_ascii=True), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (args.output / "stdout.log").open("wb") as stdout, (args.output / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        info["pid"] = process.pid
        dump(args.output / "launch.json", info)
        try:
            info["exit_code"] = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            print("Official MATLAB evaluation reached its one-hour hard timeout; stopping this child process only", flush=True)
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            info["exit_code"] = process.wait(timeout=15)
            info["timed_out"] = True
    info["elapsed_seconds"] = time.monotonic() - start
    info["matlab_executed_successfully"] = info["exit_code"] == 0 and (args.output / "matlab-metrics.json").is_file()
    if info["matlab_executed_successfully"]:
        actual = json.loads((args.output / "matlab-metrics.json").read_text(encoding="utf-8"))
        reference = json.loads((ROOT / "results/closure-evaluation/p0-locked-s20260824-stats120/det-source-metrics.json").read_text(encoding="utf-8"))["metrics"]
        assert actual["images"] == 548
        info["comparison_pp"] = {}
        for key in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500"):
            assert math.isfinite(actual[key])
            info["comparison_pp"][key] = {"matlab": actual[key], "python": reference[key],
                                          "absolute_difference": abs(actual[key] - reference[key])}
        info["within_0_01_pp"] = all(row["absolute_difference"] < 0.01 for row in info["comparison_pp"].values())
        after = preflight()
        assert after == {key: info[key] for key in after}, "Input/source changed during MATLAB execution"
    info["artifacts_sha256"] = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    dump(args.output / "comparison.json", info)
    print(json.dumps(info, ensure_ascii=True), flush=True)
    raise SystemExit(0 if info.get("within_0_01_pp") else 1)


if __name__ == "__main__":
    main()
