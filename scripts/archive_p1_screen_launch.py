"""Capture the frozen launch contract and bounded verification evidence."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from p1_screen_contract import FROZEN, PROJECT, ROOT, now, save, sha, verify_frozen

OUT = ROOT / "results/p1-screen-v2-20260915-launch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-exit-code", type=int, required=True, choices=(0,))
    args = parser.parse_args()
    verify_frozen()
    OUT.mkdir(exist_ok=False)
    sources = [
        (FROZEN, "freeze-manifest.json"),
        (PROJECT / "gpu-state.json", "gpu-state-at-launch.json"),
        (PROJECT / "cpu-state.json", "cpu-state-at-launch.json"),
    ]
    smoke = Path(
        "F:/YOLO-Master-A2-P1/diagnostics/screen-v2-smoke-s1-adaptive-reference"
    )
    sources += [
        (smoke / name, "smoke-" + Path(name).name)
        for name in ("args.yaml", "results.csv", "assignment/epoch-001.json")
    ]
    snapshots = []
    for source, name in sources:
        data = source.read_bytes()
        target = OUT / name
        target.write_bytes(data)
        snapshots.append(
            {
                "source": str(source),
                "snapshot": name,
                "bytes": len(data),
                "sha256": sha(target),
            }
        )
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "pytest",
        "tests/test_p1_screen_contract.py",
        "tests/test_p1_v2_resume_contract.py",
        "tests/test_evaluation_boundaries.py",
        "tests/test_archive_raw_logs.py",
        "-q",
        "-o",
        "addopts=",
    ]
    with (
        (OUT / "tests.stdout.log").open("xb") as out,
        (OUT / "tests.stderr.log").open("xb") as err,
    ):
        test = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, timeout=180, check=False)
    save(
        OUT / "verification.json",
        {
            "observed_at": now(),
            "snapshots": snapshots,
            "test_command": command,
            "tests_exit_code": test.returncode,
            "cpu_smoke_exit_code_observed_by_caller": args.smoke_exit_code,
            "smoke_scope": "One bundled mini image, CPU,1 epoch; not a VisDrone screening result; no separate original stdout file retained",
            "state_scope": "Snapshot of queued/running workers, not completed training",
            "files_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
        },
        exclusive=True,
    )
    print(
        json.dumps({"tests_exit_code": test.returncode, "freeze_sha256": sha(FROZEN)}),
        flush=True,
    )
    raise SystemExit(test.returncode)


if __name__ == "__main__":
    main()
