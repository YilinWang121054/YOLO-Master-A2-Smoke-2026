"""Preserve the successful retry separately from the earlier MATLAB5201 record."""

import json
import shutil
from pathlib import Path

from p1_screen_contract import ROOT, now, read, save, sha

SOURCE = Path("F:/YOLO-Master-A2-P1/diagnostics/p1-v2-fixed-matlab-20260915-ready")
OUT = ROOT / "results/p1-v2-fixed-matlab-official-20260915"


def main():
    record = read(SOURCE / "comparison.json")
    assert (
        record["exit_code"] == 0
        and record["matlab_executed_successfully"]
        and record["within_0_01_pp"]
    )
    assert record["images"] == 548
    assert all(
        row["absolute_difference"] < 0.01 for row in record["comparison_pp"].values()
    )
    for name, digest in record["artifacts_sha256"].items():
        assert sha(SOURCE / name) == digest
    assert (
        sha(ROOT / "scripts/run_p1_v2_matlab_alignment.py") == record["runner_sha256"]
    )
    assert (
        sha(ROOT / "scripts/run_visdrone_official_matlab.m") == record["wrapper_sha256"]
    )
    OUT.mkdir(exist_ok=False)
    sources = [
        *sorted(SOURCE.iterdir()),
        ROOT / "scripts/run_p1_v2_matlab_alignment.py",
        ROOT / "scripts/run_visdrone_official_matlab.m",
    ]
    index = {}
    for source in sources:
        if source.is_file():
            target = OUT / source.name
            shutil.copy2(source, target)
            assert sha(target) == sha(source)
            index[target.name] = {"sha256": sha(target), "bytes": target.stat().st_size}
    save(
        OUT / "archive-verification.json",
        {
            "archived_at": now(),
            "files": index,
            "status": "original_MATLAB_runtime_parity_passed_for_this_fixed_run",
            "earlier_failed_attempt_preserved": True,
        },
        exclusive=True,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "files": len(index),
                "max_difference_pp": max(
                    row["absolute_difference"]
                    for row in record["comparison_pp"].values()
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
