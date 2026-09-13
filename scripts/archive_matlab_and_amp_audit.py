"""Verify and archive immutable local P0 MATLAB and P1 prerequisite outputs."""

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path("F:/YOLO-Master-A2-P1/diagnostics")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    names = ("p0-matlab-official-20260913", "p1-v2-amp-audit-20260913")
    for name in names:
        if (ROOT / "results" / name).exists():
            raise FileExistsError("Archive already exists; no overwrites")
    p0 = LOCAL / names[0]
    comparison = read(p0 / "comparison.json")
    matlab = read(p0 / "matlab-metrics.json")
    python = read(ROOT / "results/closure-evaluation/p0-locked-s20260824-stats120/det-source-metrics.json")["metrics"]
    assert comparison["exit_code"] == 0 and comparison["matlab_executed_successfully"]
    assert comparison["within_0_01_pp"] and matlab["images"] == 548
    for name, expected in comparison["artifacts_sha256"].items():
        assert sha(p0 / name) == expected
    for key, row in comparison["comparison_pp"].items():
        assert row["matlab"] == matlab[key] and row["python"] == python[key]
        assert row["absolute_difference"] == abs(matlab[key] - python[key]) < 0.01
    assert b"(R2024a) Update 7" in (p0 / "stdout.log").read_bytes()
    assert (p0 / "stderr.log").stat().st_size == 0
    amp = LOCAL / names[1]
    summary = read(amp / "summary.json")
    for name, expected in summary["artifact_sha256"].items():
        assert sha(amp / name) == expected
    assert len(summary["comparisons"]) == 6
    for row in summary["comparisons"]:
        stem = f"m{int(row['mosaic'])}-{row['mode']}"
        with np.load(amp / f"{stem}-fp32-masks.npz") as fp32, np.load(amp / f"{stem}-amp-masks.npz") as mixed:
            for values in (fp32, mixed):
                assert not (values["before"] & ~values["candidates"]).any()
                assert not (values["after"] & ~values["before"]).any()
                assert (values["after"].sum(1) <= 1).all()
                assert np.array_equal(values["after"].any(1), values["fg"])
            for key, field in (("fg", "foreground_mask_xor"), ("before", "pre_conflict_mask_xor"),
                               ("after", "post_conflict_mask_xor"), ("candidates", "candidate_mask_xor")):
                assert int(np.logical_xor(fp32[key], mixed[key]).sum()) == row[field]
            assert int(fp32["fg"].sum()) == row["fp32_positives"]
            assert int(mixed["fg"].sum()) == row["amp_positives"]
            assert int(((fp32["idx"] != mixed["idx"]) & fp32["fg"] & mixed["fg"]).sum()) == row["gt_identity_changes_on_common_positives"]
        assert math.isfinite(row["fp32_loss"]) and math.isfinite(row["amp_loss"])
    for name in names:
        source, target = LOCAL / name, ROOT / "results" / name
        target.mkdir(parents=True, exist_ok=False)
        paths = sorted(p for p in source.iterdir() if p.suffix in (".json", ".log", ".npz"))
        for path in paths:
            shutil.copy2(path, target / path.name)
            assert sha(path) == sha(target / path.name)
        verification = {"status": "pass", "verified_at": datetime.now(timezone.utc).isoformat(),
                        "copied_sha256": {p.name: sha(p) for p in paths},
                        "scope": "P0 actual runtime logs and seven metric comparisons; P1 six paired mask/count archives. Gradient finiteness checked during execution, gradients not independently replayed.",
                        "local_only_exclusions": ["*.pt augmented image tensors", "MathWorks account/ServiceHost logs"]}
        (target / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"p0_matlab_alignment": "pass", "amp_mask_pairs_verified": 6, "archives": list(names)}), flush=True)


if __name__ == "__main__":
    main()
