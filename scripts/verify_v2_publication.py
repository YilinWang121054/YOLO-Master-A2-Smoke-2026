"""Check the completed fixed bundle and freeze its public file hashes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "p1-v2-fixed-s20260825-stats120"
OUT = ROOT / "results/closure-evaluation" / NAME
INDEX = OUT / "publication-verification.json"
EXTRA = [
    "README.md",
    "docs/closure-review/研究报告.md",
    "docs/closure-review/P1第二轮fixed完成-20260915.md",
    "configs/p1-v2-fixed-s20260825.train.json",
    "configs/p1-v2-fixed-s20260825.resume.json",
    "results/report-table-verification-20260915.json",
    "scripts/archive_p0_raw_logs.py",
    "scripts/evaluate_completed_run.py",
    "scripts/package_p0_predictions.py",
    "scripts/plot_assignment_evolution.py",
    "scripts/run_p1_v2_matlab_alignment.py",
    "scripts/verify_completed_v2.py",
    "scripts/verify_closure_report.py",
    "scripts/verify_v2_publication.py",
    "tests/test_archive_raw_logs.py",
    "tests/test_verify_completed_v2.py",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_bundle():
    completion = read(OUT / "completion-verification.json")
    assert (
        completion["completed_epochs"] == completion["online_assignment_epochs"] == 120
    )
    assert completion["training_batches"] == 194160
    assert completion["evaluation_exit_code_observed_by_caller"] == 0
    assert not completion["matlab_executed_successfully"]
    for record in (
        completion["extra_archives"]
        + read(OUT / "training-log-manifest.json")["records"]
    ):
        assert sha(ROOT / record["archived"]) == record["sha256"], record["archived"]
    evaluation = read(OUT / "evaluation-manifest.json")
    for name, digest in evaluation["script_sha256"].items():
        assert sha(OUT / "executed-sources" / name) == digest
    figure = read(OUT / "figures/figure-manifest.json")
    assert len(figure["source_files_sha256"]) == 120
    for name, digest in figure["source_files_sha256"].items():
        assert sha(OUT / "assignment" / name) == digest
    archive = read(OUT / "prediction-archive-manifest.json")
    assert sha(OUT / "predictions-det-548.zip") == archive["det_zip_sha256"]
    assert sha(OUT / "predictions.json.gz") == archive["predictions_gzip_sha256"]
    with gzip.open(OUT / "predictions.json.gz", "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert (
        digest
        == archive["predictions_uncompressed_sha256"]
        == evaluation["predictions_sha256"]
    )
    with zipfile.ZipFile(OUT / "predictions-det-548.zip") as bundle:
        expected = archive["det_files_sha256"]
        assert len(bundle.namelist()) == len(expected) == 548
        assert set(bundle.namelist()) == set(expected)
        for name, digest in expected.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
    report = read(ROOT / "results/report-table-verification-20260915.json")
    assert report["status"] == "pass" and report["checked_numeric_cells"] == 126
    assert sha(ROOT / EXTRA[1]) == report["report_sha256"]
    assert sha(ROOT / EXTRA[2]) == report["completed_v2"]["note_sha256"]


def selected_files():
    files = {ROOT / name for name in EXTRA}
    for folder in (OUT, ROOT / "logs/closure-evaluation" / NAME):
        files.update(path for path in folder.rglob("*") if path.is_file())
    files.discard(INDEX)
    assert all(path.is_file() for path in files)
    assert not any(
        path.suffix.lower() in {".pt", ".pth", ".jpg", ".jpeg", ".lic"}
        for path in files
    )
    return sorted(files)


def check_staged(files):
    output = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    index = {}
    for row in output.split(b"\0"):
        if not row:
            continue
        meta, name = row.split(b"\t", 1)
        _mode, digest, stage = meta.split()
        assert stage == b"0"
        index[name.decode("utf-8")] = digest.decode("ascii")
    for path in files:
        data = path.read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        assert index.get(relative) == blob, (
            f"Staged bytes differ or missing: {relative}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-index", action="store_true")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    check_bundle()
    files = selected_files()
    records = {
        path.relative_to(ROOT).as_posix(): {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    }
    if args.write_index:
        result = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "run": NAME,
            "status": "pass",
            "files": records,
            "scope": "Bundle hashes, gzip/548 DET roundtrip, 120 statistics hashes, displayed report values; not new MATLAB success or P1 acceptance",
        }
        with INDEX.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
    else:
        assert read(INDEX)["files"] == records, "Publication file set or bytes changed"
    if args.staged:
        check_staged([*files, INDEX])
    print(
        json.dumps(
            {
                "status": "pass",
                "files": len(files),
                "bytes": sum(item["bytes"] for item in records.values()),
                "staged_checked": args.staged,
            }
        )
    )


if __name__ == "__main__":
    main()
