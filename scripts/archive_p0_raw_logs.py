"""Archive completed P0/seed3 training logs byte-for-byte, including failures."""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from assignment_coverage import missing_assignment_epochs


ROOT = Path(__file__).resolve().parents[1]
NAME = "p0-locked-s20260824-stats120"
PROJECT = Path("F:/YOLO-Master-A2-P1")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def original_sources(project, name):
    """Inspect all required sources before copying; record absent recovery honestly."""
    sources = []
    for label, source in (
        ("initial", project / "recovery-logs/p1-seed3-chain" / name),
        ("recovery", project / "recovery-logs" / name),
    ):
        if label == "recovery" and name != NAME and not source.exists():
            sources.append((label, source, []))
            continue
        if not source.is_dir():
            raise ValueError(f"Required original log directory missing: {source}")
        files = sorted(source.glob("*.log")) + sorted(source.glob("*.json"))
        if not any(p.suffix == ".log" for p in files):
            raise ValueError(f"No original logs in {source}")
        sources.append((label, source, files))
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=NAME, choices=(NAME, "p1-fixed-s20260826", "p1-adaptive-s20260826", "p1-tal-s20260826"))
    name = parser.parse_args(argv).run
    run = PROJECT / name
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, 121)):
        raise ValueError("Require exactly 120 completed, sequential epochs")
    missing = missing_assignment_epochs(run, 120)
    if name == NAME and missing:
        raise ValueError("Online assignment evidence is incomplete")
    target = ROOT / "logs/closure-evaluation" / name / "training-original"
    records = []
    source_directories = []
    for label, source, files in original_sources(PROJECT, name):
        source_directories.append({
            "label": label,
            "source": str(source),
            "status": "present" if files else "directory_absent",
            "note": (
                "All observed .log/.json files are archived below"
                if files else
                "No recovery directory observed; no recovery logs supplied. "
                "This does not independently prove that no restart occurred."
            ),
        })
        for original in files:
            before = original.stat()
            digest = sha256(original)
            copied = target / label / original.name
            copied.parent.mkdir(parents=True, exist_ok=True)
            if copied.exists() and sha256(copied) != digest:
                raise ValueError(f"Refuse to replace different archived bytes: {copied}")
            if not copied.exists():
                shutil.copy2(original, copied)
            after = original.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed during archival: {original}")
            if sha256(copied) != digest:
                raise ValueError(f"Archive checksum mismatch: {copied}")
            records.append({
                "source": str(original),
                "archived": copied.relative_to(ROOT).as_posix(),
                "bytes": before.st_size,
                "sha256": digest,
            })
    manifest = {
        "run": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Original stdout/stderr plus recovery metadata; failures retained; no log trimming or re-encoding",
        "status": "local_archive_not_a_publication_or_acceptance_certificate",
        "completed_epochs": 120,
        "missing_online_assignment_epochs": missing,
        "source_directories": source_directories,
        "records": records,
    }
    path = ROOT / "results/closure-evaluation" / name / "training-log-manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records), "manifest": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
