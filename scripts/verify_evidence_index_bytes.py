"""Verify that Git will publish byte-exact P0 logs, statistics, and report."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/closure-evaluation/p0-locked-s20260824-stats120"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    index = {}
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    for entry in raw.decode("utf-8").split("\0"):
        if entry:
            metadata, name = entry.split("\t", 1)
            mode, digest, stage = metadata.split()
            if stage != "0":
                raise ValueError("Unresolved Git merge entry")
            index[name] = digest
    expected = [(ROOT / row["archived"], row["sha256"]) for row in read(P0 / "training-log-manifest.json")["records"]]
    expected += [(P0 / "assignment" / name, digest) for name, digest in read(P0 / "figures/figure-manifest.json")["source_files_sha256"].items()]
    expected.append((ROOT / "docs/closure-review/研究报告.md", read(ROOT / "results/report-table-verification-20260911.json")["report_sha256"]))
    expected += [(ROOT / "scripts" / name, digest) for name, digest in read(P0 / "evaluation-manifest.json")["script_sha256"].items()]
    for path, expected_sha256 in expected:
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError(f"Working file hash differs from recorded evidence: {path.relative_to(ROOT)}")
        git_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if index.get(path.relative_to(ROOT).as_posix()) != git_blob:
            raise ValueError(f"Git index would change evidence bytes: {path.relative_to(ROOT)}")
    print(json.dumps({"verified_files": len(expected), "status": "pass", "scope": "Recorded SHA256 and Git index preserve exact bytes; not official evaluation parity"}))


if __name__ == "__main__":
    main()
