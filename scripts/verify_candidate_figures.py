"""Cross-check the figure exports against raw records; never rerun inference."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/candidate-quality-20260913"
FIGURES = SOURCE / "figures-v1"
REPORT = ROOT / "docs/closure-review/研究报告.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run(*command):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", *map(str, command)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path; preserve previous verification")
    manifest = read_json(FIGURES / "figure-manifest.json")
    summary = read_json(SOURCE / "summary.json")
    assert sha(ROOT / "scripts/plot_candidate_quality.py") == manifest["script_sha256"]
    assert sha(SOURCE / "summary.json") == manifest["source_summary_sha256"]
    for name, expected in manifest["source_data_sha256"].items():
        assert sha(SOURCE / name) == expected
    for name, expected in manifest["outputs_sha256"].items():
        assert sha(FIGURES / name) == expected

    gt = [
        json.loads(line)
        for line in (SOURCE / "gt-records.jsonl").read_text().splitlines()
    ]
    lookup = {(r["mode"], r["image"], r["gt"]): r for r in gt}
    pairs = read_csv(FIGURES / "paired-small-gt.csv")
    expected_keys = {
        (r["image"], r["gt"])
        for r in gt
        if r["mode"] == "fixed" and r["size"] == "small"
    }
    assert len(pairs) == len(expected_keys) == 1321
    assert {(r["image"], int(r["gt"])) for r in pairs} == expected_keys
    for row in pairs:
        counts = {}
        for mode in ("fixed", "adaptive"):
            record = lookup[(mode, row["image"], int(row["gt"]))]
            assert record["size"] == "small"
            counts[mode] = record["counts"]
            assert int(row[f"{mode}_count"]) == record["counts"]["after_conflict"]
            assert float(row[f"{mode}_weight"]) == record["counts"]["weight_sum"]
        assert (
            int(row["delta_count"])
            == counts["adaptive"]["after_conflict"] - counts["fixed"]["after_conflict"]
        )
        assert (
            float(row["delta_weight"])
            == counts["adaptive"]["weight_sum"] - counts["fixed"]["weight_sum"]
        )
    with gzip.open(
        SOURCE / "candidate-records.csv.gz", "rt", encoding="utf-8", newline=""
    ) as stream:
        candidates = list(csv.DictReader(stream))
    selected = [
        r
        for r in candidates
        if r["mode"] == "adaptive" and r["size"] == "small" and int(r["after_conflict"])
    ]
    assert read_csv(FIGURES / "selected-small-candidates.csv") == selected
    cohorts = Counter(r["inside_fixed"] for r in selected)
    assert cohorts == {"1": 5530, "0": 2874}
    first = read_json(SOURCE / "manifest.json")["inputs"][0]["image"]
    index = min(r["gt"] for r in gt if r["image"] == first and r["size"] == "small")
    geometry = [r for r in candidates if r["image"] == first and int(r["gt"]) == index]
    assert read_csv(FIGURES / "geometry-example-candidates.csv") == geometry
    assert (first, index) == (manifest["geometry"]["image"], manifest["geometry"]["gt"])

    pdf_checks = {}
    for stem, height in (("candidate-quality", 130), ("candidate-geometry", 86)):
        pdf = FIGURES / f"{stem}.pdf"
        box = re.search(
            rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", pdf.read_bytes()
        )
        size = [float(v) * 25.4 / 72 for v in box.groups()]
        assert abs(size[0] - 183) < 1e-6 and abs(size[1] - height) < 1e-6
        assert "<text" in (FIGURES / f"{stem}.svg").read_text(encoding="utf-8")
        audit = json.loads(
            run(
                args.figure_tools / "audit_pdf_text.py",
                pdf.relative_to(ROOT),
                "--min-pt",
                "5",
                "--json",
            )
        )
        assert audit["auditable"] and audit["below_minimum_count"] == 0
        pdf_checks[stem] = {"size_mm": size, "text_audit": audit}

    # Match new prose to source values at displayed precision; existing tables
    # are checked separately by verify_closure_report.py.
    stats = summary["gt_statistics"]
    fragments = []
    for key in ("after_conflict", "weight_sum"):
        fixed = stats[f"fixed/small/{key}"]
        adaptive = stats[f"adaptive/small/{key}"]
        fragments.append(f"{100 * (adaptive['mean'] / fixed['mean'] - 1):.2f}%")
        fragments.extend(f"{100 * v['zero_fraction']:.4f}%" for v in (fixed, adaptive))
    for mode in ("adaptive", "fixed"):
        fragments.extend(
            f"{stats[f'{mode}/small/{stage}']['mean']:.4f}"
            for stage in ("before_conflict", "after_conflict")
        )
    new = [r for r in selected if r["inside_fixed"] == "0"]
    fragments.extend(
        f"{sum(float(r[key]) for r in new) / len(new):.4f}"
        for key in ("target_weight", "iou")
    )
    report = REPORT.read_text(encoding="utf-8")
    for fragment in fragments:
        assert fragment in report, fragment
    static = json.loads(
        run(
            args.figure_tools / "validate_figure.py",
            "scripts/plot_candidate_quality.py",
            "--json",
        )
    )
    assert static["summary"]["counts"]["FAIL"] == 0
    tests = run(
        "-m",
        "pytest",
        "tests/test_candidate_quality_figures.py",
        "tests/test_candidate_quality_probe.py",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
    )
    result = {
        "status": "pass",
        "scope": "Raw-to-figure CSV equality, hashes, rendered PDF text/size, prose precision and tests; not training reproducibility or acceptance",
        "verifier_sha256": sha(Path(__file__)),
        "report_sha256": sha(REPORT),
        "paired_gt_checked": len(pairs),
        "selected_candidates_checked": len(selected),
        "geometry_candidates_checked": len(geometry),
        "cohorts": dict(cohorts),
        "prose_values_checked": fragments,
        "source_preflight": static,
        "pdf_checks": pdf_checks,
        "tests_stdout": tests,
        "plot_tool_sha256": {
            name: sha(args.figure_tools / name)
            for name in ("validate_figure.py", "audit_pdf_text.py")
        },
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "pairs": len(pairs),
                "selected": len(selected),
                "geometry": len(geometry),
                "tests": tests.strip(),
            }
        )
    )


if __name__ == "__main__":
    main()
