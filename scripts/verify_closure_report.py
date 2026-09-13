"""Check the report's metric tables against immutable local result files."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/closure-review/研究报告.md"
KEYS = ("AP", "AP50", "AP75", "AR500", "APs", "APm", "APl", "ARs@500")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def values(det, area):
    return [f"{(det if index < 4 else area)[key]:.4f}" for index, key in enumerate(KEYS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/report-table-verification-20260911.json")
    args = parser.parse_args()
    text = REPORT.read_text(encoding="utf-8")
    checked = []
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 10 and re.fullmatch(r"2026082[456]", cells[0]):
            mode = {"fixed": "fixed", "adaptive": "adaptive", "pure TAL": "tal", "pure TAL*": "tal"}[cells[1]]
            run = f"p1-{mode}-s{cells[0]}"
            if cells[0] == "20260826":
                folder = ROOT / "results/closure-evaluation" / run
                det = read(folder / "det-source-metrics.json")["metrics"]
                area = read(folder / "coco-area-metrics.json")["metrics"]
            else:
                det = read(ROOT / f"results/official-det-strict-audit-{run}.json")["metrics"]
                area = read(ROOT / f"results/area-audit-20260910/{run}.json")["metrics"]
            if cells[2:] != values(det, area):
                raise ValueError(f"Metric table mismatch for {run}")
            checked.append(run)
    expected = {f"p1-{mode}-s{seed}" for seed in (20260824, 20260825) for mode in ("fixed", "adaptive", "tal")}
    expected.add("p1-fixed-s20260826")
    expected.add("p1-adaptive-s20260826")
    expected.add("p1-tal-s20260826")
    if set(checked) != expected or len(checked) != len(expected):
        raise ValueError("Report comparison table omits or duplicates a completed run")
    p0 = ROOT / "results/closure-evaluation/p0-locked-s20260824-stats120"
    p0_row = "| " + " | ".join(values(read(p0 / "det-source-metrics.json")["metrics"], read(p0 / "coco-area-metrics.json")["metrics"])) + " |"
    if p0_row not in text:
        raise ValueError("P0 metric row differs from evaluation files")
    assignment = read(p0 / "assignment/epoch-120.json")["branches"]["topk10_topk210"]
    for size in ("small", "medium", "large"):
        item = assignment[size]["after_conflict"]
        row = f"| {size} | {item['n']} | {item['mean']:.4f} | {item['p50']} | {item['p90']} | {100 * item['zero_fraction']:.4f}% |"
        if row not in text:
            raise ValueError(f"Final-epoch assignment row differs for {size}")
    links = re.findall(r"\]\(([^)]+)\)", text)
    for link in links:
        if not link.startswith(("https://", "http://", "#")) and not (REPORT.parent / link).exists():
            raise ValueError(f"Broken local report link: {link}")
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "comparison_rows": len(checked), "checked_numeric_cells": len(checked) * 8 + 8 + 15,
        "scope": "Table-to-JSON equality at displayed precision and local link existence; not evaluator runtime parity or acceptance",
        "status": "pass", "checked_runs": checked,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
