"""Rebuild the local closure audit from immutable Git objects and raw artifacts."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "YOLO-Master"
BASE = "acce839c7e895d6b179de7f7093fa879e237cc7b"
PARENT = "0996b7da14dfaafae9d4488e960814ff19eb19ce"
FROZEN = "52c2befa50706b9dff13b6e0813b19413d9f532d"
KEYS = (
    "model",
    "epochs",
    "imgsz",
    "batch",
    "nbs",
    "seed",
    "pretrained",
    "optimizer",
    "lr0",
    "momentum",
    "weight_decay",
    "warmup_bias_lr",
    "amp",
    "patience",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "max_det",
    "stal_mode",
    "data",
)


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={SOURCE.as_posix()}", "-C", str(SOURCE), *args],
        text=True,
    ).strip()


def main():
    paths = (
        "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
        "ultralytics/engine/trainer.py",
        "ultralytics/utils/tal.py",
        "ultralytics/utils/loss.py",
        "ultralytics/cfg/default.yaml",
    )
    blobs = {
        path: {ref: git("rev-parse", f"{ref}:{path}") for ref in (BASE, PARENT, FROZEN)}
        for path in paths
    }
    records = []
    for seed in (20260824, 20260825, 20260826):
        for mode in ("fixed", "adaptive", "tal"):
            run = Path("F:/YOLO-Master-A2-P1") / f"p1-{mode}-s{seed}"
            if not (run / "args.yaml").exists():
                records.append(
                    {"run": run.name, "status": "not_started", "completed_epochs": 0}
                )
                continue
            args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
            csv_path = run / "results.csv"
            rows = (
                list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
                if csv_path.exists()
                else []
            )
            stats = list((run / "assignment").glob("epoch-???.json"))
            records.append(
                {
                    "run": run.name,
                    "completed_epochs": len(rows),
                    "last_epoch": int(rows[-1]["epoch"]) if rows else 0,
                    "last_session_training_seconds": float(rows[-1]["time"])
                    if rows
                    else None,
                    "args": {k: args.get(k) for k in KEYS},
                    "assignment_epoch_files": len(stats),
                    "results_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()
                    if csv_path.exists()
                    else None,
                }
            )
    area = {
        "small": 0,
        "medium": 0,
        "large": 0,
        "at_1024": 0,
        "at_9216": 0,
        "valid_class_score_zero": 0,
    }
    anns = Path("F:/datasets/VisDrone/VisDrone2019-DET-val/annotations")
    for path in sorted(anns.glob("*.txt")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            fields = list(map(float, line.rstrip(",").split(",")))
            if not 1 <= fields[5] <= 10:
                continue
            if fields[4] == 0:
                area["valid_class_score_zero"] += 1
                continue
            value = fields[2] * fields[3]
            area[
                "small" if value < 1024 else "medium" if value < 9216 else "large"
            ] += 1
            area["at_1024"] += int(value == 1024)
            area["at_9216"] += int(value == 9216)
    documents = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.parent.glob("*.docx")
        if "任务书" in p.name or "补充细则" in p.name
    }
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "numpy",
                    "faster-coco-eval",
                    "PyYAML",
                    "pytest",
                    "ruff",
                )
            },
        },
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "base": BASE,
        "training_commit": FROZEN,
        "source_head": git("rev-parse", "HEAD"),
        "source_dirty": git("status", "--short"),
        "blobs": blobs,
        "baseline_is_ancestor": git("merge-base", BASE, FROZEN) == BASE,
        "student_commit_files": git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", FROZEN
        ).splitlines(),
        "upstream_intervening_commits": git("rev-list", "--count", f"{BASE}..{PARENT}"),
        "runs": records,
        "original_val_area_distribution": area,
        "instruction_document_sha256": documents,
        "review_document_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "docs/closure-review").glob("*.md"))
        },
    }
    out = ROOT / "results/closure-audit-20260909.json"
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "area": area,
                "runs": [
                    {k: r[k] for k in ("run", "completed_epochs")} for r in records
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
