"""Verify and collect the completed P1 v2 fixed control without changing results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from assignment_coverage import missing_assignment_epochs
from run_p1_v2_fixed import validate_train_args

ROOT = Path(__file__).resolve().parents[1]
NAME = "p1-v2-fixed-s20260825-stats120"
PROJECT = Path("F:/YOLO-Master-A2-P1")
RUN = PROJECT / NAME
OUT = ROOT / "results/closure-evaluation" / NAME


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def copy_exact(source, target):
    digest = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == digest, f"Refuse to replace different bytes: {target}"
    else:
        shutil.copy2(source, target)
    assert sha(target) == digest
    return {
        "source": str(source),
        "archived": target.relative_to(ROOT).as_posix(),
        "bytes": source.stat().st_size,
        "sha256": digest,
    }


def histogram_summary(hist):
    n = sum(hist.values())
    result = {
        "n": n,
        "mean": sum(k * v for k, v in hist.items()) / n,
        "zero_fraction": hist.get(0, 0) / n,
    }
    for label, fraction in (("p50", 0.5), ("p90", 0.9)):
        rank, cumulative = math.ceil(n * fraction), 0
        for key in sorted(hist):
            cumulative += hist[key]
            if cumulative >= rank:
                result[label] = key
                break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-exit-code", type=int, required=True, choices=(0,))
    args = parser.parse_args()
    if (OUT / "completion-verification.json").exists():
        raise FileExistsError(
            "Keep the existing completion record; do not overwrite it"
        )
    manifest = read(OUT / "evaluation-manifest.json")
    config = read(ROOT / "configs/p1-v2-fixed-s20260825.train.json")
    assert manifest["images"] == 548 and manifest["completed_epochs"] == 120
    assert manifest["protocol"]["source_commit"] == config["source_commit"]
    assert sha(RUN / "weights/last.pt") == manifest["protocol"]["checkpoint_sha256"]
    for name in ("args.yaml", "results.csv"):
        assert sha(RUN / name) == sha(OUT / name)
    for name, digest in manifest["script_sha256"].items():
        assert sha(ROOT / "scripts" / name) == digest, name
    with (RUN / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == list(range(1, 121))
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    assert not missing_assignment_epochs(RUN, 120)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT.parent / "YOLO-Master"))
    import torch

    torch.set_num_threads(2)
    healthy = torch.load(
        RUN / "weights/last_healthy.pt", map_location="cpu", weights_only=False
    )
    assert healthy["epoch"] == 119 and healthy["optimizer"] is not None
    validate_train_args(config["train"], healthy["train_args"])
    momenta = [
        g["momentum"] for g in healthy["optimizer"]["param_groups"] if "momentum" in g
    ]
    assert momenta and all(value == 0.9 for value in momenta)
    del healthy
    pools, total_batches = {}, 0
    for epoch in range(1, 121):
        source = RUN / "assignment" / f"epoch-{epoch:03d}.json"
        assert sha(source) == sha(OUT / "assignment" / source.name)
        record = read(source)
        assert record["batches"] == record["expected_batches"] == 1618
        total_batches += record["batches"]
        for branch, bins in record["branches"].items():
            for size, stages in bins.items():
                for stage, item in stages.items():
                    hist = Counter({int(k): v for k, v in item["histogram"].items()})
                    expected = histogram_summary(hist)
                    for key in ("n", "p50", "p90"):
                        assert item[key] == expected[key], (epoch, size, stage, key)
                    for key in ("mean", "zero_fraction"):
                        assert math.isclose(item[key], expected[key], abs_tol=1e-12)
                    pools.setdefault((branch, size, stage), Counter()).update(hist)
    assert total_batches == 194160
    log_manifest = read(OUT / "training-log-manifest.json")
    for item in log_manifest["records"]:
        assert (
            sha(ROOT / item["archived"]) == item["sha256"] == sha(Path(item["source"]))
        )
    extra = []
    eval_dir = PROJECT / "closure-evaluation" / NAME
    for name in ("evaluation.stdout.log", "evaluation.stderr.log", "state.json"):
        extra.append(
            copy_exact(eval_dir / name, ROOT / "logs/closure-evaluation" / NAME / name)
        )
    assert (eval_dir / "evaluation.stderr.log").stat().st_size == 0
    matlab_dir = PROJECT / "diagnostics/p1-v2-fixed-matlab-20260915"
    comparison = read(matlab_dir / "comparison.json")
    for name, digest in comparison["artifacts_sha256"].items():
        assert sha(matlab_dir / name) == digest
    for source in sorted(matlab_dir.iterdir()):
        if source.is_file() and source.suffix in (".json", ".log"):
            extra.append(copy_exact(source, OUT / "matlab-attempt" / source.name))
    pause_dir = PROJECT / "user-pauses/20260914-0947-p1-v2-fixed-s20260825-stats120"
    extra.append(
        copy_exact(pause_dir / "pause-manifest.json", OUT / "pause-manifest.json")
    )
    resumption = PROJECT / "recovery-logs" / NAME / "resumption-20260914-184913.md"
    extra.append(copy_exact(resumption, OUT / "resumption-record.md"))
    for filename in (
        *manifest["script_sha256"],
        "assignment_observer.py",
        "plot_assignment_evolution.py",
        "run_p1_v2_fixed.py",
        "run_p1_v2_matlab_alignment.py",
        "run_visdrone_official_matlab.m",
    ):
        extra.append(
            copy_exact(ROOT / "scripts" / filename, OUT / "executed-sources" / filename)
        )
    det = read(OUT / "det-source-metrics.json")["metrics"]
    area = read(OUT / "coco-area-metrics.json")["metrics"]
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run": NAME,
        "status": "training_and_python_evaluation_complete_matlab_retry_requires_user",
        "source_commit": config["source_commit"],
        "completed_epochs": 120,
        "online_assignment_epochs": 120,
        "training_batches": total_batches,
        "checkpoint_sha256": manifest["protocol"]["checkpoint_sha256"],
        "healthy_checkpoint_sha256": sha(RUN / "weights/last_healthy.pt"),
        "healthy_checkpoint_epoch_zero_based": 119,
        "checked_training_args": config["train"],
        "actual_optimizer_momenta": momenta,
        "primary": "epoch-120 last.pt",
        "validation_images": 548,
        "evaluation_exit_code_observed_by_caller": args.evaluation_exit_code,
        "training_exit_code": None,
        "training_completion_evidence": "120 sequential CSV rows; epoch119 healthy checkpoint; final validation log; runner no longer active",
        "matlab_exit_code": comparison["exit_code"],
        "matlab_executed_successfully": comparison["matlab_executed_successfully"],
        "matlab_failure": "Startup error 5201; no evaluation metrics produced for this run",
        "det_metric_source": "MATLAB-source Python port; P0 runtime parity verified separately on 2026-09-13",
        "metrics": {
            **{key: det[key] for key in ("AP", "AP50", "AP75", "AR500")},
            **{key: area[key] for key in ("APs", "APm", "APl", "ARs@500", "AP50s")},
        },
        "final_epoch_assignment": read(RUN / "assignment/epoch-120.json")["branches"],
        "pooled_assignment": [
            dict(branch=b, area_bin=g, stage=s, **histogram_summary(h))
            for (b, g, s), h in sorted(pools.items())
        ],
        "pooled_unit": "Augmented training GT observations pooled across completed epochs, not independent seeds or unique GTs",
        "resume": {
            "completed_before_pause": 64,
            "resumed_at_epoch": 65,
            "raw_csv_time_resets": [65],
            "note": "Keep raw time column unchanged; it restarts with the training process",
        },
        "package_versions": {
            name: version(name) for name in ("torch", "numpy", "faster-coco-eval")
        },
        "extra_archives": extra,
        "interpretation": "One completed fixed control with online statistics; not an adaptive improvement or P1 acceptance claim",
    }
    save_new(OUT / "completion-verification.json", result)
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "epochs": 120,
                "batches": total_batches,
                "last_epoch": {
                    size: values["after_conflict"]
                    for size, values in result["final_epoch_assignment"][
                        "topk10_topk210"
                    ].items()
                },
                "matlab_exit_code": comparison["exit_code"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
