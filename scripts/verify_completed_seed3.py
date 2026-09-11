"""Verify a completed seed3 checkpoint and calculate paired descriptive deltas."""
import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("F:/YOLO-Master-A2-P1")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("adaptive", "tal"), required=True)
    args = parser.parse_args()
    name = f"p1-{args.mode}-s20260826"
    source = ROOT.parent / "YOLO-Master"
    git = ["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    assert commit == "52c2befa50706b9dff13b6e0813b19413d9f532d"
    assert not subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no", "--", "ultralytics"]).strip()
    folder = ROOT / "results/closure-evaluation" / name
    manifest = read(folder / "evaluation-manifest.json")
    assert manifest["images"] == 548 and manifest["completed_epochs"] == 120
    assert manifest["protocol"]["source_commit"] == commit
    run = PROJECT / name
    assert sha(run / "weights/last.pt") == manifest["protocol"]["checkpoint_sha256"]
    assert sha(run / "args.yaml") == sha(folder / "args.yaml")
    assert sha(run / "results.csv") == sha(folder / "results.csv")
    for filename, digest in manifest["script_sha256"].items():
        assert sha(ROOT / "scripts" / filename) == digest
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert [int(r["epoch"]) for r in csv.DictReader(handle)] == list(range(1, 121))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(source))
    import torch

    torch.set_num_threads(2)
    healthy = torch.load(run / "weights/last_healthy.pt", map_location="cpu", weights_only=False)
    assert healthy["epoch"] == 119 and healthy["optimizer"] is not None
    train_args = healthy["train_args"]
    resume = read(ROOT / "configs" / f"{name}.resume.json")
    assert resume["expected_git_commit"] == commit
    assert resume["expected_epochs"] == 120 and resume["expected_stal_mode"] == args.mode
    # Seed3 recovery JSON predates the full parameter-contract field. Read the
    # frozen launcher's literal kwargs without importing/executing training.
    launcher = ROOT / "scripts/run_p1_seed.py"
    train_calls = [node for node in ast.walk(ast.parse(launcher.read_text(encoding="utf-8")))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "train"]
    assert len(train_calls) == 1
    expected = {item.arg: item.value.value for item in train_calls[0].keywords
                if isinstance(item.value, ast.Constant)}
    # exist_ok/name are output controls changed by resumptions, not model factors.
    expected.pop("exist_ok", None)
    expected.update(seed=20260826, stal_mode=args.mode, nbs=64, mixup=0.0, copy_paste=0.0)
    expected.pop("device", None)  # Recovery stores the same GPU id as a string.
    for key, value in expected.items():
        assert train_args[key] == value, (key, train_args[key], value)
    assert train_args["epochs"] == 120 and train_args["stal_mode"] == args.mode
    groups = healthy["optimizer"]["param_groups"]
    momenta = [group["momentum"] for group in groups if "momentum" in group]
    assert momenta and all(value == .9 for value in momenta)
    optimizer_lrs = [group["lr"] for group in groups]
    del healthy
    baseline = ROOT / "results/closure-evaluation/p1-fixed-s20260826"
    baseline_protocol = read(baseline / "evaluation-manifest.json")["protocol"]
    for key, value in manifest["protocol"].items():
        if key != "checkpoint_sha256":
            assert baseline_protocol[key] == value, key
    metrics, fixed = {}, {}
    for filename, keys in (
        ("det-source-metrics.json", ("AP", "AP50", "AP75", "AR500")),
        ("coco-area-metrics.json", ("APs", "APm", "APl", "ARs@500")),
    ):
        current, reference = read(folder / filename)["metrics"], read(baseline / filename)["metrics"]
        metrics.update({key: current[key] for key in keys})
        fixed.update({key: reference[key] for key in keys})
    logs = read(folder / "training-log-manifest.json")
    for record in logs["records"]:
        assert sha(ROOT / record["archived"]) == record["sha256"] == sha(Path(record["source"]))
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run": name, "completed_epochs": 120, "source_commit": commit,
        "checkpoint_sha256": manifest["protocol"]["checkpoint_sha256"],
        "healthy_checkpoint_epoch_zero_based": 119,
        "healthy_checkpoint_sha256": sha(run / "weights/last_healthy.pt"),
        "checked_training_args": expected, "actual_optimizer_momenta": momenta,
        "training_launcher_sha256": sha(launcher),
        "last_optimizer_lrs": optimizer_lrs, "images": 548,
        "evaluation_protocol_matches_fixed_except_checkpoint": True,
        "metrics": metrics, "fixed_same_seed_metrics": fixed,
        "delta_vs_fixed_pp": {key: metrics[key] - fixed[key] for key in metrics},
        "raw_training_log_files": len(logs["records"]),
        "raw_training_log_bytes": sum(record["bytes"] for record in logs["records"]),
        "missing_online_assignment_epochs": logs["missing_online_assignment_epochs"],
        "interpretation": "Paired descriptive result only; not a complete uniform-protocol 3-seed P1 decision",
        "matlab_runtime_parity": "not executed; source-level alternative evidence published separately",
        "verification_scope": "Stored checkpoint/configuration/artifact checks and arithmetic; no retraining or evaluation rerun",
    }
    output = folder / "completion-verification.json"
    if output.exists():
        raise FileExistsError("Preserve previous verification; choose a new record before rerunning")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics, "delta_vs_fixed_pp": result["delta_vs_fixed_pp"]}), flush=True)


if __name__ == "__main__":
    main()
