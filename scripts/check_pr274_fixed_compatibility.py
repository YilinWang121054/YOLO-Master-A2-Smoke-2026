"""Run CPU-only PR #274 review checks without touching the active training checkout."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT.parent / "YOLO-Master-A2-final-review"
BASE = ROOT.parent / "YOLO-Master-ci-base-7bfbfd3"
TREE = "f1ae8461c9108b89d41fee6df4fd95023ab031c1"
BASE_COMMIT = "7bfbfd374a6b44720f98366e088f3a962e16321d"
OUTPUT = ROOT / "results/pr274-review-checks-20260914"


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
        encoding="utf-8",
    ).strip()


def worker(output):
    # Keep pytest's cleanup hook confined to disposable, initially empty paths.
    work = output / "isolated-work"
    work.mkdir()
    os.chdir(work)
    sys.path.insert(0, str(HEAD))
    import torch
    import yaml
    from ultralytics import utils

    torch.set_num_threads(2)
    utils.WEIGHTS_DIR = work / "weights"
    utils.WEIGHTS_DIR.mkdir()
    assert not torch.cuda.is_available(), (
        "Review checks must not occupy the training GPU"
    )
    assert Path(utils.__file__).resolve().is_relative_to(HEAD)
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.tal import TaskAlignedAssigner

    print(f"Imported source: {utils.__file__}", flush=True)
    print(f"Python {sys.version}; torch {torch.__version__}; device CPU", flush=True)
    base_yaml = yaml.safe_load(
        (BASE / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8")
    )
    head_yaml = yaml.safe_load(
        (HEAD / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8")
    )
    added = sorted(set(head_yaml) - set(base_yaml))
    assert added and all(key.startswith("stal_") for key in added)
    assert {key: head_yaml[key] for key in base_yaml} == base_yaml
    assert get_cfg().stal_mode == "fixed"

    # The reference TAL module comes from the exact PR base. Its imported helpers
    # are byte-identical Git blobs across these two versions (checked by main).
    spec = importlib.util.spec_from_file_location(
        "ultralytics.utils._pr274_base_tal", BASE / "ultralytics/utils/tal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    axis = torch.arange(16, dtype=torch.float32) * 8 + 4
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((xx, yy), -1).reshape(-1, 2)
    for seed in range(8):
        for case in ("random", "dense_overlap", "tiny_boundary", "empty"):
            torch.manual_seed(seed)
            scores = torch.rand(2, len(points), 3)
            half = 1 + torch.rand(2, len(points), 2) * 28
            predicted = torch.cat((points - half, points + half), -1)
            center = torch.rand(2, 9, 2) * 128
            extent = 0.05 + torch.rand(2, 9, 2) * 60
            if case == "dense_overlap":
                center = 64 + torch.rand(2, 9, 2) * 8
                extent = 12 + torch.rand(2, 9, 2) * 40
            elif case == "tiny_boundary":
                extent = torch.tensor([0.01, 1, 15.999, 16, 16.001, 31.999, 32, 96, 97])
                extent = extent.view(1, 9, 1).expand(2, 9, 2)
            boxes = torch.cat((center - extent / 2, center + extent / 2), -1)
            labels = torch.randint(0, 3, (2, 9, 1)).float()
            valid = torch.ones(2, 9, 1, dtype=torch.bool)
            valid[1, -2:] = False
            if case == "empty":
                boxes, labels, valid = boxes[:, :0], labels[:, :0], valid[:, :0]
            inputs = (scores, predicted, points, labels, boxes, valid)
            for topk, topk2 in ((10, 10), (13, 5), (1, 1)):
                params = {
                    "topk": topk,
                    "topk2": topk2,
                    "num_classes": 3,
                    "alpha": 0.5,
                    "beta": 6,
                    "stride": [8, 16, 32],
                }
                reference = module.TaskAlignedAssigner(**params)
                expected = reference(*(value.clone() for value in inputs))
                for mode in ("implicit_default", "explicit_fixed"):
                    options = (
                        {} if mode == "implicit_default" else {"stal_mode": "fixed"}
                    )
                    assigner = TaskAlignedAssigner(**params, **options)
                    actual = assigner(*(value.clone() for value in inputs))
                    assert len(actual) == len(expected) == 5
                    for left, right in zip(actual, expected):
                        assert left.dtype == right.dtype and torch.equal(left, right), (
                            seed,
                            case,
                            topk,
                            mode,
                        )
                        assert torch.isfinite(left).all()
                    if case != "empty":
                        a = assigner.select_candidates_in_gts(
                            points, boxes.clone(), valid
                        )
                        b = reference.select_candidates_in_gts(
                            points, boxes.clone(), valid
                        )
                        assert torch.equal(a, b), (seed, case, "candidate geometry")
                    checks.append(
                        {
                            "seed": seed,
                            "case": case,
                            "topk": topk,
                            "topk2": topk2,
                            "mode": mode,
                        }
                    )
    print(
        f"Exact CPU assignment comparisons passed: {len(checks)} x 5 output tensors",
        flush=True,
    )
    import pytest

    tests = [
        "test_default_config_integrity.py",
        "test_stal_assignment.py",
        "test_tal_conflict_resolution.py",
    ]
    result = pytest.main(
        [
            *(str(HEAD / "tests" / name) for name in tests),
            "-v",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            f"--junitxml={output / 'pytest.xml'}",
            f"--basetemp={work / 'pytest-tmp'}",
        ]
    )
    summary = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if result == 0 else "fail",
        "device": "CPU FP32",
        "cuda_visible": False,
        "pytest_exit_code": int(result),
        "test_files": tests,
        "unchanged_existing_default_keys": len(base_yaml),
        "new_keys": added,
        "default_mode": get_cfg().stal_mode,
        "exact_assignment_comparisons": len(checks),
        "exact_output_tensors": len(checks) * 5,
        "cases": checks,
        "scope": "Deterministic synthetic CPU regression against exact PR base; not proof for all inputs, CUDA/AMP or training outcomes.",
        "reference_import_helpers": "Same Git blobs for metrics.py, ops.py and torch_utils.py in base and head",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    raise SystemExit(int(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    assert not OUTPUT.exists(), "Keep previous evidence immutable"
    assert git(HEAD, "rev-parse", "HEAD^{tree}") == TREE
    assert git(BASE, "rev-parse", "HEAD") == BASE_COMMIT
    for repo in (HEAD, BASE):
        assert not git(repo, "status", "--porcelain", "--untracked-files=no"), (
            "Modified tracked source"
        )
    for name in ("metrics.py", "ops.py", "torch_utils.py"):
        path = "ultralytics/utils/" + name
        assert git(HEAD, "rev-parse", f"HEAD:{path}") == git(
            BASE, "rev-parse", f"HEAD:{path}"
        )
    OUTPUT.mkdir()
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "--worker",
        str(OUTPUT),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "YOLO_CONFIG_DIR": str(OUTPUT / "isolated-config"),
        "PYTHONUTF8": "1",
    }
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "public_pr_head": "2efc4d91d7363d65b6f38e7d47c585231bb98158",
        "public_pr_tree": TREE,
        "tested_local_head": git(HEAD, "rev-parse", "HEAD"),
        "base": BASE_COMMIT,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cpu_threads": 2,
        "pytest_cleanup_isolated": True,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (
        (OUTPUT / "stdout.log").open("wb") as stdout,
        (OUTPUT / "stderr.log").open("wb") as stderr,
    ):
        result = subprocess.run(
            command, env=env, stdout=stdout, stderr=stderr, timeout=300, check=False
        )
    print(json.dumps({"output": str(OUTPUT), "exit_code": result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
