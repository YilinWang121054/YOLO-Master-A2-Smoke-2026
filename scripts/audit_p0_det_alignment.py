"""Archive source provenance, input checks, edge tests and a P0 Python rerun.

This does not execute MATLAB and must never be called MATLAB runtime parity.
No training or original result files are modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = "p0-locked-s20260824-stats120"
SOURCE_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
METRICS = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, default=Path("F:/tools/VisDrone2018-DET-toolkit"))
    parser.add_argument("--dataset", type=Path, default=Path("F:/datasets/VisDrone/VisDrone2019-DET-val"))
    parser.add_argument("--run-dir", type=Path, default=Path("F:/YOLO-Master-A2-P1") / RUN)
    parser.add_argument("--det-dir", type=Path, default=Path("F:/YOLO-Master-A2-P1/closure-evaluation") / RUN / "det")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit.json").exists():
        raise FileExistsError("Use a new output directory; preserve previous audit evidence")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={args.toolkit.resolve().as_posix()}", "-C", str(args.toolkit), *arguments],
            timeout=30,
        )

    assert git("rev-parse", "HEAD").decode().strip() == SOURCE_COMMIT
    assert not git("status", "--porcelain", "--untracked-files=no").strip()
    sources = {}
    for path in [args.toolkit / "evalDET.m", *sorted((args.toolkit / "utils").glob("*.m"))]:
        relative = path.relative_to(args.toolkit).as_posix()
        committed = git("show", f"{SOURCE_COMMIT}:{relative}")
        # Git checkout may use CRLF; record both byte hashes, verify logical text.
        assert path.read_bytes().replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")
        sources[relative] = {
            "git_blob": git("rev-parse", f"{SOURCE_COMMIT}:{relative}").decode().strip(),
            "git_content_sha256": hashlib.sha256(committed).hexdigest(),
            "local_sha256": sha(path),
            "lines": len(committed.splitlines()),
            "url": f"https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/{SOURCE_COMMIT}/{relative}",
        }
    script = ROOT / "scripts/evaluate_visdrone_det_strict.py"
    functions = {
        node.name: [node.lineno, node.end_lineno]
        for node in ast.parse(script.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    original = ROOT / "results/closure-evaluation" / RUN
    manifest = json.loads((original / "evaluation-manifest.json").read_text(encoding="utf-8"))
    assert sha(script) == manifest["script_sha256"][script.name]
    assert sha(args.run_dir / "weights/last.pt") == manifest["protocol"]["checkpoint_sha256"]
    assert sha(args.run_dir / "args.yaml") == sha(original / "args.yaml")
    config = yaml.safe_load((original / "args.yaml").read_text(encoding="utf-8"))
    assert config["epochs"] == 120 and config["pretrained"] is False
    assert config["optimizer"] == "MuSGD" and config["momentum"] == .9
    annotations = sorted((args.dataset / "annotations").glob("*.txt"))
    assert len(annotations) == 548
    assert len(list((args.dataset / "annotations").iterdir())) == 548
    assert {p.name for p in annotations} == {p.name for p in args.det_dir.glob("*.txt")}
    inputs = []
    total_gt, total_dt, half_ties, near_half_non_ties = 0, 0, 0, 0
    zero_extent_rows = 0
    for gt_path in annotations:
        assert gt_path.name.isascii()
        dt_path = args.det_dir / gt_path.name
        gt = np.loadtxt(gt_path, delimiter=",", ndmin=2)
        dt = np.loadtxt(dt_path, delimiter=",", ndmin=2) if dt_path.stat().st_size else np.zeros((0, 8))
        assert gt.shape[1] == dt.shape[1] == 8
        assert np.isfinite(gt).all() and np.isfinite(dt).all()
        assert np.equal(gt, np.trunc(gt)).all() and np.max(np.abs(gt)) < 2**31 - 1
        assert set(gt[:, 4]).issubset({0, 1})
        assert set(gt[:, 5]).issubset(set(range(12)))
        assert set(dt[:, 5]).issubset(set(range(1, 11)))
        assert (dt[:, 2:4] >= 0).all()
        zero_extent_rows += int(np.count_nonzero((dt[:, 2:4] == 0).any(axis=1)))
        assert np.max(np.abs(dt[:, :4]), initial=0) < 2**31 - 1
        fraction = dt[:, :4] - np.floor(dt[:, :4])
        distance = np.abs(fraction - .5)
        half_ties += int(np.count_nonzero(distance == 0))
        near_half_non_ties += int(np.count_nonzero((distance > 0) & (distance < 1e-10)))
        total_gt += len(gt)
        total_dt += len(dt)
        inputs.append({"name": gt_path.name, "gt_sha256": sha(gt_path), "det_sha256": sha(dt_path)})
    save(out / "input-sha256.json", inputs)

    commands = []
    def run_command(command, log_name, timeout):
        print(f"START {log_name}", flush=True)
        started = time.monotonic()
        with (out / log_name).open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        commands.append({"argv": command, "cwd": str(ROOT), "exit_code": result.returncode,
                         "duration_seconds": time.monotonic() - started, "log": log_name})
        if result.returncode:
            raise RuntimeError(f"{log_name} failed: {result.returncode}")
        print(f"DONE {log_name}", flush=True)

    run_command([sys.executable, "-m", "pytest", "tests/test_det_source_audit.py", "-q",
                 f"--junitxml={out / 'tests.xml'}"], "edge-tests.log", 120)
    rerun_path = out / "p0-python-rerun.json"
    run_command([sys.executable, str(script), "--dataset-dir", str(args.dataset),
                 "--results-dir", str(args.det_dir), "--output", str(rerun_path)], "p0-python-rerun.log", 3600)
    previous = json.loads((original / "det-source-metrics.json").read_text(encoding="utf-8"))
    rerun = json.loads(rerun_path.read_text(encoding="utf-8"))
    comparison = {
        key: {"original": previous["metrics"][key], "rerun": rerun["metrics"][key],
              "absolute_difference_pp": abs(previous["metrics"][key] - rerun["metrics"][key])}
        for key in METRICS
    }
    assert all(item["absolute_difference_pp"] == 0 for item in comparison.values())
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route": "teacher-approved source-level comparison route (a); submitted evidence, not acceptance",
        "matlab_executed": False,
        "matlab_vs_python_numeric_difference": None,
        "python_rerun_verdict": "REPRODUCIBLE: deterministic metrics exactly equal",
        "matlab_acceptance_threshold_pp": {"floating_difference_below": .01, "explain_difference_above": .05},
        "source_commit": SOURCE_COMMIT,
        "source_files": sources,
        "python_functions": functions,
        "evaluator_sha256": sha(script),
        "test_sha256": sha(ROOT / "tests/test_det_source_audit.py"),
        "audit_script_sha256": sha(Path(__file__)),
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "device": "cpu"},
        "input_checks": {"images": 548, "all_annotation_files_are_ascii_named_txt": True,
                         "finite_integer_gt_and_binary_score_flags": True,
                         "finite_detection_rows_nonnegative_width_height": True,
                         "zero_width_or_height_detection_rows": zero_extent_rows,
                         "gt_rows": total_gt, "dt_rows": total_dt,
                         "exact_coordinate_half_ties": half_ties,
                         "nonexact_coordinates_within_1e_minus_10_of_half": near_half_non_ties},
        "checkpoint_sha256": manifest["protocol"]["checkpoint_sha256"],
        "full_training_yaml": str((original / "args.yaml").relative_to(ROOT)),
        "full_training_yaml_sha256": sha(original / "args.yaml"),
        "commands": commands,
        "python_reproducibility_comparison_not_matlab": comparison,
        "rerun_output_byte_equal_to_original": sha(rerun_path) == sha(original / "det-source-metrics.json"),
    }
    save(out / "audit.json", audit)
    print(json.dumps({"output": str(out), "matlab_executed": False, "comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
