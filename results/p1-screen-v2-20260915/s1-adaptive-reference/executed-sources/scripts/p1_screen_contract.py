"""Shared paths, immutable protocol, and recovery checks for one finite screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "YOLO-Master"
CONFIG = ROOT / "configs/p1-screen-v2-20260915.json"
PROJECT = Path("F:/YOLO-Master-A2-P1/p1-screen-v2-20260915")
FROZEN = PROJECT / "freeze-manifest.json"
DATA = ROOT / "configs/VisDrone-full.yaml"
MODEL = SOURCE / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml"
CONTRACT_FILES = [
    CONFIG,
    DATA,
    *[
        ROOT / "scripts" / name
        for name in (
            "p1_screen_contract.py",
            "run_p1_screen.py",
            "evaluate_p1_screen.py",
            "supervise_p1_screen.py",
            "select_p1_screen.py",
            "assignment_observer.py",
            "assignment_coverage.py",
            "resume_p1_training.py",
            "run_p1_v2_fixed.py",
            "export_visdrone_det_txt.py",
            "evaluate_visdrone_det_strict.py",
            "evaluate_visdrone_coco_style.py",
        )
    ],
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)


def validate_config(config):
    ids = [case["id"] for case in config["cases"]]
    if len(ids) != 6 or len(set(ids)) != 6 or ids[0] != "s0-fixed":
        raise ValueError(
            "Expected one fixed reference and five distinct adaptive cases"
        )
    if (
        config["train"]["epochs"] != 20
        or 20 * len(ids) != config["training_epoch_budget"]
        or config["training_epoch_budget"] != 120
    ):
        raise ValueError("Screen budget must remain 6 x 20 epochs")
    if config["evaluation_epochs"] != list(range(16, 21)):
        raise ValueError("Use exactly the final five epochs")
    for index, case in enumerate(config["cases"]):
        allowed = {"stal_mode", "stal_candidate_scale", "stal_topk_small"}
        if set(case["overrides"]) - allowed:
            raise ValueError("Unplanned training parameter override")
        if case["overrides"]["stal_mode"] != ("fixed" if index == 0 else "adaptive"):
            raise ValueError("Incorrect reference/candidate mode")
    return config


def configuration():
    return validate_config(read(CONFIG))


def train_args(case_id):
    config = configuration()
    case = next(case for case in config["cases"] if case["id"] == case_id)
    return {
        **config["train"],
        **case["overrides"],
        "data": str(DATA),
        "project": str(PROJECT),
        "name": case_id,
        "exist_ok": False,
    }


def verify_source():
    config = configuration()
    git = ["git", "-C", str(SOURCE)]
    if (
        subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        != config["source_commit"]
    ):
        raise ValueError("Training source commit drift")
    if subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Frozen source contains tracked edits")


def verify_frozen():
    verify_source()
    manifest = read(FROZEN)
    for relative, digest in manifest["files_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Frozen contract changed: {relative}")
    if sha(MODEL) != manifest["model_yaml_sha256"]:
        raise ValueError("Model YAML changed")
    return manifest


def csv_rows(run):
    path = run / "results.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(1, len(rows) + 1)) or len(
        rows
    ) > 20:
        raise ValueError("Training CSV has gaps, duplicates, or excess epochs")
    if not all(math.isfinite(float(v)) for row in rows for v in row.values()):
        raise ValueError("Non-finite training metric")
    return rows


def verify_completion(case_id):
    run = PROJECT / case_id
    marker = run / "training-complete.json"
    if not marker.exists():
        return False
    record = read(marker)
    if record["config_sha256"] != sha(CONFIG) or len(csv_rows(run)) != 20:
        raise ValueError("Completed experiment protocol or CSV drift")
    for relative, digest in record["files_sha256"].items():
        if sha(run / relative) != digest:
            raise ValueError(f"Completed artifact changed: {case_id}/{relative}")
    return True


@contextmanager
def exclusive_lock(path):
    """OS-held lock is released on shutdown, unlike a stale PID file."""
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(f"Already running: {path.name}") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
