"""Archive one fully evaluated screening case without changing its frozen experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

from p1_screen_contract import (
    CONFIG,
    CONTRACT_FILES,
    FROZEN,
    PROJECT,
    ROOT,
    configuration,
    now,
    read,
    save,
    sha,
    verify_frozen,
)
from select_p1_screen import summarize_case

METRICS = {"AP", "AP50", "AP75", "AR500", "APs", "APm", "APl", "ARs@500"}


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Archive path escapes declared root")
    return path


def archive_file(source, target, *, compress=False):
    """Preserve bytes, including ANSI/CRLF, and verify compressed round trips."""
    before = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, target.open("xb") as out:
        if compress:
            with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as zipped:
                shutil.copyfileobj(inp, zipped)
        else:
            shutil.copyfileobj(inp, out)
    if compress:
        with gzip.open(target, "rb") as stream:
            restored = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        restored = sha(target)
    if sha(source) != before or restored != before:
        raise ValueError("Source changed or archive byte verification failed")
    return {
        "source": str(source),
        "source_sha256": before,
        "source_bytes": source.stat().st_size,
        "archived_sha256": sha(target),
        "archived_bytes": target.stat().st_size,
        "encoding": "gzip, byte-exact after decompression"
        if compress
        else "original bytes",
    }


def validate_evaluations(records):
    if [r["epoch"] for r in records] != list(range(16, 21)):
        raise ValueError("Exactly epochs 16-20 are required")
    for record in records:
        if record["images"] != 548 or record["device"] != "cpu":
            raise ValueError("Incomplete or unexpected evaluation")
        if set(record["metrics"]) != METRICS:
            raise ValueError("Missing required evaluation metrics")
        if not all(
            math.isfinite(v) and 0 <= v <= 100 for v in record["metrics"].values()
        ):
            raise ValueError("Invalid metric value")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", required=True, choices=[c["id"] for c in configuration()["cases"]]
    )
    args = parser.parse_args(argv)
    freeze = verify_frozen()
    summary = summarize_case(args.case)  # Checks all checkpoint and evaluation hashes.
    run = PROJECT / args.case
    completed = read(run / "training-complete.json")
    stats = [read(run / f"assignment/epoch-{e:03d}.json") for e in range(1, 21)]
    if (
        [s["epoch"] for s in stats] != list(range(1, 21))
        or any(s["batches"] != 1618 or s["expected_batches"] != 1618 for s in stats)
        or completed["training_batches"] != 32360
    ):
        raise ValueError("Incomplete online training statistics")
    eval_root = PROJECT / "evaluation" / args.case
    records = [read(eval_root / f"epoch-{e:03d}/complete.json") for e in range(16, 21)]
    validate_evaluations(records)
    logs = PROJECT / "logs" / args.case
    launches = sorted(logs.glob("*.launch.json"))
    if not launches or any(
        not p.with_name(p.name.replace(".launch.json", ".exit.json")).exists()
        for p in launches
    ):
        raise ValueError(
            "A child log has no recorded exit; defer archive for inspection"
        )
    for launch in launches:
        for suffix in (".stdout.log", ".stderr.log"):
            if not launch.with_name(
                launch.name.replace(".launch.json", suffix)
            ).exists():
                raise ValueError("Missing original child log")
    out = ROOT / "results/p1-screen-v2-20260915" / args.case
    out.mkdir(parents=True, exist_ok=False)
    files = {}

    def take(source, relative, *, compress=False):
        target = contained(out, relative)
        files[relative] = archive_file(source, target, compress=compress)

    take(FROZEN, "freeze-manifest.json")
    take(CONFIG, "screen-config.json")
    for source in [*CONTRACT_FILES, Path(__file__).resolve()]:
        relative = source.relative_to(ROOT).as_posix()
        take(source, "executed-sources/" + relative)
    for name in ("training-complete.json", "args.yaml", "results.csv"):
        take(run / name, "training/" + name)
    for epoch in range(1, 21):
        relative = f"assignment/epoch-{epoch:03d}.json"
        take(run / relative, "training/" + relative)
    for source in sorted(logs.iterdir()):
        if source.is_file() and (
            source.name.endswith(".log") or source.name.endswith(".json")
        ):
            take(source, "original-logs/" + source.name)
    for record in records:
        folder = eval_root / f"epoch-{record['epoch']:03d}"
        take(
            folder / "complete.json",
            f"evaluation/epoch-{record['epoch']:03d}/complete.json",
        )
        for relative in record["files_sha256"]:
            source = contained(folder, relative)
            compress = relative.endswith("/predictions.json")
            target = f"evaluation/epoch-{record['epoch']:03d}/{relative}" + (
                ".gz" if compress else ""
            )
            take(source, target, compress=compress)
    # Confirm frozen inputs and completed source evidence remained unchanged during copying.
    if verify_frozen() != freeze or summarize_case(args.case) != summary:
        raise ValueError("Evidence changed during archive")
    save(
        out / "summary.json",
        {
            **summary,
            "observed_at": now(),
            "training_completed_at": completed["completed_at"],
            "training_batches": 32360,
            "last_checkpoint_sha256": completed["files_sha256"]["weights/last.pt"],
            "scope": "One 20-epoch single-seed screening case; no adaptive comparison or P1 acceptance yet",
        },
        exclusive=True,
    )
    files["summary.json"] = {"archived_sha256": sha(out / "summary.json")}
    save(
        out / "archive-verification.json",
        {
            "archived_at": now(),
            "case": args.case,
            "files": files,
            "config_sha256": sha(CONFIG),
            "freeze_sha256": sha(FROZEN),
            "all_training_and_evaluation_source_hashes_verified": True,
            "excluded": [
                "dataset images",
                "GT annotations",
                "checkpoint binaries",
                "local filtered GT/predictions",
            ],
            "prediction_archives": "Original model predictions gzip-compressed with SHA-256 round-trip verification; DET txt can be regenerated with the recorded converter",
        },
        exclusive=True,
    )
    print(
        json.dumps({"case": args.case, "files": len(files), "mean": summary["mean"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
