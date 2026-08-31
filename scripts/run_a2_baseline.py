#!/usr/bin/env python3
"""Run an exploratory A2 VisDrone TAL/STAL experiment with assignment probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BASE_REF = "0996b7da14dfaafae9d4488e960814ff19eb19ce"
NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
AREA_BINS = (
    (0.0, 32.0**2, "small"),
    (32.0**2, 96.0**2, "medium"),
    (96.0**2, float("inf"), "large"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="a2-visdrone-baseline-r1")
    parser.add_argument("--train-count", type=int, default=647)
    parser.add_argument(
        "--val-count", type=int, default=0, help="0 means all validation images"
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--stal-mode", choices=("tal", "fixed", "adaptive"), default="fixed"
    )
    parser.add_argument("--stal-candidate-scale", type=float, default=1.5)
    parser.add_argument("--stal-min-candidates", type=int, default=3)
    parser.add_argument("--stal-topk-small", type=int, default=13)
    parser.add_argument("--stal-topk-medium", type=int, default=10)
    parser.add_argument("--stal-topk-large", type=int, default=10)
    parser.add_argument("--mosaic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp", action="store_true", help="Enable AMP; default is FP32 for stability"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(root: Path, split: str, count: int, seed: int) -> list[Path]:
    images = sorted((root / "images" / split).glob("*.jpg"))
    if count <= 0:
        return images
    if len(images) < count:
        raise RuntimeError(
            f"{split} has {len(images)} images, expected at least {count}"
        )
    return sorted(random.Random(seed).sample(images, count))


def area_bin(area: float) -> str:
    for low, high, name in AREA_BINS:
        if low <= area < high:
            return name
    return "large"


def collect_area_stats(
    data_root: Path, selected: dict[str, list[Path]]
) -> dict[str, Any]:
    from PIL import Image

    stats: dict[str, Any] = {}
    for split, paths in selected.items():
        counts = {name: 0 for _, _, name in AREA_BINS}
        areas: list[float] = []
        missing = 0
        for image_path in paths:
            label_path = data_root / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing += 1
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                area = float(fields[3]) * width * float(fields[4]) * height
                areas.append(area)
                counts[area_bin(area)] += 1
        ordered = sorted(areas)
        stats[split] = {
            "images": len(paths),
            "gt_count": len(areas),
            "missing_label_files": missing,
            "area_bins": counts,
            "area_px_min": ordered[0] if ordered else None,
            "area_px_median": ordered[len(ordered) // 2] if ordered else None,
            "area_px_mean": sum(ordered) / len(ordered) if ordered else None,
            "area_px_max": ordered[-1] if ordered else None,
        }
    return stats


def write_subset(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, list[Path]]]:
    import yaml

    args.subset_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        "train": select_images(args.data_root, "train", args.train_count, args.seed),
        "val": select_images(args.data_root, "val", args.val_count, args.seed + 1),
    }
    file_hashes: dict[str, str] = {}
    for split, paths in selected.items():
        list_path = args.subset_dir / f"{split}.txt"
        list_path.write_text(
            "\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8"
        )
        file_hashes[list_path.name] = sha256(list_path)

    yaml_path = args.subset_dir / "VisDrone-baseline-r1.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": args.data_root.as_posix(),
                "train": (args.subset_dir / "train.txt").as_posix(),
                "val": (args.subset_dir / "val.txt").as_posix(),
                "names": NAMES,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    file_hashes[yaml_path.name] = sha256(yaml_path)
    manifest = {
        "dataset": "VisDrone2019-DET",
        "source_root": str(args.data_root.resolve()),
        "seed": args.seed,
        "counts": {split: len(paths) for split, paths in selected.items()},
        "lists": file_hashes,
        "images": {
            split: [path.name for path in paths] for split, paths in selected.items()
        },
        "area_definition": {
            "unit": "original-image pixel squared",
            "small": "area < 32^2",
            "medium": "32^2 <= area < 96^2",
            "large": "area >= 96^2",
            "status": "COCO-style A2 project bins confirmed by the mentor; not an official VisDrone area split",
        },
        "area_stats": collect_area_stats(args.data_root, selected),
    }
    manifest_path = args.subset_dir / "baseline-r1-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return yaml_path, manifest, selected


def aggregate_assignments(
    records: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> dict[str, Any]:
    by_phase: dict[str, Any] = {}
    for phase in ("train", "val"):
        phase_records = [record for record in records if record["phase"] == phase]
        by_bin: dict[str, Any] = {}
        for bin_name in ("small", "medium", "large"):
            values = [
                record["positive_count"]
                for record in phase_records
                if record["area_bin"] == bin_name
            ]
            by_bin[bin_name] = {
                "gt_count": len(values),
                "positive_total": int(sum(values)),
                "positive_per_gt_mean": sum(values) / len(values) if values else None,
                "positive_per_gt_min": min(values) if values else None,
                "positive_per_gt_max": max(values) if values else None,
                "zero_positive_gt_count": sum(value == 0 for value in values),
            }
        by_phase[phase] = {
            "gt_count": len(phase_records),
            "by_area_bin": by_bin,
            "positive_total": int(
                sum(record["positive_count"] for record in phase_records)
            ),
            "assigner_invocations": sum(1 for call in calls if call["phase"] == phase),
        }
    return {"aggregate": by_phase, "records_count": len(records), "calls": calls}


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase",
        "assigner_call",
        "sample_index",
        "gt_index",
        "area_px",
        "area_bin",
        "positive_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo.resolve()))

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.extensions.recovery import TrainingRecoveryController
    from ultralytics.utils.tal import TaskAlignedAssigner

    yaml_path, subset_manifest, _selected = write_subset(args)
    calls: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    recovery_events: list[dict[str, Any]] = []
    original_forward = TaskAlignedAssigner.forward
    original_recover = TrainingRecoveryController.recover

    def recording_forward(self, *forward_args, **forward_kwargs):
        result = original_forward(self, *forward_args, **forward_kwargs)
        fg_mask = result[3].detach()
        target_gt_idx = result[4].detach()
        gt_bboxes = forward_args[4].detach()
        mask_gt = forward_args[5].detach().bool()[..., 0]
        area_px = (
            (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_min(0)
            * (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_min(0)
        ).cpu()
        phase = "train" if torch.is_grad_enabled() else "val"
        call_index = len(calls)
        values = fg_mask.sum(dim=1).to(device="cpu", dtype=torch.int64).tolist()
        for sample_index in range(mask_gt.shape[0]):
            positive_indices = target_gt_idx[sample_index][fg_mask[sample_index]].to(
                device="cpu", dtype=torch.long
            )
            positive_counts = torch.bincount(
                positive_indices, minlength=mask_gt.shape[1]
            ).tolist()
            for gt_index in range(mask_gt.shape[1]):
                if bool(mask_gt[sample_index, gt_index].item()):
                    area = float(area_px[sample_index, gt_index].item())
                    records.append(
                        {
                            "phase": phase,
                            "assigner_call": call_index,
                            "sample_index": sample_index,
                            "gt_index": gt_index,
                            "area_px": area,
                            "area_bin": area_bin(area),
                            "positive_count": int(positive_counts[gt_index]),
                        }
                    )
        calls.append(
            {
                "phase": phase,
                "topk": int(getattr(self, "topk", -1)),
                "positive_per_image": values,
                "positive_total": int(sum(values)),
            }
        )
        return result

    TaskAlignedAssigner.forward = recording_forward

    def recording_recover(self, epoch: int) -> bool:
        trainer = self.trainer
        flags = {
            "loss_nonfinite": bool(getattr(trainer, "_loss_nonfinite", False))
            or (
                trainer.loss is not None
                and not bool(torch.isfinite(trainer.loss.detach()).all().item())
            ),
            "fitness_nonfinite": trainer.fitness is not None
            and not bool(torch.isfinite(torch.as_tensor(trainer.fitness)).all().item()),
            "gradient_nonfinite": bool(getattr(trainer, "_gradient_nonfinite", False)),
            "ema_nonfinite": bool(getattr(trainer, "_ema_nonfinite", False)),
        }
        event = {
            "epoch_zero_based": epoch,
            "flags_before": flags,
            "amp_before": bool(getattr(trainer, "amp", False)),
        }
        try:
            event["recovered"] = original_recover(self, epoch)
            return event["recovered"]
        finally:
            event["amp_after"] = bool(getattr(trainer, "amp", False))
            if any(flags.values()) or event.get("recovered"):
                recovery_events.append(event)

    TrainingRecoveryController.recover = recording_recover
    started = time.time()
    status = "failed"
    error = None
    metrics: dict[str, float] = {}
    run_dir = args.project / args.name
    try:
        model = YOLO(str(args.model))
        result = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=0,
            device=0,
            seed=args.seed,
            deterministic=True,
            amp=args.amp,
            patience=0,
            mosaic=1.0 if args.mosaic else 0.0,
            stal_mode=args.stal_mode,
            stal_candidate_scale=args.stal_candidate_scale,
            stal_min_candidates=args.stal_min_candidates,
            stal_topk_small=args.stal_topk_small,
            stal_topk_medium=args.stal_topk_medium,
            stal_topk_large=args.stal_topk_large,
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            plots=False,
            verbose=True,
        )
        metrics = {key: float(value) for key, value in result.results_dict.items()}
        run_dir = Path(result.save_dir)
        status = "success"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        TaskAlignedAssigner.forward = original_forward
        TrainingRecoveryController.recover = original_recover
        records_path = args.project / f"{args.name}-gt-assignment.csv"
        write_records(records_path, records)
        summary = {
            "experiment_id": args.name,
            "topic": "A2",
            "status": status,
            "error": error,
            "base_ref": BASE_REF,
            "repo_head": None,
            "environment": {
                "python": sys.version,
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu",
            },
            "budget": {
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "workers": 0,
                "amp": args.amp,
                "mosaic": args.mosaic,
            },
            "stal": {
                "mode": args.stal_mode,
                "candidate_scale": args.stal_candidate_scale,
                "min_candidates": args.stal_min_candidates,
                "topk_small": args.stal_topk_small,
                "topk_medium": args.stal_topk_medium,
                "topk_large": args.stal_topk_large,
            },
            "dataset": subset_manifest,
            "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
            "metrics": metrics,
            "assigner": aggregate_assignments(records, calls),
            "recovery_events": recovery_events,
            "run_dir": str(run_dir.resolve()),
            "elapsed_seconds": time.time() - started,
            "limitations": [
                "Exploratory subset/short run only, not P0/P1.",
                "Area bins use mentor-confirmed COCO-style thresholds that are not official VisDrone area bins.",
                "The current run reports global mAP; area-sliced APs require a separate evaluator or validator extension.",
            ],
        }
        try:
            summary["repo_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True, encoding="utf-8"
            ).strip()
        except (OSError, subprocess.SubprocessError):
            summary["repo_head"] = "unavailable"
        artifacts = {
            "args": run_dir / "args.yaml",
            "results": run_dir / "results.csv",
            "best_checkpoint": run_dir / "weights" / "best.pt",
            "last_checkpoint": run_dir / "weights" / "last.pt",
            "gt_assignment_csv": records_path,
        }
        summary["artifacts"] = {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in artifacts.items()
            if path.exists()
        }
        summary_path = args.project / f"{args.name}-summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
