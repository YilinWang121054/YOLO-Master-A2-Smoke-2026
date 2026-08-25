#!/usr/bin/env python3
"""Run the A2 VisDrone admission smoke on a deterministic subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


BASE_REF = "acce839c7e895d6b179de7f7093fa879e237cc7b"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="a2-visdrone-smoke")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--val-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(root: Path, split: str, count: int, seed: int) -> list[Path]:
    images = sorted((root / "images" / split).glob("*.jpg"))
    if len(images) < count:
        raise RuntimeError(f"{split} has {len(images)} images, expected at least {count}")
    return sorted(random.Random(seed).sample(images, count))


def write_subset(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    import yaml

    args.subset_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        "train": select_images(args.data_root, "train", args.train_count, args.seed),
        "val": select_images(args.data_root, "val", args.val_count, args.seed + 1),
    }
    file_hashes: dict[str, str] = {}
    for split, paths in selected.items():
        list_path = args.subset_dir / f"{split}.txt"
        list_path.write_text("\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8")
        file_hashes[list_path.name] = sha256(list_path)

    yaml_path = args.subset_dir / "VisDrone-smoke.yaml"
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
        "images": {split: [path.name for path in paths] for split, paths in selected.items()},
    }
    manifest_path = args.subset_dir / "subset-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return yaml_path, manifest


def aggregate_assigner(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, Any] = {}
    for phase in ("train", "val"):
        phase_calls = [call for call in calls if call["phase"] == phase]
        values = [item for call in phase_calls for item in call["positive_per_image"]]
        branches: dict[str, Any] = {}
        for topk in sorted({call["topk"] for call in phase_calls}):
            branch_calls = [call for call in phase_calls if call["topk"] == topk]
            branch_values = [item for call in branch_calls for item in call["positive_per_image"]]
            branches[f"topk_{topk}"] = {
                "assigner_invocations": len(branch_calls),
                "sample_evaluations": len(branch_values),
                "positive_total": int(sum(branch_values)),
                "positive_per_sample_mean": (sum(branch_values) / len(branch_values)) if branch_values else None,
                "positive_per_sample_min": min(branch_values) if branch_values else None,
                "positive_per_sample_max": max(branch_values) if branch_values else None,
            }
        by_phase[phase] = {
            "assigner_invocations": len(phase_calls),
            "sample_evaluations": len(values),
            "positive_total": int(sum(values)),
            "positive_per_sample_mean": (sum(values) / len(values)) if values else None,
            "positive_per_sample_min": min(values) if values else None,
            "positive_per_sample_max": max(values) if values else None,
            "branches": branches,
        }
    return {
        "note": "Assigner invocations include any epoch attempt repeated by the trainer recovery controller; inspect recovery_events before interpreting sample_evaluations.",
        "aggregate": by_phase,
        "calls": calls,
    }


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo.resolve()))

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.extensions.recovery import TrainingRecoveryController
    from ultralytics.utils.tal import TaskAlignedAssigner

    yaml_path, subset_manifest = write_subset(args)
    calls: list[dict[str, Any]] = []
    recovery_events: list[dict[str, Any]] = []
    original_forward = TaskAlignedAssigner.forward
    original_recover = TrainingRecoveryController.recover

    def recording_forward(self, *forward_args, **forward_kwargs):
        result = original_forward(self, *forward_args, **forward_kwargs)
        fg_mask = result[3]
        values = fg_mask.detach().sum(dim=1).to(device="cpu", dtype=torch.int64).tolist()
        calls.append(
            {
                "phase": "train" if torch.is_grad_enabled() else "val",
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
            or (trainer.loss is not None and not bool(torch.isfinite(trainer.loss.detach()).all().item())),
            "fitness_nonfinite": trainer.fitness is not None
            and not bool(torch.isfinite(torch.as_tensor(trainer.fitness)).all().item()),
            "gradient_nonfinite": bool(getattr(trainer, "_gradient_nonfinite", False)),
            "ema_nonfinite": bool(getattr(trainer, "_ema_nonfinite", False)),
        }
        event = {
            "epoch_zero_based": epoch,
            "flags_before": flags,
            "amp_before": bool(getattr(trainer, "amp", False)),
            "diagnostic_before": getattr(trainer, "_nonfinite_diagnostic", None),
        }
        try:
            recovered = original_recover(self, epoch)
            event["recovered"] = recovered
            return recovered
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
            epochs=1,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=0,
            device=0,
            seed=args.seed,
            deterministic=True,
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
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            "budget": {"epochs": 1, "batch": args.batch, "imgsz": args.imgsz, "workers": 0},
            "dataset": subset_manifest,
            "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
            "metrics": metrics,
            "assigner": aggregate_assigner(calls),
            "recovery_events": recovery_events,
            "run_dir": str(run_dir.resolve()),
            "elapsed_seconds": time.time() - started,
            "limitations": [
                "Admission smoke only: one epoch on a deterministic subset.",
                "Small/medium/large AP requires the full P0 validation run.",
            ],
        }
        artifact_paths = {
            "args": run_dir / "args.yaml",
            "results": run_dir / "results.csv",
            "best_checkpoint": run_dir / "weights" / "best.pt",
            "last_checkpoint": run_dir / "weights" / "last.pt",
        }
        summary["artifacts"] = {
            name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in artifact_paths.items()
            if path.exists()
        }
        try:
            import subprocess

            summary["repo_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True, encoding="utf-8"
            ).strip()
        except Exception:
            pass
        args.project.mkdir(parents=True, exist_ok=True)
        summary_path = args.project / f"{args.name}-summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
