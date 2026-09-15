"""Observe actual training assignments without changing targets or consuming RNG."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


def area_bin(area):
    return "small" if area < 1024 else "medium" if area < 9216 else "large"


def describe(histogram):
    count = sum(histogram.values())
    if not count:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "zero_fraction": None,
            "histogram": {},
        }

    def quantile(fraction):
        target, cumulative = math.ceil(fraction * count), 0
        for value, number in sorted(histogram.items()):
            cumulative += number
            if cumulative >= target:
                return value

    return {
        "n": count,
        "mean": sum(k * v for k, v in histogram.items()) / count,
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "zero_fraction": histogram.get(0, 0) / count,
        "histogram": dict(sorted(histogram.items())),
    }


class AssignmentObserver:
    """Record every completed train batch; exclude validation and failed epoch attempts."""

    def __init__(self, assigner_class, expected_batch=None):
        self.cls = assigner_class
        self.expected_batch = expected_batch
        self.enabled = False
        self.epoch = 0
        self.batches = 0
        self.calls = Counter()
        self.histograms = defaultdict(Counter)
        self.pending = {}
        self.originals = {}

    def install(self):
        for name in ("forward", "select_candidates_in_gts", "select_highest_overlaps"):
            self.originals[name] = getattr(self.cls, name)
        observer = self

        def candidates(assigner, xy_centers, gt_bboxes, mask_gt, *args, **kwargs):
            result = observer.originals["select_candidates_in_gts"](
                assigner, xy_centers, gt_bboxes, mask_gt, *args, **kwargs
            )
            if observer.enabled:
                # Chunk to avoid allocating another full (batch, GT, anchors, 4) tensor.
                raw = torch.zeros(
                    gt_bboxes.shape[:2], device=gt_bboxes.device, dtype=torch.int64
                )
                for points in xy_centers.split(1024):
                    lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)
                    raw += ((points - lt > 1e-9) & (rb - points > 1e-9)).all(3).sum(-1)
                observer.pending[id(assigner)] = {
                    "raw": raw,
                    "expanded": result.sum(-1),
                }
            return result

        def conflict(assigner, mask_pos, overlaps, n_max_boxes, align_metric):
            before = mask_pos.sum(-1) if observer.enabled else None
            result = observer.originals["select_highest_overlaps"](
                assigner, mask_pos, overlaps, n_max_boxes, align_metric
            )
            if observer.enabled:
                observer.pending[id(assigner)]["before_conflict"] = before
            return result

        def forward(
            assigner, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        ):
            result = observer.originals["forward"](
                assigner,
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            if observer.enabled:
                branch = f"topk{assigner.topk}_topk2{assigner.topk2}"
                observer.calls[branch] += 1
                valid = mask_gt[..., 0].bool()
                if valid.any():
                    final = torch.zeros(
                        gt_bboxes.shape[:2], device=result[4].device, dtype=torch.int64
                    )
                    final.scatter_add_(1, result[4].long(), result[3].long())
                    state = observer.pending.pop(id(assigner))
                    state["after_conflict"] = final
                    areas = (
                        (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp_min(0).prod(-1)
                    )
                    bins = [area_bin(a) for a in areas[valid].detach().cpu().tolist()]
                    for stage, counts in state.items():
                        values = counts.to(valid.device)[valid].detach().cpu().tolist()
                        for group, value in zip(bins, values):
                            observer.histograms[(branch, group, stage)][int(value)] += 1
            return result

        self.cls.select_candidates_in_gts = candidates
        self.cls.select_highest_overlaps = conflict
        self.cls.forward = forward
        return self

    def close(self):
        for name, value in self.originals.items():
            setattr(self.cls, name, value)

    def start_epoch(self, trainer):
        if (
            self.expected_batch is not None
            and trainer.batch_size != self.expected_batch
        ):
            raise RuntimeError(
                "Training batch changed after OOM: refusing a silent protocol change"
            )
        self.enabled = True
        self.epoch = trainer.epoch + 1
        self.batches = 0
        self.calls.clear()
        self.histograms.clear()
        self.pending.clear()

    def batch_end(self, trainer):
        self.batches += 1

    def end_training(self, trainer):
        self.enabled = False
        # Persist before validation/checkpoint writes: a checkpoint must never
        # advance past its assignment evidence. Replayed epochs replace an
        # earlier attempt only after preserving it as a superseded record.
        self.save_epoch(trainer)

    def save_epoch(self, trainer):
        if not self.epoch or not self.batches:
            return
        if self.batches != len(trainer.train_loader):
            raise RuntimeError(
                "Incomplete training epoch: assignment coverage cannot be certified"
            )
        branches = {}
        for branch in self.calls:
            branches[branch] = {
                group: {
                    stage: describe(self.histograms[(branch, group, stage)])
                    for stage in (
                        "raw",
                        "expanded",
                        "before_conflict",
                        "after_conflict",
                    )
                }
                for group in ("small", "medium", "large")
            }
        payload = {
            "schema_version": 1,
            "epoch": self.epoch,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "online training: all augmented/resized batches, before validation",
            "area_definition": "A2 COCO-style: small <1024; medium [1024,9216); large >=9216",
            "quantile_definition": "inverse empirical CDF (nearest rank)",
            "batches": self.batches,
            "expected_batches": len(trainer.train_loader),
            "calls": dict(self.calls),
            "branches": branches,
        }
        directory = Path(trainer.save_dir) / "assignment"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"epoch-{self.epoch:03d}.json"
        if target.exists():
            # Retain earlier observations if an epoch is replayed after recovery.
            backup = target.with_name(
                target.stem
                + ".superseded-"
                + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
                + ".json"
            )
            target.replace(backup)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        self.batches = 0

    def attach(self, model):
        for event, callback in (
            ("on_train_epoch_start", self.start_epoch),
            ("on_train_batch_end", self.batch_end),
            ("on_train_epoch_end", self.end_training),
        ):
            model.add_callback(event, callback)
