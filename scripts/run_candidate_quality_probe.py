"""Trace three assignment rules on identical fixed-checkpoint predictions; no training."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def choose_images(paths, count, seed):
    key = lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    groups = defaultdict(list)
    for path in paths:
        groups[path.stem.split("_")[0]].append(path)
    if len(groups) < count:
        raise ValueError("Insufficient distinct sequence prefixes")
    return [
        min(groups[group], key=lambda p: key(p.name))
        for group in sorted(groups, key=key)[:count]
    ]


def describe(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "zero_fraction": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite diagnostic values")
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5, method="inverted_cdf")),
        "p90": float(np.quantile(array, 0.9, method="inverted_cdf")),
        "zero_fraction": float((array == 0).mean()),
    }


def traced_class(base):
    class TracedAssigner(base):
        def select_candidates_in_gts(self, *args, **kwargs):
            value = super().select_candidates_in_gts(*args, **kwargs)
            self.candidates = value.clone().bool()
            return value

        def get_pos_mask(self, *args, **kwargs):
            value = super().get_pos_mask(*args, **kwargs)
            self.before, self.alignment, self.ciou = [x.clone() for x in value]
            return value

        def select_highest_overlaps(self, *args, **kwargs):
            value = super().select_highest_overlaps(*args, **kwargs)
            self.after = value[2].clone().bool()
            return value

    return TracedAssigner


def trace_assignment(base, kwargs, inputs):
    snapshots = [x.clone() for x in inputs]
    rng = torch.get_rng_state().clone()
    expected = base(**kwargs)(*inputs)
    trace = traced_class(base)(**kwargs)
    actual = trace(*inputs)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right), "Observer changes an assignment output"
        assert torch.isfinite(right).all(), "Nonfinite assignment output"
    assert torch.equal(rng, torch.get_rng_state()), "Assignment consumed RNG"
    assert all(torch.equal(a, b) for a, b in zip(inputs, snapshots)), "Input mutation"
    if inputs[4].shape[1] == 0:
        shape = (inputs[0].shape[0], 0, inputs[0].shape[1])
        trace.candidates = trace.before = trace.after = torch.zeros(
            shape, dtype=torch.bool
        )
        trace.alignment = trace.ciou = torch.zeros(shape)
    assert not (trace.before.bool() & ~trace.candidates).any()
    assert not (trace.after & ~trace.before.bool()).any()
    assert (trace.after.sum(-2) <= 1).all()
    assert torch.equal(trace.after.sum(-2).bool(), actual[3].bool())
    return trace, actual


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source = ROOT.parent / "YOLO-Master"
    git = ["git", "-C", str(source)]
    assert (
        subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        == cfg["source_commit"]
    )
    assert not subprocess.check_output(
        git + ["status", "--porcelain", "--untracked-files=no"]
    ).strip()
    assert sha(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    assert cfg["modes"] == ["tal", "fixed", "adaptive"] and cfg["dtype"] == "float32"
    assert cfg["assigner_device"] == "cpu" and cfg["mosaic"] is False
    data = Path(cfg["dataset_root"])
    images = choose_images(
        list((data / "images/train").glob("*.jpg")),
        cfg["sample_count"],
        cfg["sampling_seed"],
    )
    records = [
        {
            "image": p.relative_to(data).as_posix(),
            "image_sha256": sha(p),
            "label": f"labels/train/{p.stem}.txt",
            "label_sha256": sha(data / f"labels/train/{p.stem}.txt"),
        }
        for p in images
    ]
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out / "execution.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    started = time.monotonic()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "config_sha256": sha(args.config),
        "script_sha256": sha(__file__),
        "inputs": records,
        "evidence_commit_before_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "Post-hoc fixed-checkpoint assignment probe, no optimizer steps; not historical online statistics",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    try:
        run(cfg, data, records, out, source, started)
    except Exception:
        LOGGER.exception(
            "Probe stopped; partial artifacts retained, no automatic retry"
        )
        raise


def run(cfg, data, records, out, source, started):
    sys.path.insert(0, str(source))
    import cv2
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["sampling_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = (
        (checkpoint.get("ema") or checkpoint["model"])
        .float()
        .eval()
        .to(cfg["inference_device"])
    )
    del checkpoint
    head = model.model[-1]
    assert not head.end2end and head.nc == 10 and head.reg_max == 16
    assert head.stride.tolist() == cfg["assigner"]["stride"]
    initial_state = state_digest(model)
    letterbox = LetterBox(
        new_shape=(cfg["imgsz"], cfg["imgsz"]), auto=False, scaleup=True
    )
    gt_stats, pair_stats = defaultdict(list), defaultdict(list)
    fields = [
        "image",
        "mode",
        "gt",
        "size",
        "anchor",
        "x",
        "y",
        "stride",
        "inside_gt",
        "inside_fixed",
        "before_conflict",
        "after_conflict",
        "iou",
        "rank_ciou",
        "class_score",
        "alignment",
        "target_weight",
    ]
    with (
        (out / "candidate-records.csv.gz").open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream,
        (out / "gt-records.jsonl").open("w", encoding="utf-8") as gt_file,
    ):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image_index, record in enumerate(records):
            if time.monotonic() - started > cfg["timeout_seconds"]:
                raise TimeoutError("Diagnostic hard budget exceeded between images")
            assert sha(data / record["image"]) == record["image_sha256"]
            assert sha(data / record["label"]) == record["label_sha256"]
            image = cv2.imread(str(data / record["image"]))
            if image is None:
                raise ValueError("Unreadable image")
            params = letterbox.get_params({"img": image})
            prepared = letterbox.apply_image({"img": image}, params)["img"]
            label_path = data / record["label"]
            labels = (
                np.loadtxt(label_path, dtype=np.float32, ndmin=2)
                if label_path.read_text().strip()
                else np.empty((0, 5), dtype=np.float32)
            )
            assert labels.shape[1] == 5 and np.isfinite(labels).all()
            assert (
                (labels[:, 0] >= 0)
                & (labels[:, 0] < 10)
                & (labels[:, 0] == labels[:, 0].astype(int))
            ).all()
            assert ((labels[:, 1:] >= 0) & (labels[:, 1:] <= 1)).all()
            height, width = image.shape[:2]
            original_wh = labels[:, 3:5] * [width, height]
            center = labels[:, 1:3] * [width, height]
            boxes = np.concatenate(
                (center - original_wh / 2, center + original_wh / 2), axis=1
            )
            boxes = (
                boxes * [*params["ratio"], *params["ratio"]]
                + [params["left"], params["top"]] * 2
            )
            gt = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)
            valid = ((gt[..., 2:] - gt[..., :2]) > 0).all(-1, keepdim=True)
            assert valid.all(), (
                "Invalid GT requires explicit review, not silent filtering"
            )
            tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(prepared[:, :, ::-1].transpose(2, 0, 1))
                )
                .unsqueeze(0)
                .to(cfg["inference_device"])
                .float()
                / 255
            )
            with torch.inference_mode():
                result = model(tensor)
                preds = result[1] if isinstance(result, tuple) else result
                points, strides = make_anchors(preds["feats"], head.stride, 0.5)
                distances = preds["boxes"].permute(0, 2, 1).contiguous()
                distances = (
                    distances.reshape(1, -1, 4, 16)
                    .softmax(-1)
                    .matmul(torch.arange(16, device=tensor.device, dtype=torch.float32))
                )
                decoded = dist2bbox(distances, points, xywh=False) * strides
                scores = preds["scores"].permute(0, 2, 1).sigmoid().contiguous()
                scores, decoded, points, strides = [
                    v.cpu().clone()
                    for v in (scores, decoded, points * strides, strides)
                ]
            assert torch.isfinite(scores).all() and torch.isfinite(decoded).all()
            inputs = (
                scores,
                decoded,
                points,
                torch.tensor(labels[:, :1]).unsqueeze(0),
                gt,
                valid,
            )
            traces = {
                mode: trace_assignment(
                    TaskAlignedAssigner, {**cfg["assigner"], "stal_mode": mode}, inputs
                )
                for mode in cfg["modes"]
            }
            raw_candidates = traces["tal"][0].candidates[0]
            fixed_candidates = traces["fixed"][0].candidates[0]
            for mode, (trace, output) in traces.items():
                candidates, before, after = (
                    trace.candidates[0],
                    trace.before[0].bool(),
                    trace.after[0],
                )
                target_weights = output[2][0].sum(-1)
                for index in range(gt.shape[1]):
                    area = float((gt[0, index, 2:] - gt[0, index, :2]).prod())
                    size = (
                        "small" if area < 1024 else "medium" if area < 9216 else "large"
                    )
                    count = {
                        "raw": int(raw_candidates[index].sum()),
                        "expanded": int(candidates[index].sum()),
                        "before_conflict": int(before[index].sum()),
                        "after_conflict": int(after[index].sum()),
                        "weight_sum": float(target_weights[after[index]].sum()),
                    }
                    for stage, value in count.items():
                        gt_stats[(mode, size, stage)].append(value)
                    gt_file.write(
                        json.dumps(
                            {
                                "image": record["image"],
                                "mode": mode,
                                "gt": index,
                                "class": int(labels[index, 0]),
                                "input_bbox": boxes[index].tolist(),
                                "original_area": float(original_wh[index].prod()),
                                "input_area": area,
                                "size": size,
                                "counts": count,
                                "transform": params,
                            }
                        )
                        + "\n"
                    )
                    anchors = candidates[index].nonzero().flatten()
                    ordinary_iou = bbox_iou(
                        gt[0, index].expand(len(anchors), 4),
                        decoded[0, anchors],
                        xywh=False,
                    ).flatten()
                    for j, anchor in enumerate(anchors.tolist()):
                        final = bool(after[index, anchor])
                        weight = float(target_weights[anchor]) if final else 0.0
                        row = dict(
                            zip(
                                fields,
                                [
                                    record["image"],
                                    mode,
                                    index,
                                    size,
                                    anchor,
                                    float(points[anchor, 0]),
                                    float(points[anchor, 1]),
                                    float(strides[anchor]),
                                    int(raw_candidates[index, anchor]),
                                    int(fixed_candidates[index, anchor]),
                                    int(before[index, anchor]),
                                    int(final),
                                    float(ordinary_iou[j]),
                                    float(trace.ciou[0, index, anchor]),
                                    float(scores[0, anchor, int(labels[index, 0])]),
                                    float(trace.alignment[0, index, anchor]),
                                    weight,
                                ],
                            )
                        )
                        writer.writerow(row)
                        if final:
                            categories = [
                                "all_selected",
                                "selected_inside_gt"
                                if row["inside_gt"]
                                else "selected_outside_gt",
                            ]
                            if not row["inside_fixed"]:
                                categories.append("selected_new_vs_fixed_candidates")
                            for category in categories:
                                for metric in ("iou", "rank_ciou", "target_weight"):
                                    pair_stats[(mode, size, category, metric)].append(
                                        row[metric]
                                    )
            gt_file.flush()
            stream.flush()
            LOGGER.info(
                "Image %s/%s complete: %s, GT=%s",
                image_index + 1,
                len(records),
                record["image"],
                len(labels),
            )
    assert state_digest(model) == initial_state, "Inference changed model state_dict"
    assert sha(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    summary = {
        "status": "completed",
        "images": len(records),
        "modes": cfg["modes"],
        "elapsed_seconds": time.monotonic() - started,
        "observer_exact_output_checks": len(records) * 3,
        "model_state_unchanged": True,
        "checkpoint_unchanged": True,
        "gt_statistics": {
            "/".join(key): describe(values) for key, values in sorted(gt_stats.items())
        },
        "selected_pair_statistics": {
            "/".join(key): describe(values)
            for key, values in sorted(pair_stats.items())
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "artifacts_sha256": {
            name: sha(out / name)
            for name in (
                "manifest.json",
                "gt-records.jsonl",
                "candidate-records.csv.gz",
            )
        },
        "limits": [
            "Single fixed checkpoint and 16 selected train images; not a long-training comparison",
            "No random augmentation; not Mosaic interaction, not online training history",
            "Pair metrics weight GTs by their selected count; not independent experimental repetitions",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Complete: %s; observer equality checks=%s", out, len(records) * 3)


if __name__ == "__main__":
    main()
