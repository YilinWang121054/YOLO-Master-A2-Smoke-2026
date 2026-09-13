"""Bounded paired CUDA FP32/FP16 forward/backward audit; never updates a weight file."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_candidate_quality_probe import sha, state_digest, traced_class

ROOT = Path(__file__).resolve().parents[1]


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare(left, right):
    union = left["fg"] | right["fg"]
    both = left["fg"] & right["fg"]
    grad_l2, ref_l2 = 0.0, 0.0
    assert left["grads"].keys() == right["grads"].keys()
    for key in left["grads"]:
        a, b = left["grads"][key].double(), right["grads"][key].double()
        grad_l2 += float(((b - a) ** 2).sum())
        ref_l2 += float((a ** 2).sum())
    return {
        "fp32_positives": int(left["fg"].sum()),
        "amp_positives": int(right["fg"].sum()),
        "foreground_mask_xor": int((left["fg"] ^ right["fg"]).sum()),
        "foreground_union": int(union.sum()),
        "gt_identity_changes_on_common_positives": int(((left["idx"] != right["idx"]) & both).sum()),
        "pre_conflict_mask_xor": int((left["before"] ^ right["before"]).sum()),
        "post_conflict_mask_xor": int((left["after"] ^ right["after"]).sum()),
        "candidate_mask_xor": int((left["candidates"] ^ right["candidates"]).sum()),
        "fp32_loss": left["loss"], "amp_loss": right["loss"],
        "loss_relative_difference": abs(right["loss"] - left["loss"]) / max(abs(left["loss"]), 1e-12),
        "gradient_relative_l2": (grad_l2 / max(ref_l2, 1e-24)) ** 0.5,
        "finite_loss_and_gradients": True,
    }


def execute(cfg, out):
    source = ROOT.parent / "YOLO-Master"
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == cfg["source_commit"]
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]).strip()
    assert sha(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    sys.path.insert(0, str(source))
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import YAML
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    manifest_path = ROOT / cfg["input_manifest"]
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = prior["inputs"][:cfg["batch"]]
    assert len(selected) == 4 and cfg["batch"] == 4 and cfg["imgsz"] == 800
    data_root = Path(prior["config"]["dataset_root"])
    for item in selected:
        assert sha(data_root / item["image"]) == item["image_sha256"]
        assert sha(data_root / item["label"]) == item["label_sha256"]
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().requires_grad_(True)
    del ckpt
    initial_state = copy.deepcopy(model.state_dict())
    initial_hash = state_digest(model)
    model = model.to("cuda")
    data = YAML.load(ROOT / "configs/VisDrone-full.yaml")
    data["nc"] = len(data["names"])
    summary = {"config": cfg, "started_at": datetime.now(timezone.utc).isoformat(),
               "source_commit": cfg["source_commit"], "input_manifest_sha256": sha(manifest_path),
               "script_sha256": sha(__file__), "selected": selected, "batches": [], "comparisons": [],
               "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(), "python": sys.version}}
    dump(out / "manifest.json", summary)
    trace_type = traced_class(TaskAlignedAssigner)
    started = time.monotonic()
    for mosaic in cfg["mosaic_conditions"]:
        reset_rng(cfg["seed"])
        hyp = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 4, "workers": 0,
                                 "mosaic": mosaic, "mixup": 0.0, "copy_paste": 0.0, "seed": cfg["seed"]})
        dataset = build_yolo_dataset(hyp, str(data_root / "images/train"), 4, data)
        assert len(dataset) == 6471
        indices = {Path(path).name: i for i, path in enumerate(dataset.im_files)}
        loaded = []
        original_get = dataset.get_image_and_label

        def record_get(index):
            loaded.append(index)
            return original_get(index)

        dataset.get_image_and_label = record_get
        reset_rng(cfg["seed"])
        batch = dataset.collate_fn([dataset[indices[Path(row["image"]).name]] for row in selected])
        batch = {k: v for k, v in batch.items() if k in ("img", "cls", "bboxes", "batch_idx")}
        # Local-only actual augmented tensors make future checks independent of augmentation RNG implementation.
        local_batch = out / f"batch-mosaic-{int(mosaic)}.pt"
        torch.save(batch, local_batch)
        batch_info = {"mosaic": mosaic, "batch_file_sha256": sha(local_batch),
                      "tensor_sha256": {k: tensor_sha(v) for k, v in batch.items()},
                      "gt_count": len(batch["cls"]), "loaded_inputs": []}
        for index in sorted(set(loaded)):
            img, label = Path(dataset.im_files[index]), Path(dataset.label_files[index])
            batch_info["loaded_inputs"].append({"image": img.relative_to(data_root).as_posix(),
                                               "image_sha256": sha(img), "label_sha256": sha(label)})
        summary["batches"].append(batch_info)
        gpu_batch = {k: v.to("cuda") for k, v in batch.items()}
        gpu_batch["img"] = gpu_batch["img"].float() / 255
        immutable_inputs = {k: tensor_sha(v) for k, v in gpu_batch.items()}
        for mode in cfg["modes"]:
            paired = []
            for amp in (False, True):
                if time.monotonic() - started > cfg["timeout_seconds"]:
                    raise TimeoutError("Audit budget exceeded; no automatic retry")
                model.load_state_dict(initial_state)
                model.train()
                model.zero_grad(set_to_none=True)
                reset_rng(cfg["seed"])
                model.args = get_cfg(overrides={"stal_mode": mode})
                criterion = v8DetectionLoss(model)
                # Trace methods only copy outputs. Same class and configuration as the real loss.
                criterion.assigner.__class__ = trace_type
                captured = {}

                def hook(_module, _inputs, output):
                    captured["fg"] = output[3].detach().cpu().bool()
                    captured["idx"] = output[4].detach().cpu()
                    captured["assigner_input_dtypes"] = [str(x.dtype) for x in _inputs]
                    assert all(torch.isfinite(x).all() for x in output)

                handle = criterion.assigner.register_forward_hook(hook)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    preds = model(gpu_batch["img"])
                    loss_vector, _ = criterion(preds, gpu_batch)
                    loss = loss_vector.sum()
                assert torch.isfinite(loss)
                scale = cfg["amp_loss_scale"] if amp else 1.0
                (loss * scale).backward()
                handle.remove()
                grads = {name: (p.grad.detach().cpu() / scale) for name, p in model.named_parameters() if p.grad is not None}
                assert grads and all(torch.isfinite(x).all() for x in grads.values())
                captured.update({"loss": float(loss.detach()), "grads": grads,
                                 "before": criterion.assigner.before.detach().cpu().bool(),
                                 "after": criterion.assigner.after.detach().cpu().bool(),
                                 "candidates": criterion.assigner.candidates.detach().cpu().bool()})
                assert {k: tensor_sha(v) for k, v in gpu_batch.items()} == immutable_inputs
                paired.append(captured)
                archive = {k: v.numpy() for k, v in captured.items() if isinstance(v, torch.Tensor)}
                np.savez_compressed(out / f"m{int(mosaic)}-{mode}-{'amp' if amp else 'fp32'}-masks.npz", **archive)
                print(json.dumps({"mosaic": mosaic, "mode": mode, "amp": amp, "loss": captured["loss"],
                                  "positives": int(captured["fg"].sum())}), flush=True)
                del preds, loss, loss_vector, criterion
            comparison = {"mosaic": mosaic, "mode": mode, **compare(*paired),
                          "input_dtypes": [x["assigner_input_dtypes"] for x in paired]}
            comparison["assignment_exact"] = all(comparison[k] == 0 for k in (
                "foreground_mask_xor", "gt_identity_changes_on_common_positives", "pre_conflict_mask_xor",
                "post_conflict_mask_xor", "candidate_mask_xor"))
            comparison["advisory"] = (not comparison["assignment_exact"] or any(
                comparison[k] > v for k, v in cfg["advisory_thresholds"].items()))
            summary["comparisons"].append(comparison)
            dump(out / "progress.json", summary)
            del paired, captured, grads
        del dataset, batch, gpu_batch
    model.load_state_dict(initial_state)
    assert state_digest(model) == initial_hash
    assert sha(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    summary.update({"status": "completed_with_advisories" if any(x["advisory"] for x in summary["comparisons"]) else "completed",
                    "elapsed_seconds": time.monotonic() - started, "checkpoint_unchanged": True,
                    "model_state_restored": True, "optimizer_steps": 0,
                    "artifact_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.suffix != ".log"}})
    dump(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "comparisons": summary["comparisons"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "launch.json", {"command": sys.argv, "config_sha256": sha(args.config),
                               "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
    try:
        execute(cfg, out)
    except Exception as error:
        dump(out / "failure.json", {"type": type(error).__name__, "message": str(error),
                                   "ended_at": datetime.now(timezone.utc).isoformat(), "retry": False})
        raise


if __name__ == "__main__":
    main()
