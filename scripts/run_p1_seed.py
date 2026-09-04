from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import freeze_support

REPO = r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master"
MODEL = os.path.join(REPO, "ultralytics", "cfg", "models", "master", "v0_1", "det", "yolo-master-n.yaml")
DATA = r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master-A2-Smoke-2026\configs\VisDrone-full.yaml"
PROJECT = r"F:\YOLO-Master-A2-P1"

os.chdir(REPO)
sys.path.insert(0, REPO)

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen A2 P1 seed.")
    parser.add_argument("--mode", choices=("tal", "fixed", "adaptive"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=120,
        imgsz=800,
        batch=4,
        workers=0,
        device=0,
        seed=args.seed,
        deterministic=True,
        amp=False,
        patience=0,
        pretrained=False,
        optimizer="MuSGD",
        lr0=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        lora_r=0,
        val=True,
        max_det=500,
        mosaic=1.0,
        close_mosaic=10,
        save=True,
        save_period=10,
        stal_mode=args.mode,
        stal_small_area=1024,
        stal_medium_area=9216,
        stal_candidate_scale=1.5,
        stal_min_candidates=3,
        stal_topk_small=13,
        stal_topk_medium=10,
        stal_topk_large=10,
        project=PROJECT,
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
