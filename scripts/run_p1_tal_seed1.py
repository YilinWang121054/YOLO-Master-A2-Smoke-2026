from __future__ import annotations

import os
import sys
from multiprocessing import freeze_support

REPO = r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master"
os.chdir(REPO)
sys.path.insert(0, REPO)

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    model = YOLO(r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master\ultralytics\cfg\models\master\v0_1\det\yolo-master-n.yaml")
    model.train(
        data=r"E:\desktop\保研+工作\就业\实践\腾讯犀牛鸟\YOLO-Master-A2-Smoke-2026\configs\VisDrone-full.yaml",
        epochs=120,
        imgsz=800,
        batch=4,
        workers=0,
        device=0,
        seed=20260824,
        deterministic=True,
        amp=False,
        patience=0,
        pretrained=False,
        optimizer="MuSGD",
        lr0=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        lora_r=0,
        val=True,
        max_det=500,
        mosaic=1.0,
        close_mosaic=10,
        save=True,
        save_period=10,
        stal_mode="tal",
        project=r"F:\YOLO-Master-A2-P1",
        name="p1-tal-s20260824",
        exist_ok=False,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
