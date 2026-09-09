"""Read actual saved optimizer momentum rather than trusting auto-start logs."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "YOLO-Master"))

import torch


def main():
    records = []
    for seed in (20260824, 20260825):
        for mode in ("fixed", "adaptive", "tal"):
            name = f"p1-{mode}-s{seed}"
            path = Path("F:/YOLO-Master-A2-P1") / name / "weights/last_healthy.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            groups = checkpoint["optimizer"]["param_groups"]
            records.append({"run": name, "checkpoint": str(path), "epoch_zero_based": checkpoint["epoch"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "actual_momentum": sorted({float(g["momentum"]) for g in groups if "momentum" in g}),
                "warmup_bias_lr": checkpoint["train_args"].get("warmup_bias_lr")})
            del checkpoint
    (ROOT / "results/optimizer-audit-20260909.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
