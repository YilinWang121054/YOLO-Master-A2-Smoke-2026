# A2 P1 正式实验计划

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-01
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Title**: YOLO-Master v0.1-N VisDrone 120-epoch TAL/STAL comparison
- **Objective**: 在导师确认的正式协议下，检验 adaptive STAL 相对仓库现有 fixed STAL 是否将 `APs@[.50:.95]` 提高至少 1.0 个绝对百分点，并以 pure TAL 作为机制对照。
- **Hypothesis**: adaptive STAL 降低 small GT 的零正样本比例，并最终改善 APs；smoke 结果只支持前半句，尚不能证明 APs 改善。
- **Type**: training

## Setup

- **Language/Framework**: Python 3.11.15, PyTorch 2.5.1+cu121, YOLO-Master/Ultralytics 8.4.101
- **Working Directory**: `E:/desktop/保研+工作/就业/实践/腾讯犀牛鸟/YOLO-Master`
- **Code**: `feat/a2-adaptive-stal` at `52c2befa50706b9dff13b6e0813b19413d9f532d`
- **Environment**: Windows, NVIDIA GeForce RTX 3060 Laptop GPU 6 GB; outputs written to `F:/YOLO-Master-A2-P1`

## Inputs

| Input | Path | Description |
| --- | --- | --- |
| Model | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` | v0.1-N, built from YAML without pretrained weights |
| Dataset | `configs/VisDrone-full.yaml` | Full 6471-image train and 548-image val splits |
| Original val GT | `F:/datasets/VisDrone/VisDrone2019-DET-val/annotations` | COCO-style area evaluation source |

## Controlled Matrix

| Order | Run | `stal_mode` | Mosaic | Purpose |
| ---: | --- | --- | --- | --- |
| 1 | `p1-fixed-s20260824` | `fixed` | on | P0/current repository behavior |
| 2 | `p1-adaptive-s20260824` | `adaptive` | on | P1 candidate and primary comparison |
| 3 | `p1-tal-s20260824` | `tal` | on | Pure TAL mechanism control |

Shared settings: full VisDrone train/val, 120 epoch, `imgsz=800`, `batch=4`, `workers=0`, `seed=20260824`, deterministic, FP32, `patience=0`, `optimizer=auto`, `lora_r=0`, no pretrained weights, `max_det=500`, default Mosaic schedule (`mosaic=1.0`, `close_mosaic=10`).

## Protocol Assumptions Pending Mentor Clarification

- Use a single fixed seed for the first formal pass because only one 6 GB GPU is available; multi-seed confirmation is a later robustness stage.
- Use `batch=4` because it is the largest smoke-verified FP32 batch at `imgsz=800` on this GPU.
- Use repository reproduction defaults: from-scratch YAML initialization and `optimizer=auto`.
- Select `best.pt` by the trainer's validation fitness, then run the same external evaluator for every group.

## Expected Outputs

| Output | Path | Format | Success Criterion |
| --- | --- | --- | --- |
| Training metrics | `F:/YOLO-Master-A2-P1/<run>/results.csv` | CSV | 120 completed epoch rows |
| Checkpoints | `F:/YOLO-Master-A2-P1/<run>/weights/{best,last}.pt` | PyTorch | Both files exist and are finite/loadable |
| Raw process log | async job stdout/stderr paths in the agent manifest | text | Process exits with code 0 |
| COCO-style metrics | evidence repository `results/p1-*/coco-style-metrics.json` | JSON | AP/AP50/AP75/AR500/APs/APm/APl/ARs@500 present |
| Assignment summary | evidence repository `results/p1-*/assignment-summary.json` | JSON | Per-area mean positives and zero-positive ratio present |

## Monitoring Configuration

- **Hard timeout**: 48 hours per 120-epoch run
- **Check interval**: 5 minutes
- **Monitor files**: run `results.csv`, agent job progress, stdout/stderr
- **Experiment type override**: training
- **Metric file**: `F:/YOLO-Master-A2-P1/<run>/results.csv`
- **Metric key**: `metrics/mAP50-95(B)`; APs is computed only after external evaluation

## Analysis Plan

- **Primary metric**: COCO-style `APs@[IoU=.50:.95]`, small `<32^2` on original val GT, `maxDets=500`.
- **Success threshold**: adaptive minus fixed `APs >= 1.0` absolute percentage point.
- **Required reports**: VisDrone overall AP, AP50, AP75, AR500; APs, APm, APl, ARs@500; per-area mean positive count and zero-positive GT ratio.
- **Claim boundary**: one-seed full runs can establish the requested first P1 comparison but not variance or statistical robustness.
- **Next interaction check**: after the three primary runs, repeat fixed and adaptive with Mosaic off; do not launch the full interaction matrix before inspecting the primary result.
