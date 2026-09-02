# A2 P1 正式实验计划

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-02
- Verification Status: MENTOR-CONFIRMED
- Version Label: code_plan_v2

## Experiment Overview

- **Title**: YOLO-Master v0.1-N VisDrone 120-epoch TAL/STAL comparison
- **Objective**: 在导师确认的正式协议下，检验 adaptive STAL 相对仓库现有 fixed STAL 是否将 `APs@[.50:.95]` 提高至少 1.0 个绝对百分点，并以 pure TAL 作为机制对照。总体 AP/AP50/AP75/AR500 必须另用 VisDrone 官方 DET devkit 报告。
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

## First-seed execution matrix (current)

| Order | Run | `stal_mode` | Mosaic | Purpose |
| ---: | --- | --- | --- | --- |
| 1 | `p1-fixed-s20260824` | `fixed` | on | P0/current repository behavior |
| 2 | `p1-adaptive-s20260824` | `adaptive` | on | P1 candidate and primary comparison |
| 3 | `p1-tal-s20260824` | `tal` | on | Pure TAL mechanism control |

Shared settings: full VisDrone train/val, 120 epoch, `imgsz=800`, `batch=4`, `workers=0`, `seed=20260824`, deterministic, FP32, `patience=0`, `optimizer=auto`, `lora_r=0`, no pretrained weights, `max_det=500`, default Mosaic schedule (`mosaic=1.0`, `close_mosaic=10`).

## Formal 3-seed matrix (after optimizer freeze)

- Run the same three modes with paired seeds `20260824`, `20260825`, and `20260826` (or an explicitly recorded replacement set), keeping batch, gradient accumulation, effective batch, data split, code commit, and frozen adaptive parameters identical.
- The current `20260824` fixed/adaptive runs can serve as seed 1 evidence. They do not satisfy the final P1 claim until seeds 2 and 3 are completed and official/devkit plus supplementary metrics are collected for all three.
- Use epoch-120 `last.pt` as the primary checkpoint for every seed/mode. Record `best.pt` only as a supplementary diagnostic.

## Mentor-Confirmed Formal Protocol (2026-09-02)

- VisDrone overall `AP/AP50/AP75/AR500` must be reported with the official DET devkit (or a strictly aligned Python implementation). The COCO evaluator is supplementary only for `APs/APm/APl`; ignore regions must follow the official filtering rule.
- `batch=4` is allowed on the RTX 3060 6 GB. All TAL/STAL modes and all seeds must share batch, gradient accumulation, and effective batch.
- Build from YAML with `pretrained=False`, train the full dataset for 120 epochs, and use `patience=0`. Before the formal multi-seed long runs, resolve `optimizer=auto` and explicitly freeze optimizer, lr, momentum, and weight decay.
- Use 3 paired seeds. The primary checkpoint is `weights/last.pt` from epoch 120; `best.pt` is supplementary only.
- P1 acceptance is mean `ΔAPs >= 1.0` absolute percentage point across the 3 seeds, with at least 2/3 seeds positive and per-seed plus mean±std reporting.
- The default Mosaic-on matrix contains pure TAL, existing fixed-stride STAL, and adaptive STAL. The reduced Mosaic interaction compares pure TAL and final adaptive STAL only; fixed Mosaic-off is conditional.
- The current adaptive candidate (`small_area=1024`, `medium_area=9216`, `candidate_scale=1.5`, `min_candidates=3`, `topk=13/10/10`) is frozen for the current long run. Optional screening is capped at six 20-epoch full-data configurations and one 120-epoch-equivalent budget; no tuning is allowed after formal long training begins.
- Mandatory positive-assignment statistics are from the augmented training batches entering the assigner. Report mean, P50/P90, zero-positive ratio, and counts before/after candidate expansion and conflict resolution. Validation data is used for raw area/count distributions unless a validation assigner probe is explicitly run.

## Current validity boundary

- `p1-fixed-s20260824` and the in-progress `p1-adaptive-s20260824` are single-seed runs with shared `batch=4`, YAML initialization, `pretrained=False`, and the frozen adaptive candidate above. They are first-seed evidence, not the final 3-seed P1 claim.
- The existing COCO-style fixed metrics remain supplementary until official DET-devkit scoring is run on the same predictions.

## Expected Outputs

| Output | Path | Format | Success Criterion |
| --- | --- | --- | --- |
| Training metrics | `F:/YOLO-Master-A2-P1/<run>/results.csv` | CSV | 120 completed epoch rows |
| Checkpoints | `F:/YOLO-Master-A2-P1/<run>/weights/{best,last}.pt` | PyTorch | Epoch-120 `last.pt` is primary; both files are finite/loadable |
| Raw process log | async job stdout/stderr paths in the agent manifest | text | Process exits with code 0 |
| Official metrics | evidence repository `results/p1-*/official-det-metrics.json` | JSON | Official AP/AP50/AP75/AR500 present |
| COCO-style metrics | evidence repository `results/p1-*/coco-style-metrics.json` | JSON | Supplementary APs/APm/APl/ARs@500 present |
| Assignment summary | evidence repository `results/p1-*/assignment-summary.json` | JSON | Per-area mean positives and zero-positive ratio present |

## Monitoring Configuration

- **Hard timeout**: 48 hours per 120-epoch run
- **Check interval**: 5 minutes
- **Monitor files**: run `results.csv`, agent job progress, stdout/stderr
- **Experiment type override**: training
- **Metric file**: `F:/YOLO-Master-A2-P1/<run>/results.csv`
- **Metric key**: `metrics/mAP50-95(B)`; APs is computed only after external evaluation

## Interruption Recovery

- Every completed epoch updates `weights/last.pt`; `weights/last_healthy.pt` is the finite-state fallback, and periodic `epoch*.pt` files provide a third recovery tier.
- `scripts/resume_p1_training.py` validates the checkpoint epoch, optimizer, scaler, dataset config, STAL mode, code commit, CSV/checkpoint alignment, and absence of an active duplicate process before resuming.
- If `results.csv` is ahead of the latest valid checkpoint after an abrupt interruption, the script preserves a timestamped backup and removes only the uncheckpointed tail before replaying it.
- The Windows logon task runs the recovery check once after login. It does not loop-retry a crashing job; failures are recorded under `F:/YOLO-Master-A2-P1/recovery-logs/<run>` for inspection.
- Checkpoints, datasets, and prediction artifacts remain local and are not committed to the public evidence repository.
- After an interruption, the old async wrapper status may remain stale because its PID belongs to the pre-reboot process. The authoritative recovery evidence is the new resume log, the live training PID, and the checkpoint/CSV state; record the event without guessing the interruption cause.

## Analysis Plan

- **Primary metric**: supplementary COCO-style `APs@[IoU=.50:.95]`, small `<32^2` on original val GT, `maxDets=500`.
- **Official overall metrics**: VisDrone DET devkit `AP/AP50/AP75/AR500`; do not substitute the crowd-per-class COCO approximation.
- **Success threshold**: 3-seed mean adaptive minus fixed `APs >= 1.0` absolute percentage point, at least 2/3 positive.
- **Required reports**: official overall metrics; supplementary APs/APm/APl/ARs@500; per-seed and mean±std; training assigner positive statistics and zero-positive ratios.
- **Next interaction check**: after the default Mosaic-on three-way runs, run Mosaic off for pure TAL and final adaptive STAL only. Add fixed Mosaic-off only if adaptive is unstable versus fixed or the interaction is clearly material.
