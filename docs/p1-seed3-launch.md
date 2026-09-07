# A2 P1 seed 3 (20260826) launch manifest

Launch timestamp recorded by the local chain state: `2026-09-07 20:27:14 UTC` (`2026-09-08 04:27:14 +08:00` China Standard Time).

## Status

- The chain is running; current phase: `fixed STAL`.
- Fixed order: `fixed STAL -> adaptive STAL -> pure TAL`.
- The chain is managed by `scripts/chain_p1_seed3.py` and uses one GPU serially.
- State file: `F:\YOLO-Master-A2-P1\recovery-logs\p1-seed3-chain\chain-state.json`.
- Current fixed output: `F:\YOLO-Master-A2-P1\p1-fixed-s20260826`.
- Logon recovery task: `YOLO-Master-A2-P1-Seed3-Chain`.

## Frozen protocol

- YOLO-Master v0.1-N from YAML, `pretrained=False`.
- Full VisDrone train/val: 6471/548.
- `imgsz=800`, `batch=4`, `workers=0`, FP32, deterministic seed `20260826`.
- Mosaic on, `close_mosaic=10`, `patience=0`.
- Explicit `MuSGD`: `lr0=0.01`, `momentum=0.9`, `weight_decay=0.0005`, `warmup_bias_lr=0.0`.
- `max_det=500`; epoch-120 `last.pt` is the primary checkpoint.
- Adaptive parameters: `small_area=1024`, `medium_area=9216`, `candidate_scale=1.5`, `min_candidates=3`, `topk=13/10/10`.
- Code commit: `52c2befa50706b9dff13b6e0813b19413d9f532d`.

## Reboot recovery

Each mode has its own resume JSON, output directory, and recovery log. The chain invokes `resume_p1_training.py` after validating epoch, optimizer/scaler/model state, dataset config, and code commit. The logon task resumes the chain after reboot and skips any completed mode.

## Evidence boundary

This manifest proves only that seed 3 started under the frozen protocol. It is not an APs-improvement or P1-acceptance claim. Final conclusions require all three modes, official DET metrics, official ignore-region-filtered COCO-style area metrics, and the three-seed summary.
