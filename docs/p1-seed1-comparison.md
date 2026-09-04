# A2 P1 seed 1 三组统一比较

更新时间：2026-09-04

## 状态

`p1-fixed-s20260824`、`p1-adaptive-s20260824`、`p1-tal-s20260824` 均已完成完整 VisDrone train/val、120 epoch。三组主结果统一使用 epoch-120 `last.pt`；`best.pt` 仅作为补充。

总体 AP/AP50/AP75/AR500 使用 VisDrone DET 官方算法的 Python 移植实现；APs/APm/APl 使用本课题采用的 COCO-style 补充口径（small `<32^2`、medium `32^2 <= area < 96^2`、large `>=96^2`，原始验证图像 GT bbox，maxDets=500）。COCO-style 结果不替代官方总体指标。

## 结果（绝对百分点）

| 模式 | 官方 AP | 官方 AP50 | 官方 AP75 | 官方 AR500 | APs | APm | APl | ARs@500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed STAL | 22.4055 | 40.5716 | 21.4056 | 39.0019 | 13.4311 | 31.9709 | 40.2977 | 29.0172 |
| adaptive STAL | 22.4328 | 40.3226 | 21.4831 | 38.5890 | 13.2775 | 32.0691 | 40.6922 | 28.6725 |
| pure TAL | 21.5406 | 38.7786 | 20.3948 | 37.8632 | 12.4978 | 31.2384 | 39.7333 | 27.6399 |

相对 fixed，adaptive seed 1 的 APs 变化为 **-0.1536 个百分点**；pure TAL 为 **-0.9333 个百分点**。因此 seed 1 不支持“adaptive 已提升”的结论，也不能判定 P1 最终是否达标。

## 验收边界与后续

- P1 需要 3 个配对 seed 的 adaptive-fixed APs 平均提升至少 1.0 个绝对百分点，且至少 2/3 个 seed 为正向提升；目前只有 seed 1。
- 仍需完成 seed 2、seed 3 的 fixed/adaptive/TAL 三组训练与评测。
- 正式报告还需补齐训练增强后实际进入 assigner 的正样本统计（均值、P50/P90、零正样本比例，以及候选扩展/冲突消解前后数量）。已有 `mechanism-r4` 是 1 epoch 子集机制证据，不能替代正式长训统计。
- Mosaic-off 精简交互实验应在默认 Mosaic-on 三组完成后进行，只比较 pure TAL 与最终 adaptive STAL；若交互或稳定性显示必要，再补 fixed Mosaic-off。
