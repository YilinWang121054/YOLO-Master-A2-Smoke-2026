# A2 P1 seed 2 三组统一比较

更新时间：2026-09-07

## 状态

`p1-fixed-s20260825`、`p1-adaptive-s20260825`、`p1-tal-s20260825` 均已完成完整 VisDrone train/val、120 epoch。三组主结果使用 epoch-120 `last.pt`；`best.pt` 仅作补充。训练、验证和评测的原始日志已归档在公开证据仓库。

总体 AP/AP50/AP75/AR500 使用 VisDrone DET 官方算法的 Python 移植实现。APs/APm/APl 使用本课题采用的 COCO-style 补充口径：原始验证图像 GT bbox 面积、small `<32^2`、medium `32^2 <= area < 96^2`、large `>=96^2`、`maxDets=500`。正式补充结果在 COCO evaluator 前应用 VisDrone DET 工具包的 50% 像素覆盖 ignore-region 过滤；这些面积档仍不是 VisDrone 官方面积分档。

## 结果（绝对百分点）

| 模式 | 官方 AP | 官方 AP50 | 官方 AP75 | 官方 AR500 | APs | APm | APl | ARs@500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed STAL | 21.7619 | 39.4406 | 20.7115 | 38.1235 | 13.0893 | 31.0922 | 39.4663 | 28.4870 |
| adaptive STAL | 21.4915 | 38.9398 | 20.1943 | 37.7670 | 12.7558 | 30.4260 | 36.7081 | 27.9972 |
| pure TAL | 21.5139 | 38.6953 | 20.3479 | 37.9989 | 12.7300 | 31.0032 | 42.0973 | 28.3063 |

相对 fixed，adaptive seed 2 的正式 `APs` 变化为 **-0.3335** 个百分点；pure TAL 为 **-0.3593** 个百分点。因此 seed 2 不支持“adaptive 已提升”的结论。

## 评测文件与协议审计

- 官方总体：各实验目录中的 `official-det-metrics.json`。
- 正式面积补充：各实验目录中的 `coco-style-official-filter-metrics.json`。
- 旧的 `coco-style-metrics.json` 使用 `crowd-per-class` 近似，保留用于审计，不用于正式 APs 验收。
- 三组 seed 2 均显式冻结 `MuSGD`、`lr0=0.01`、`momentum=0.9`、`weight_decay=0.0005`、`warmup_bias_lr=0.0`、FP32、`batch=4`、Mosaic on、`close_mosaic=10`。

## 两个已完成 seed 的合并观察

采用正式 official-filter 补充结果时：

- seed 20260824：adaptive-fixed `ΔAPs = -0.1637` pp；
- seed 20260825：adaptive-fixed `ΔAPs = -0.3335` pp；
- 两个已完成 seed 的平均 `ΔAPs = -0.2486` pp，两个 seed 均不是正向提升。

这不是最终 P1 判定，因为验收要求 3 个配对 seed、平均提升至少 1.0 pp 且至少 2/3 seed 为正向提升。

## 后续

- seed 20260826 的 fixed/adaptive/TAL 三组 120 epoch 尚未运行；在启动前应保持当前冻结协议不变。
- 正式报告仍需补齐训练增强后实际进入 assigner 的正样本统计（均值、P50/P90、零正样本比例，以及候选扩展/冲突消解前后数量）。
- Mosaic-off 精简交互实验按导师口径只比较 pure TAL 与最终 adaptive STAL；fixed 保留 Mosaic-on 主结果，除非交互结果显示需要补做。
