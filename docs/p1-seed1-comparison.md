# A2 P1 seed 1 三组统一比较

更新时间：2026-09-07

## 状态

`p1-fixed-s20260824`、`p1-adaptive-s20260824`、`p1-tal-s20260824` 均已完成完整 VisDrone train/val、120 epoch。三组主结果使用 epoch-120 `last.pt`；`best.pt` 仅作补充。

总体 AP/AP50/AP75/AR500 使用 VisDrone DET 官方算法的 Python 移植实现。APs/APm/APl 使用本课题采用的 COCO-style 补充口径：原始验证图像 GT bbox 面积、small `<32^2`、medium `32^2 <= area < 96^2`、large `>=96^2`、`maxDets=500`。正式补充结果在 COCO evaluator 前应用 VisDrone DET 工具包的 50% 像素覆盖 ignore-region 过滤；这些面积档仍不是 VisDrone 官方面积分档。

## 结果（绝对百分点）

| 模式 | 官方 AP | 官方 AP50 | 官方 AP75 | 官方 AR500 | APs | APm | APl | ARs@500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed STAL | 22.4055 | 40.5716 | 21.4056 | 39.0019 | 13.5427 | 31.9938 | 39.4703 | 29.1808 |
| adaptive STAL | 22.4328 | 40.3226 | 21.4831 | 38.5890 | 13.3790 | 32.0548 | 40.5131 | 28.8496 |
| pure TAL | 21.5406 | 38.7786 | 20.3948 | 37.8632 | 12.6167 | 31.2153 | 39.5915 | 27.7986 |

相对 fixed，adaptive seed 1 的正式 `APs` 变化为 **-0.1637** 个百分点；pure TAL 为 **-0.9260** 个百分点。该 seed 不支持“adaptive 已提升”的结论。

## 评测文件与协议审计

- 官方总体：各实验目录中的 `official-det-metrics.json`。
- 正式面积补充：各实验目录中的 `coco-style-official-filter-metrics.json`。
- 旧的 `coco-style-metrics.json` 使用 `crowd-per-class` 近似，保留用于审计，不用于正式 APs 验收。
- seed 1 的 fixed/adaptive 是严格配对；pure TAL 当时的 `warmup_bias_lr=0.1`，因此不纳入严格三组配对结论。后续 seed 2 已显式冻结 `warmup_bias_lr=0.0`。

## 验收边界与后续

- P1 需要 3 个配对 seed 的 adaptive-fixed `APs` 平均提升至少 1.0 个绝对百分点，且至少 2/3 个 seed 为正向提升。
- 目前 seed 1 和 seed 2 已完成；seed 3（建议 seed `20260826`）尚未运行，因此不能宣布 P1 最终达标或未达标。
- 正式报告仍需补齐训练增强后实际进入 assigner 的正样本统计（均值、P50/P90、零正样本比例，以及候选扩展/冲突消解前后数量）。
- Mosaic-off 精简交互实验仍按导师口径只比较 pure TAL 与最终 adaptive STAL；fixed 保留 Mosaic-on 主结果，除非交互结果显示需要补做。
