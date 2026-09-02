# 登记表填写内容

公开仓库地址：<https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026>

社区进度 Issue 评论：<https://github.com/Tencent/YOLO-Master/issues/246#issuecomment-5482342694>

Issue 已发布评论正文备份：[`github-issue-246-update.md`](github-issue-246-update.md)

独立 TAL 冲突修复 PR：<https://github.com/Tencent/YOLO-Master/pull/253>；提交材料备份：[`github-pr-tal-conflict.md`](github-pr-tal-conflict.md)

最终分工：`A2`

| 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 | 代码/方案链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 已完成；Windows + Python 3.11.15 + PyTorch 2.5.1+cu121 + RTX 3060 Laptop GPU，详见 README | 已完成；基线 acce839c，VisDrone 64/32 确定性子集，1 epoch | 已提供 scripts/run_a2_smoke.py 及 README PowerShell 命令 | configs/VisDrone-smoke.yaml、results/args.yaml、train/val 清单 | logs/a2-visdrone-smoke-acce839-final-v2-full.log | results/results.csv、results/summary.json、checksums.sha256；退出码 0 | docs/assigner-and-config-entrypoints.md | 首次 AMP 出现非有限梯度，基线控制器降级 FP32 重跑同一 epoch 后成功；P0 建议显式 amp=False 并短跑验证 | https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/main/README.md |

首轮 exploratory 补充：`a2-visdrone-baseline-r1`，647 train / 548 val，3 epoch，最终 mAP50 `0.0558`、mAP50-95 `0.0264`；它不是正式 P0。v0.1-N 三组 1 epoch 机制 smoke 中，adaptive 相比 fixed 将 train/val small 零正样本比例从 `17.91%/16.03%` 降至 `5.88%/7.66%`，但三组 mAP 均为 0，只支持覆盖机制判断。详见 [首轮报告](first-round-analysis.md)、[`results/mechanism-r4/`](../results/mechanism-r4/) 和 [Issue #246](https://github.com/Tencent/YOLO-Master/issues/246)。

## 当前 P1 进度

- fixed STAL：120/120 epoch 已完成；独立 COCO-style 评测 `APs=13.4311`，完整指标见 [`docs/p1-fixed-evaluation.md`](p1-fixed-evaluation.md)。
- adaptive STAL：120 epoch 正在运行，PID、原始日志和可恢复配置保存在本地 `F:` 盘，完成后再补充公开结果链接。
- pure TAL：等待 adaptive 完成后启动，保持同一 seed、数据、模型和训练协议。

## 仍需老师确认

请重点确认 [`docs/teacher-questions.md`](teacher-questions.md) 中四项：总体指标是否必须使用 VisDrone 官方 devkit、120e 的 batch/初始化/seed/checkpoint 规则、`APs +1.0` 的多 seed 验收方式，以及 Mosaic on/off 是否必须包含 fixed 组。当前实验按仓库复现协议和单 seed（显存限制）推进，未确认项不会被表述为最终验收结论。
