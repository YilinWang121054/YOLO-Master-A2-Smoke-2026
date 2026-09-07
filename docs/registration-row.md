# 登记表填写内容

公开仓库地址：<https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026>

社区进度 Issue 评论：<https://github.com/Tencent/YOLO-Master/issues/246#issuecomment-5482342694>（smoke/方案）

seed 2/seed 3 进度评论：待 GitHub 发布后补充链接；评论正文备份见 [`github-issue-246-p1-seed2-update.md`](github-issue-246-p1-seed2-update.md)

Issue 已发布评论正文备份：[`github-issue-246-update.md`](github-issue-246-update.md)

独立 TAL 冲突修复 PR：<https://github.com/Tencent/YOLO-Master/pull/253>；提交材料备份：[`github-pr-tal-conflict.md`](github-pr-tal-conflict.md)

最终分工：`A2`

| 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 | 代码/方案链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 已完成；Windows + Python 3.11.15 + PyTorch 2.5.1+cu121 + RTX 3060 Laptop GPU，详见 README | 已完成；基线 acce839c，VisDrone 64/32 确定性子集，1 epoch | 已提供 scripts/run_a2_smoke.py 及 README PowerShell 命令 | configs/VisDrone-smoke.yaml、results/args.yaml、train/val 清单 | logs/a2-visdrone-smoke-acce839-final-v2-full.log | results/results.csv、results/summary.json、checksums.sha256；退出码 0 | docs/assigner-and-config-entrypoints.md | 首次 AMP 出现非有限梯度，基线控制器降级 FP32 重跑同一 epoch 后成功；P0 建议显式 amp=False 并短跑验证 | https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/main/README.md |

首轮 exploratory 补充：`a2-visdrone-baseline-r1`，647 train / 548 val，3 epoch，最终 mAP50 `0.0558`、mAP50-95 `0.0264`；它不是正式 P0。v0.1-N 三组 1 epoch 机制 smoke 中，adaptive 相比 fixed 将 train/val small 零正样本比例从 `17.91%/16.03%` 降至 `5.88%/7.66%`，但三组 mAP 均为 0，只支持覆盖机制判断。详见 [首轮报告](first-round-analysis.md)、[`results/mechanism-r4/`](../results/mechanism-r4/) 和 [Issue #246](https://github.com/Tencent/YOLO-Master/issues/246)。

## 当前 P1 进度

- seed 20260824：fixed/adaptive/pure TAL 均已完成 120/120 epoch；正式补充指标使用官方 ignore-region 过滤后的 COCO-style 结果。
- seed 20260825：fixed/adaptive/pure TAL 均已完成 120/120 epoch；训练中断恢复配置已验证，当前无训练进程。
- 两个已完成 seed 的 adaptive-fixed 正式 `APs` 变化分别为 `-0.1637` 和 `-0.3335` pp，均值 `-0.2486` pp；这不是最终三 seed 判定。
- seed 20260826 三组 120 epoch 链已于 `2026-09-08 04:27:14 +08:00` 启动，当前 fixed STAL 运行中，随后串行 adaptive STAL、pure TAL；启动清单见 [`docs/p1-seed3-launch.md`](p1-seed3-launch.md)。
- 公开结果索引：[`docs/p1-seed1-comparison.md`](p1-seed1-comparison.md)、[`docs/p1-seed2-comparison.md`](p1-seed2-comparison.md)、[`results/p1-seed1-summary.json`](../results/p1-seed1-summary.json)、[`results/p1-seed2-summary.json`](../results/p1-seed2-summary.json)。

## 仍需老师确认

老师此前已统一确认 [`docs/teacher-questions.md`](teacher-questions.md) 中的六项口径。当前仍需向老师/班长确认的是：正式训练期 assigner 统计的采集窗口和提交截止时间。seed 3 未完成前不把当前结果表述为最终 P1 验收结论。
