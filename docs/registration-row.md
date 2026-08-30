# 登记表填写内容

公开仓库地址：<https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026>

社区进度 Issue：<https://github.com/Tencent/YOLO-Master/issues/246>

最终分工：`A2`

| 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 | 代码/方案链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 已完成；Windows + Python 3.11.15 + PyTorch 2.5.1+cu121 + RTX 3060 Laptop GPU，详见 README | 已完成；基线 acce839c，VisDrone 64/32 确定性子集，1 epoch | 已提供 scripts/run_a2_smoke.py 及 README PowerShell 命令 | configs/VisDrone-smoke.yaml、results/args.yaml、train/val 清单 | logs/a2-visdrone-smoke-acce839-final-v2-full.log | results/results.csv、results/summary.json、checksums.sha256；退出码 0 | docs/assigner-and-config-entrypoints.md | 首次 AMP 出现非有限梯度，基线控制器降级 FP32 重跑同一 epoch 后成功；P0 建议显式 amp=False 并短跑验证 | https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/main/README.md |

首轮 baseline 补充：`a2-visdrone-baseline-r1`，647 train / 548 val，3 epoch，最终 mAP50 `0.0558`、mAP50-95 `0.0264`；验证 small GT 平均 `3.377` 个正样本，`10.98%` 空分配。详见 [首轮报告](first-round-analysis.md) 和 [Issue #246](https://github.com/Tencent/YOLO-Master/issues/246)。
