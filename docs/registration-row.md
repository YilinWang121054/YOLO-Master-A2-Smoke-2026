# 登记表填写内容

公开仓库地址：<https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026>

社区进度 Issue 评论：<https://github.com/Tencent/YOLO-Master/issues/246#issuecomment-5482342694>（smoke/方案）

seed 2/seed 3 进度评论：<https://github.com/Tencent/YOLO-Master/issues/246#issuecomment-5575508288>；评论正文备份见 [`github-issue-246-p1-seed2-update.md`](github-issue-246-p1-seed2-update.md)

Issue 已发布评论正文备份：[`github-issue-246-update.md`](github-issue-246-update.md)

独立 TAL 冲突修复 PR：<https://github.com/Tencent/YOLO-Master/pull/253>；提交材料备份：[`github-pr-tal-conflict.md`](github-pr-tal-conflict.md)

最终分工：`A2`

最终课题 PR：[Tencent/YOLO-Master #274](https://github.com/Tencent/YOLO-Master/pull/274)，北京时间2026-09-11 01:53已提交，标题含 `[犀牛鸟-A2]`，当前待review、尚未合并。

研究报告：[已发布的固定版本](https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/ce408692c0ce86c08f3803b2c4ceacedb6e941b6/docs/closure-review/研究报告.md)。P0材料：[120轮统计、分档结果及预测包](../results/closure-evaluation/p0-locked-s20260824-stats120/)，[原始训练日志](../logs/closure-evaluation/p0-locked-s20260824-stats120/training-original/)。材料公开不等于导师已认定P0验收通过。

9月11日补充：[研究报告更新版](closure-review/研究报告.md)、[DET源码逐项对照](closure-review/DET源码逐项对照-20260911.md)、[完整复现入口](closure-review/P0复现入口与后续安排-20260911.md)。建议登记文字：**“P0材料已提交，评测对齐待补；已补交Python与官方DET源码逐项对照、29项边界测试及548张预测复算，待review。PR #274已提交，待review。”** 未找到实际登记表入口，尚未代填。

| 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 | 代码/方案链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 已完成；Windows + Python 3.11.15 + PyTorch 2.5.1+cu121 + RTX 3060 Laptop GPU，详见 README | 已完成；基线 acce839c，VisDrone 64/32 确定性子集，1 epoch | 已提供 scripts/run_a2_smoke.py 及 README PowerShell 命令 | configs/VisDrone-smoke.yaml、results/args.yaml、train/val 清单 | logs/a2-visdrone-smoke-acce839-final-v2-full.log | results/results.csv、results/summary.json、checksums.sha256；退出码 0 | docs/assigner-and-config-entrypoints.md | 首次 AMP 出现非有限梯度，基线控制器降级 FP32 重跑同一 epoch 后成功；P0 建议显式 amp=False 并短跑验证 | https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/main/README.md |

首轮 exploratory 补充：`a2-visdrone-baseline-r1`，647 train / 548 val，3 epoch，最终 mAP50 `0.0558`、mAP50-95 `0.0264`；它不是正式 P0。v0.1-N 三组 1 epoch 机制 smoke 中，adaptive 相比 fixed 将 train/val small 零正样本比例从 `17.91%/16.03%` 降至 `5.88%/7.66%`，但三组 mAP 均为 0，只支持覆盖机制判断。详见 [首轮报告](first-round-analysis.md)、[`results/mechanism-r4/`](../results/mechanism-r4/) 和 [Issue #246](https://github.com/Tencent/YOLO-Master/issues/246)。

## 当前 P0/P1 进度（2026-09-11）

- seed 20260824：fixed/adaptive/pure TAL 均已完成 120/120 epoch；正式补充指标使用官方 ignore-region 过滤后的 COCO-style 结果。
- seed 20260825：fixed/adaptive/pure TAL 均已完成120/120 epoch；中断恢复记录已保留。
- 经20260910版ignore规则及面积边界重评，两对adaptive-fixed的APs变化为−0.1625、−0.3405 pp，描述性均值−0.2515 pp。按9月11日确认的口径，seed20260824归为探索性并排除正式矩阵；早期统计不完整组仅供参考，当前不能形成同协议三seed的P1达标结论。
- seed 20260826：fixed/adaptive均完成120轮及CPU完整验证；adaptive的APs为13.0384，比fixed高0.3930个百分点，但这不是P1达标结论。[新结果与原始日志](../results/closure-evaluation/p1-adaptive-s20260826/)。9月11日14:42观察到pure TAL已完成4/120轮；以实际CSV为准。启动历史见 [p1-seed3-launch.md](p1-seed3-launch.md)。
- 原始P0 acce839补跑已完成120轮、548张验证和120份真实增强后在线统计，覆盖194160个训练batch。9月11日补交允许的DET源码核验替代材料；Python完整重算与原记录逐字节一致，未取得实际MATLAB执行结果。
- 当前数值以[结题报告](closure-review/研究报告.md)及其引用的20260910重评结果为准；seed1/2早期比较文档和汇总JSON保留为历史记录，不替代修正后的结果。

## 仍需老师确认

此前[六项实验口径](teacher-questions.md)和[结题确认问题](closure-review/给老师的结题确认问题.md)的主要事项已获答复，不重复询问。新的落实记录见[复现入口与后续安排](closure-review/P0复现入口与后续安排-20260911.md)。尚需具体入口/时间安排时再确认登记表、报告/答辩模板及冻结时间；Mosaic-off的最低训练预算也可补问。PR已在9月12日前提交，不以截止前合并作为必要条件。
