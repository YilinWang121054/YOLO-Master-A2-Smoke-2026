# 当前局限（2026-09-09）

本轮结题材料尚在本地审核，P0 尚未全部完成。

- 缺正式原始基线 120 个 epoch 的在线正样本统计；原长训没有记录，不能事后补造。
- 当前两个已完成配对 seed 的 adaptive APs 低于 fixed，当前冻结候选不满足至少 2/3 seed 正向的 P1 条件。
- seed 1 的实际 momentum 为 0.937，后续为 0.9；pure TAL seed 1 的 warmup_bias_lr 也不同。全矩阵不完全同协议。
- 当前 fixed 对照含已合并的 TAL 冲突修复，不能称为与 acce839 逐位相同的原始代码基线。
- 面积评测的闭区间边界已更正。以 `results/area-audit-20260909` 为新审阅口径；历史文件不回写覆盖。
- DET Python 移植与原 MATLAB devkit 尚未完成数值对齐。MATLAB 运行报 5201。
- Mosaic 交互、P2 系统扫描和第二数据集/任务未完成。CPU 单测不等于真实训练 batch 的 CUDA AMP 全量验证。
- PR 审阅分支尚需对齐最新上游。全文件 lint 的 15 项基线既有诊断仍在，不能声称整个仓库所有检查通过。

P0/P1/P2 逐项证据及待确认事项见 `docs/closure-review/`。用户审核之前不上传研究报告或 PR。
