# 当前局限（2026-09-11）

P0训练及运行证据已收集，不等同于导师验收通过。报告先发布，再按作者审核意见修订。

- 原始基线补跑已记录120轮全部训练batch；旧的TAL/fixed/adaptive长训仍没有同样完整的在线统计，不能事后补造。
- 当前两个已完成配对 seed 的 adaptive APs 低于 fixed，当前冻结候选不满足至少 2/3 seed 正向的 P1 条件。
- seed 1 的实际 momentum 为 0.937，后续为 0.9；pure TAL seed 1 的 warmup_bias_lr 也不同。全矩阵不完全同协议。
- 当前 fixed 对照含已合并的 TAL 冲突修复，不能称为与 acce839 逐位相同的原始代码基线。
- 面积评测的闭区间边界、ignore取整与边界已更正。本文以 `results/area-audit-20260910` 为口径；历史版本保留。
- DET Python 移植与原 MATLAB devkit 尚未完成数值对齐。MATLAB 运行报 5201。
- Mosaic 交互、P2 系统扫描和第二数据集/任务未完成。CPU 单测不等于真实训练 batch 的 CUDA AMP 全量验证。
- 功能分支已与最新上游7bfbfd3对齐；33项CPU测试通过、1项MPS测试跳过。改动文件的15项既有lint诊断仍在，没有新增，不能声称整个仓库所有检查通过。

P0/P1/P2逐项证据及待确认事项见 `docs/closure-review/`。最关键的剩余确认是原DET工具对齐、协议偏差处理及负结果的结题认定；尚未得到导师答复。
