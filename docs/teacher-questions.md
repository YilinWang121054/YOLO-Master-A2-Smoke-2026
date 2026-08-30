# A2 需要向老师确认的问题

1. A2 正式验收对 VisDrone small/medium/large 的面积定义是什么？是否采用 COCO 的 `32^2/96^2` 像素阈值，还是使用 VisDrone 官方评测脚本中的定义？
2. 首轮 baseline 是否必须使用完整 VisDrone 训练集和指定 epoch？当前已完成的是固定 10% 训练子集、全量验证集、3 epoch 的 exploratory baseline，能否作为机制分析前置证据？
3. A2 所说的“STAL 式小目标自适应 IoU/标签分配”更希望优先改面积感知 `top-k`，还是优先改 `alpha/beta`、IoU 计算或 warmup？
4. `topk`、面积阈值和 warmup 是否需要作为 `default.yaml` 的正式配置键，并在 PR 中补齐配置校验和单元测试？
5. P0/P1 的 small AP 应以哪一个验证划分、哪个 IoU 区间和哪个 checkpoint 作为对比？是否需要报告 AP50、COCO mAP 或二者都要？
6. 当前 RTX 3060 6 GB 在 AMP 下出现过非有限梯度，FP32 可稳定完成首轮。正式实验是否允许固定 `amp=False`，还是需要同时提交 AMP/FP32 对照？
7. 训练启用 mosaic 时，GT 面积和正样本统计会受拼接增强影响。机制分析是否建议关闭 mosaic，或要求同时报告增强前后两套统计？
8. 社区同步应优先创建 Issue 还是 Discussion？后续 STAL 代码是否应单独开 PR，并在 Issue 中关联实验编号和证据仓库？
