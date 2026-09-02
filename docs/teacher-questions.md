# A2 导师确认结论与正式实验契约

更新日期：2026-09-02

## 已确认口径

1. VisDrone 官方 DET 评测没有 small/medium/large 面积分档。本课题补充分析统一采用 COCO-style：small `<32^2`、medium `32^2 <= area < 96^2`、large `>=96^2`。报告必须注明这不是 VisDrone 官方定义。
2. 评测分档按原始验证图像中的 GT bbox 面积计算；训练时 STAL 的尺度判断按数据增强、resize 后进入 assigner 的实际 bbox 面积计算。
3. Smoke 仅要求固定子集 1 epoch。P0/P1 正式结论使用 YOLO-Master v0.1-N、完整 VisDrone train、完整 val 548、`imgsz=800`、120 epoch、`patience=0`。子集或短训只能用于参数筛选。
4. P1 优先研究小目标候选区域扩展、最小正样本保障和面积自适应 top-k。`alpha=0.5`、`beta=6.0` 和当前 CIoU 排序保持不变，留到 P2 独立消融。
5. 必须包含三组对照：纯 TAL、仓库现有 fixed-stride STAL、新增 adaptive STAL。重新实现现有 fixed-stride 行为不构成本轮新增贡献。
6. 影响结果的 STAL 参数必须进入 `default.yaml` 或等价集中配置，并完成类型、范围、枚举和参数关系校验。默认配置保持当前 fixed-stride 行为，新机制显式开启。
7. 单元测试覆盖：关闭 STAL 与纯 TAL 等价、面积阈值边界、空 GT、极小框、重叠 GT、正样本冲突，以及 FP32/AMP 下 mask、top-k、正样本数、loss 和梯度的一致性与有限性。
8. 主指标为 `APs@[IoU=.50:.95]`，small `<32^2`，`maxDets=500`；“提升 >=1.0”表示至少 1.0 个绝对百分点。还需报告 AP、AP50、AP75、AR500、APm、APl、ARs@500、每档平均正样本数和零正样本 GT 比例。AP50s 仅作辅助诊断。
9. Mosaic 至少做 baseline/STAL 在 on/off 下的精简交互对照。AMP 先做分配、loss 和梯度一致性检查；若出现明显漂移或短训指标差异，再补完整 AMP on/off 训练对照。

## 2026-09-02 导师统一回复（正式执行口径）

1. **总体指标**：VisDrone 总体 `AP/AP50/AP75/AR500` 必须以官方 DET devkit 为准。可以使用与官方结果严格对齐的 Python 实现，但不能用“COCO evaluator + 按类别复制 crowd 区域”替代官方总体指标。COCO evaluator 只用于补充 `APs/APm/APl`，且必须按官方规则过滤 ignore region。
2. **120 epoch 协议**：不强制 `batch=6`；RTX 3060 6 GB 可使用 `batch=2` 或 `4`，但全部 TAL/STAL 对照和不同 seed 必须保持相同 batch、梯度累积和有效 batch。统一 YAML 建模、`pretrained=False`、完整训练 120 epoch、`patience=0`。`optimizer=auto` 只能用于确认实际选择；正式长训前必须显式冻结 optimizer、lr、momentum、weight decay 等。建议 3 个配对 seed；主结果使用第 120 轮 `last.pt`，`best.pt` 仅作补充。
3. **P1 验收**：`APs` 提升按 3 个 seed 的平均提升计算，要求平均 `ΔAPs >= 1.0` 个绝对百分点，且至少 2/3 seed 正向，同时报告每个 seed 和 mean±std。
4. **Mosaic 交互**：默认 Mosaic-on 主实验必须包含 pure TAL、现有 fixed-stride STAL、最终 adaptive STAL 三组。精简 Mosaic on/off 交互只需 pure TAL 与最终 adaptive STAL；fixed-stride 保留 Mosaic-on 结果即可。若 adaptive 不稳定超过 fixed，或交互效应明显，再补 fixed Mosaic-off。
5. **adaptive 参数**：`small_area=1024`、`medium_area=9216`、`candidate_scale=1.5`、`min_candidates=3`、`topk=13/10/10` 可作为首个候选配置。允许在长训前做一次有限筛选：最多 6 个配置、每个完整数据集 20 epoch、固定单 seed，总预算不超过一次 120 epoch。以最后 5 epoch 平均 APs 为主，并要求总体 AP 不降超过 0.3、APm/APl 原则上不降超过 0.5、零正样本率不恶化。筛选后只能选一个配置进入正式 3-seed×120e；长训开始后不得调参。P1 先冻结 1024/9216，系统阈值扫描放 P2。
6. **正样本统计**：最好同时保留训练增强后 assigner 统计和固定验证探针；P1 必交训练阶段真实进入 assigner 的统计。每档报告均值、P50/P90、零正样本比例，以及候选扩展和冲突消解前后的数量变化。原始验证集只统计面积和数量分布；未显式运行 assigner 时不得称为“验证阶段正样本统计”。

## 当前落实状态

- 首轮 3 epoch 实验已重新定位为 exploratory fixed-stride 机制观测，不能作为 P0/P1。
- 已完成 `maxDets=500` 的 COCO-style 面积分档 evaluator，并使用原始 VisDrone val GT bbox 计算面积。
- 本地 A2 分支已实现 `tal/fixed/adaptive` 三模式、集中配置和上述单元测试；默认仍为 `fixed`。
- adaptive 原型包含：small 候选框尺度扩展、候选不足时补最近网格、small/medium/large 自适应 top-k。最小保障作用于冲突消解前；重叠 GT 在候选集合内按最高 IoU 保持一锚一 GT。
- v0.1-N、64/32 固定子集、1 epoch、FP32、Mosaic off 三组机制 smoke 已完成。adaptive 相比 fixed 将 train/val small 零正样本比例从 `17.91%/16.03%` 降至 `5.88%/7.66%`；这只作为覆盖机制证据，不作为 APs 结论。
- smoke 发现并修复了零 IoU 冲突可能将 anchor 分给非候选 GT 的问题；修复后单 GT 正样本数不再超过配置 top-k，并已有回归测试。

## 历史提问（已由上述统一回复闭环）

下面内容不会阻塞短训筛选，但正式 P0/P1 前最好锁定：

1. “VisDrone 官方总体指标”是否必须调用官方 DET devkit，还是允许使用 COCO evaluator 的同一 IoU 网格、`maxDets=500`，并将 ignore region 按类别复制为 crowd 区域近似处理？两者对 ignored-region FP 的处理可能略有差异。
2. 正式 120 epoch 是否指定固定 `batch`、优化器、初始权重/从零训练、seed 数量和 checkpoint 选择规则？仓库旧脚本使用 `batch=6`、`optimizer=auto`、从 YAML 建模，但 RTX 3060 6 GB 可能无法复用 batch 6。
3. P1 的 `>=1.0 APs` 是否要求单 seed 达标，还是至少 2/3 seeds 的均值达标并报告离散度？建议正式结论至少报告多个 seed。
4. Mosaic on/off 精简交互对照中，是否只要求 pure TAL 与最终 adaptive STAL 两组，还是 fixed-stride 也必须加入该交互对照？
5. 当前 adaptive 正式实验暂定 `small_area=1024`、`medium_area=9216`、`candidate_scale=1.5`、`min_candidates=3`、`topk=13/10/10`。这些参数是否应在 P1 长训前冻结，还是允许基于 smoke 的有限参数筛选；若允许，筛选预算和选择规则是什么？
6. 每档平均正样本数和零正样本比例是否需要同时报告训练增强后 assigner 统计与原始验证集统计；若只保留一套，老师建议以验证阶段还是训练阶段为准？

可直接发给老师：

> 老师您好，六点口径已按要求落实。正式实验前还想确认六个协议细节：官方总体指标是否必须使用 VisDrone 官方 DET devkit（尤其 ignored region 处理），还是 COCO evaluator + maxDets=500 可以作为总体指标；120 epoch 是否需固定 batch、优化器、初始化、seed 数和 checkpoint 规则；APs +1.0 是按单 seed 还是多 seed 均值验收；Mosaic on/off 是否只需 pure TAL 与最终 adaptive 两组；当前 adaptive 的面积阈值、候选扩展、最小候选和 top-k 参数是否需先冻结；正样本统计是否需同时报告训练增强后与验证阶段两套。当前短训筛选会严格标注 exploratory，不用于 P1 结论。
