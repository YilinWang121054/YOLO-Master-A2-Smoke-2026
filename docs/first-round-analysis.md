# A2 首轮 baseline 与机制分析

实验编号：`a2-visdrone-baseline-r1`

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 上游基线 | `acce839c7e895d6b179de7f7093fa879e237cc7b` |
| 数据集 | VisDrone2019-DET；固定种子 `20260824` |
| 训练/验证 | 训练 647 张（约 10% 固定子集），验证 548 张（全量） |
| 训练预算 | 3 epoch，`imgsz=640`，`batch=1`，`workers=0`，`amp=False` |
| 设备 | RTX 3060 Laptop GPU 6 GB；PyTorch 2.5.1+cu121 |
| 复现脚本 | [`run_a2_baseline.py`](../scripts/run_a2_baseline.py) |

这是首轮 exploratory baseline，不是完整 P0，也不是 STAL 实现。训练子集比例、epoch 数和 FP32 设置均应在后续正式实验中重新锁定。

## 全局结果

| epoch | precision | recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.35159 | 0.05372 | 0.03912 | 0.01999 |
| 2 | 0.48098 | 0.06599 | 0.05131 | 0.02566 |
| 3 | 0.42761 | 0.08499 | 0.05576 | 0.02639 |

训练、验证和结构化采集均成功，退出码为 0；本次显式 `amp=False`，没有触发恢复控制器。完整原始输出见 [`a2-visdrone-baseline-r1-full.log`](../results/baseline-r1/a2-visdrone-baseline-r1-full.log)，逐 epoch 指标见 [`results.csv`](../results/baseline-r1/results.csv)。

## 面积分布

面积按原始图像像素面积计算，采用 COCO 风格探索性分档：small `<32^2`，medium `32^2 <= area < 96^2`，large `>=96^2`。这不是已确认的 VisDrone/A2 官方阈值。

| split | GT 总数 | small | medium | large | small 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 子集 | 34,692 | 21,029 | 11,489 | 2,174 | 60.62% |
| val 全量 | 38,759 | 26,586 | 11,105 | 1,068 | 68.59% |

原始分布和固定清单见 [`baseline-r1-manifest.json`](../results/baseline-r1/baseline-r1-manifest.json)。

## Assigner 观察

当前实现是 `TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)`。探针记录 `fg_mask` 与 `target_gt_idx`，并按每个 GT 的变换后面积统计最终正样本数。验证阶段跨 3 epoch 的汇总如下：

| area bin | GT 记录数 | 平均正样本数/GT | 最小-最大 | 空分配比例 |
| --- | ---: | ---: | ---: | ---: |
| small | 107,481 | 3.377 | 0-12 | 10.98% |
| medium | 8,691 | 9.903 | 0-16 | 0.05% |
| large | 105 | 10.000 | 10-10 | 0% |

训练阶段的对应均值为 small 3.223、medium 9.861、large 9.964；训练启用 mosaic 后 GT 会被拼接增强重复，因此训练记录数不等于原始清单 GT 数。原始 GT 级记录见 [`a2-visdrone-baseline-r1-gt-assignment.csv`](../results/baseline-r1/a2-visdrone-baseline-r1-gt-assignment.csv)，汇总见 [`a2-visdrone-baseline-r1-summary.json`](../results/baseline-r1/a2-visdrone-baseline-r1-summary.json)。

### 初步机制假设

1. VisDrone 验证集以 small 目标为主，但 fixed `topk=10` 并不意味着每个 GT 获得 10 个正样本；候选中心约束和 task-aligned metric 的 IoU^6 项会共同筛掉大量 small 候选。
2. small GT 的空分配比例明显高于 medium/large，说明“对小目标增加可用候选或降低早期 IoU 过度主导”是合理的 A2 消融方向。
3. 目前只有分配统计，尚未建立 small AP/Recall 与正样本数的因果关系；不能据此宣称 AP 提升。

## 下一轮实验

- 锁定面积定义和是否关闭 mosaic 后，再比较 baseline、STAL-off 和 STAL-on。
- 对 `topk_small/topk_medium/topk_large` 或面积感知 `alpha/beta` 做单变量消融，固定数据、预算和 seed。
- 增加面积切片 APs/Recall evaluator，并至少使用两个 seed 后再判断是否达到 P1 的 small AP +1.0 目标。
