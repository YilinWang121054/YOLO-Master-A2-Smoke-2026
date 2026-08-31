# A2 首轮实验与初步机制分析

实验编号：`a2-visdrone-baseline-r1`

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 实验代码 HEAD | `5d49ed2dd6c3bf8bc3164ed7e62533ac3c6f74c8`；STAL 路径与当时上游一致 |
| 模型/初始化 | `YOLO-Master-EsMoE-N.pt`；不是正式协议要求的 v0.1-N |
| 数据集 | VisDrone2019-DET；固定种子 `20260824` |
| 训练/验证 | 训练 647 张（约 10% 固定子集），验证 548 张（全量） |
| 训练预算 | 3 epoch，`imgsz=640`，`batch=1`，`workers=0`，`amp=False` |
| 设备 | RTX 3060 Laptop GPU 6 GB；PyTorch 2.5.1+cu121 |
| 复现脚本 | [`run_a2_baseline.py`](../scripts/run_a2_baseline.py) |

这是首轮 exploratory 机制观测，不是完整 P0/P1。它实际使用仓库已有 fixed-stride STAL、EsMoE-N 权重、10% 训练子集和 3 epoch；不能作为 v0.1-N 正式 baseline，也不能用于认定 `APs` 提升。

## 全局结果

| epoch | precision | recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.35159 | 0.05372 | 0.03912 | 0.01999 |
| 2 | 0.48098 | 0.06599 | 0.05131 | 0.02566 |
| 3 | 0.42761 | 0.08499 | 0.05576 | 0.02639 |

训练、验证和结构化采集均成功，退出码为 0；本次显式 `amp=False`，没有触发恢复控制器。完整原始输出见 [`a2-visdrone-baseline-r1-full.log`](../results/baseline-r1/a2-visdrone-baseline-r1-full.log)，逐 epoch 指标见 [`results.csv`](../results/baseline-r1/results.csv)。

## COCO-style 面积切片诊断

使用原始 VisDrone val GT bbox 面积、`maxDets=500` 和 COCO IoU `0.50:0.05:0.95` 对现有 checkpoint 重新验证。VisDrone ignored region 以逐类别 crowd 区域近似处理。该切片是本课题采用的 COCO-style 补充口径，不是 VisDrone 官方面积定义。

| AP | AP50 | AP75 | AR500 | APs | APm | APl | ARs@500 | AP50s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.857 | 5.993 | 2.454 | 10.711 | 0.987 | 5.492 | 12.350 | 5.267 | 3.251 |

数值单位均为绝对百分点。完整结构化结果见 [`a2-coco-style-metrics.json`](../results/baseline-r1/a2-coco-style-metrics.json)，评测脚本见 [`evaluate_visdrone_coco_style.py`](../scripts/evaluate_visdrone_coco_style.py)。原始 `predictions.json` 与本地转换的 GT JSON 含数据派生内容，未上传公开仓库。

## 面积分布

面积按原始图像像素面积计算，采用导师已确认的本课题 COCO-style 分档：small `<32^2`，medium `32^2 <= area < 96^2`，large `>=96^2`。VisDrone 官方 DET 本身不提供这三档。

| split | GT 总数 | small | medium | large | small 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 子集 | 34,692 | 21,029 | 11,489 | 2,174 | 60.62% |
| val 全量 | 38,759 | 26,586 | 11,105 | 1,068 | 68.59% |

原始分布和固定清单见 [`baseline-r1-manifest.json`](../results/baseline-r1/baseline-r1-manifest.json)。

## Assigner 观察

当前实现是 `TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)`，并带有仓库已有的 fixed-stride STAL：宽或高小于最小 stride 8 px 时，对应维度扩到 16 px 后再筛 anchor center。探针记录 `fg_mask` 与 `target_gt_idx`，并按每个 GT 的变换后面积统计最终正样本数。验证阶段跨 3 epoch 的汇总如下：

| area bin | GT 记录数 | 平均正样本数/GT | 最小-最大 | 空分配比例 |
| --- | ---: | ---: | ---: | ---: |
| small | 107,481 | 3.377 | 0-12 | 10.98% |
| medium | 8,691 | 9.903 | 0-16 | 0.05% |
| large | 105 | 10.000 | 10-10 | 0% |

训练阶段的对应均值为 small 3.223、medium 9.861、large 9.964；训练启用 mosaic 后 GT 会被拼接增强重复，因此训练记录数不等于原始清单 GT 数。原始 GT 级记录见 [`a2-visdrone-baseline-r1-gt-assignment.csv`](../results/baseline-r1/a2-visdrone-baseline-r1-gt-assignment.csv)，汇总见 [`a2-visdrone-baseline-r1-summary.json`](../results/baseline-r1/a2-visdrone-baseline-r1-summary.json)。

### 初步机制假设

1. VisDrone 验证集以 small 目标为主，但 fixed `topk=10` 并不意味着每个 GT 获得 10 个正样本；候选中心约束和 task-aligned metric 的 IoU^6 项会共同筛掉大量 small 候选。
2. 即使已有 fixed-stride 扩框，small GT 的空分配比例仍明显高于 medium/large，说明候选覆盖与最小正样本保障仍有改进空间。
3. 目前只有分配统计，尚未建立 small AP/Recall 与正样本数的因果关系；不能据此宣称 AP 提升。

## 导师确认后的实验契约

- 三组必要对照统一为纯 TAL、现有 fixed-stride STAL、新增 adaptive STAL；默认配置保持 fixed，新机制显式开启。
- P1 只筛候选区域扩展、最小正样本保障和面积自适应 top-k；`alpha=0.5`、`beta=6.0` 与 CIoU 不变。
- 正式协议使用 v0.1-N、完整 train/val、`imgsz=800`、120 epoch、`patience=0`、`maxDets=500`。先做吞吐/显存 probe 和短训参数筛选，再启动长训。
- Mosaic 做 baseline/adaptive 的 on/off 精简交互对照；AMP 先检查分配、loss 与梯度一致性，出现明显漂移后才扩展完整训练对照。

## v0.1-N 三组机制 smoke（r4）

### 协议与边界

| 项目 | 设置 |
| --- | --- |
| 上游基线 | `0996b7da14dfaafae9d4488e960814ff19eb19ce` |
| 实验代码 | `52c2befa50706b9dff13b6e0813b19413d9f532d` |
| 模型 | YOLO-Master v0.1-N YAML，从零训练 |
| 数据 | 固定 64 train / 32 val，清单哈希三组一致 |
| 预算 | 1 epoch，`imgsz=800`，batch 4，workers 0 |
| 数值/增强 | FP32，Mosaic off，seed `20260824`，deterministic |
| 固定参数 | `alpha=0.5`、`beta=6.0`、CIoU 不变 |
| adaptive 参数 | candidate scale 1.5，minimum candidates 3，top-k 13/10/10 |

这组实验是固定子集 1 epoch 的机制 smoke，不是正式 P0/P1。训练分档使用增强、resize 后进入 assigner 的 bbox 面积；验证分配统计同样记录 assigner 输入面积。正式 APs 评测则必须改用原始验证图像 GT bbox 面积。

### 每档正样本覆盖

| 模式 | 阶段 | 面积档 | GT 数 | 平均正样本/GT | 零正样本 GT | 零正样本比例 | 最小-最大 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TAL | train | small | 2859 | 3.362 | 760 | 26.58% | 0-10 |
| TAL | train | medium | 495 | 9.572 | 1 | 0.20% | 0-10 |
| TAL | train | large | 39 | 10.000 | 0 | 0% | 10-10 |
| fixed | train | small | 2859 | 3.721 | 512 | 17.91% | 0-10 |
| fixed | train | medium | 495 | 9.570 | 0 | 0% | 1-10 |
| fixed | train | large | 39 | 10.000 | 0 | 0% | 10-10 |
| adaptive | train | small | 2859 | 6.473 | 168 | 5.88% | 0-13 |
| adaptive | train | medium | 495 | 9.576 | 0 | 0% | 1-10 |
| adaptive | train | large | 39 | 10.000 | 0 | 0% | 10-10 |
| TAL | val | small | 1959 | 3.720 | 426 | 21.75% | 0-10 |
| TAL | val | medium | 264 | 9.928 | 0 | 0% | 8-10 |
| TAL | val | large | 5 | 10.000 | 0 | 0% | 10-10 |
| fixed | val | small | 1959 | 4.040 | 314 | 16.03% | 0-10 |
| fixed | val | medium | 264 | 9.928 | 0 | 0% | 8-10 |
| fixed | val | large | 5 | 10.000 | 0 | 0% | 10-10 |
| adaptive | val | small | 1959 | 6.557 | 150 | 7.66% | 0-13 |
| adaptive | val | medium | 264 | 9.928 | 0 | 0% | 8-10 |
| adaptive | val | large | 5 | 10.000 | 0 | 0% | 10-10 |

adaptive 相比现有 fixed STAL，训练 small 平均正样本增加 `2.752`，零正样本比例下降 `12.03` 个百分点；验证分别增加 `2.518`、下降 `8.37` 个百分点。medium/large 基本不变，符合当前机制只优先干预 small GT 的设计目标。三组没有非有限恢复事件，单 GT 正样本上界符合各自 top-k。

三组 1 epoch 的 precision、recall、mAP50 和 mAP50-95 均为 0。该预算只能验证训练链路、数值有限和候选覆盖，不能认定 `APs@[.50:.95]` 提升，更不能据此判断已达到 `+1.0` 绝对百分点的 P1 要求。

### Smoke 中发现并修复的冲突缺陷

早期 `adaptive-r2` 诊断出现单个 GT 获得 86 个正样本，超过配置 top-k 13。根因是冲突消解对所有 GT 的 overlap 直接 `argmax`；当候选扩展产生多个零 IoU 候选时，anchor 可能被分给一个并未提出该候选的 GT。修复将非候选 GT overlap 屏蔽为负无穷后再比较，并新增零 IoU 冲突回归测试。`r2` 已从结论中排除，`r4` 的 adaptive 最大值为 13。

公开证据位于 [`results/mechanism-r4/`](../results/mechanism-r4/)；其中保留三组完整日志、逐 GT 分配 CSV、summary、results.csv、固定清单和校验和，不包含数据、预测 JSON 或 checkpoint。

## 后续正式实验

1. 用完整 6471 train、548 val、v0.1-N、`imgsz=800`、120 epoch、`patience=0` 建立 pure TAL、fixed STAL、adaptive STAL 正式对照。
2. 使用原始 val GT bbox 面积和 `maxDets=500` 报告 AP、AP50、AP75、AR500、APs、APm、APl、ARs@500；AP50s 仅作辅助诊断。
3. 至少完成 pure TAL/adaptive 的 Mosaic on/off 交互对照；最终范围待导师确认是否必须包含 fixed。
4. FP32/AMP 单元级 mask、top-k、正样本数、loss、梯度已通过；只有短训或正式训练出现明显分配/指标漂移时再扩大 AMP on/off 对照。
