# Assigner 与配置注入点

本说明记录首轮代码路径和 2026-08-31 的 A2 原型状态，路径均相对于 `Tencent/YOLO-Master` 仓库根目录。

## 当前调用链

```text
model.train(overrides)
  -> DetectionTrainer(cfg=DEFAULT_CFG, overrides=...)
  -> get_cfg(): default.yaml 与 overrides 合并、校验
  -> model.init_criterion()
  -> v8DetectionLoss(model, tal_topk=10, tal_topk2=None)
  -> TaskAlignedAssigner(...)
  -> v8DetectionLoss.get_assigned_targets_and_loss()
  -> self.assigner(...)
  -> fg_mask / target_scores / target_bboxes
```

## 代码位置

| 位置 | 作用 |
| --- | --- |
| `ultralytics/utils/tal.py` | `TaskAlignedAssigner` 定义及 `forward()` 实现；输出 `fg_mask` 等分配结果 |
| `ultralytics/utils/loss.py` | `v8DetectionLoss.__init__()` 创建 assigner；`get_assigned_targets_and_loss()` 调用 assigner 并计算检测损失 |
| `ultralytics/nn/tasks.py` | 检测模型的 `init_criterion()` 创建 `v8DetectionLoss(self)` |
| `ultralytics/cfg/default.yaml` | 训练配置的全局默认入口 |
| `ultralytics/cfg/__init__.py` | `get_cfg()` 合并默认配置和 overrides，并进行键、类型和值校验 |
| `ultralytics/models/yolo/detect/train.py` | `DetectionTrainer` 接收 cfg/overrides，定义检测训练与 loss 日志入口 |

## 上游 fixed-stride 参数状态

- `v8DetectionLoss` 的 `tal_topk` 默认值为 10，`tal_topk2` 默认值为 `None`。
- 构造 `TaskAlignedAssigner` 时，`alpha=0.5`、`beta=6.0`，stride 取检测头 stride。
- `TaskAlignedAssigner` 内部执行 `topk2 or topk`，因此该 smoke 实际只观察到 `topk_10`。
- `select_candidates_in_gts()` 已包含 fixed-stride STAL：GT 宽或高小于最小 stride 8 px 时，对应维度固定扩到 16 px。因此首轮实验不是纯 TAL。

## A2 adaptive 原型

独立分支 `feat/a2-adaptive-stal` 建立三种显式模式：

| `stal_mode` | 行为 |
| --- | --- |
| `tal` | 不扩展 GT candidate box，作为纯 TAL 对照 |
| `fixed` | 保持上游现有 8 px/16 px fixed-stride 行为，也是默认值 |
| `adaptive` | 按进入 assigner 的实际面积扩展 small candidate box、补足最近候选网格并使用面积自适应 top-k |

集中配置包括面积阈值、候选扩展系数、最小候选数和三档 top-k；`get_cfg()` 完成类型、枚举、范围和参数关系校验，`v8DetectionLoss` 负责透传。`alpha/beta/CIoU` 未改动。

“最小候选数”是冲突消解前保障。重叠 GT 竞争同一 anchor 时，仍由现有最高 IoU 规则保持一锚一 GT；当可用唯一 anchor 不足时，不承诺每个 GT 在冲突后都达到该数量。

三组 smoke 暴露出原冲突规则的零 IoU 边界问题：直接在所有 GT 上 `argmax` 可能把冲突 anchor 分给非候选 GT。当前分支先将非候选 overlap 屏蔽为负无穷，再在候选 GT 内选最高 IoU，保证冲突消解不会让任一 GT 凭空超过其 pre-conflict top-k。

## 本 smoke 的采集方式

`scripts/run_a2_smoke.py` 在运行时临时包装 `TaskAlignedAssigner.forward()`，读取返回 tuple 的第 4 项 `fg_mask`，记录每张图的前景正样本数。它还包装 `TrainingRecoveryController.recover()` 记录非有限状态和 AMP 切换。两个包装都在 `finally` 路径中还原，不修改锁定基线源码。

这套探针适合验证链路、调用次数和正样本非空，不能替代完整精度、速度、显存与稳定性验收。新增 adaptive 逻辑另由 `tests/test_stal_assignment.py` 覆盖纯 TAL 等价、阈值、空 GT、极小框、重叠冲突、非候选冲突和 FP32/AMP 一致性。当前扩展测试结果为 `21 passed, 1 skipped`（Windows 需 `PYTHONUTF8=1` 读取上游含 Unicode 的默认配置）。
