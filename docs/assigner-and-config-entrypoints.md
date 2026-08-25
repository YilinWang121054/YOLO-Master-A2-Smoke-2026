# Assigner 与配置注入点

本说明针对锁定基线 `acce839c7e895d6b179de7f7093fa879e237cc7b`，路径均相对于 `Tencent/YOLO-Master` 仓库根目录。

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

## 基线参数状态

- `v8DetectionLoss` 的 `tal_topk` 默认值为 10，`tal_topk2` 默认值为 `None`。
- 构造 `TaskAlignedAssigner` 时，`alpha=0.5`、`beta=6.0`，stride 取检测头 stride。
- `TaskAlignedAssigner` 内部执行 `topk2 or topk`，因此该 smoke 实际只观察到 `topk_10`。
- 这些 assigner 参数尚未作为普通训练键出现在 `default.yaml`；若 A2 后续需要双分支或消融配置，必须同时完成默认配置、配置键校验/类型集合、loss 构造透传以及测试，不能只修改 `tal.py`。

## 本 smoke 的采集方式

`scripts/run_a2_smoke.py` 在运行时临时包装 `TaskAlignedAssigner.forward()`，读取返回 tuple 的第 4 项 `fg_mask`，记录每张图的前景正样本数。它还包装 `TrainingRecoveryController.recover()` 记录非有限状态和 AMP 切换。两个包装都在 `finally` 路径中还原，不修改锁定基线源码。

这套探针适合验证链路、调用次数和正样本非空，不构成 A2 新 assigner 的实现，也不能替代完整精度、速度、显存与稳定性验收。
