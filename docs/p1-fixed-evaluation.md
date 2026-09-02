# A2 fixed STAL 120e 评测记录

实验编号：`p1-fixed-s20260824`

## 训练协议

- 模型：YOLO-Master v0.1-N，YAML 从零初始化
- 数据：VisDrone2019-DET，6471 train / 548 val
- 训练：`imgsz=800`、`batch=4`、`workers=0`、120 epoch、`patience=0`
- 数值与增强：FP32、确定性 seed `20260824`、Mosaic on、`close_mosaic=10`
- 分配器：仓库已有 fixed-stride STAL，`stal_mode=fixed`
- 评测上限：`maxDets=500`
- 训练代码 commit：`52c2befa50706b9dff13b6e0813b19413d9f532d`

训练曾因用户确认的电脑关机中断，之后从带 optimizer/scaler 状态的 `last_healthy.pt` 恢复，连续完成至 120 epoch。恢复过程见 [`p1-fixed-s20260824-resume-full.log`](../logs/p1-fixed-s20260824-resume-full.log)，恢复清单见 [`run-manifest.json`](../results/p1-fixed-s20260824/run-manifest.json)。

## 独立验证

训练结束后用 `best.pt` 在完整 548 张验证集重新推理，生成未上传的 `predictions.json`，再由 [`evaluate_visdrone_coco_style.py`](../scripts/evaluate_visdrone_coco_style.py) 计算 COCO-style 指标。原始验证图像 GT bbox 面积用于分档：small `<32^2`、medium `32^2 <= area < 96^2`、large `>=96^2`。这是本课题采用的补充口径，不是 VisDrone 官方面积分档。

| 指标 | 绝对百分点 |
| --- | ---: |
| AP@[.50:.95] | 21.8304 |
| AP50 | 38.0217 |
| AP75 | 21.6330 |
| AR500 | 38.6685 |
| APs | 13.4311 |
| APm | 31.9709 |
| APl | 40.2977 |
| ARs@500 | 29.0172 |
| AP50s（辅助） | 28.4500 |

结构化结果见 [`p1-fixed-s20260824-metrics.json`](../results/p1-fixed-s20260824-metrics.json)。结果只建立 fixed 对照，不能单独说明 adaptive 的提升，也不能替代三组完整 P1 对照。

