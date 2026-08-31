# A2 v0.1-N TAL/fixed/adaptive mechanism smoke r4

本目录保存 A2 三组机制 smoke 的公开证据。三组只改变 `stal_mode`，其余设置完全一致。

## 协议

- 上游基线：`0996b7da14dfaafae9d4488e960814ff19eb19ce`
- 实验代码：`52c2befa50706b9dff13b6e0813b19413d9f532d`
- 模型：YOLO-Master v0.1-N YAML，从零训练
- 数据：VisDrone2019-DET 固定 64 train / 32 val
- 预算：1 epoch，`imgsz=800`，batch 4，workers 0
- 数值/增强：FP32，Mosaic off，seed `20260824`，deterministic
- adaptive：candidate scale 1.5，minimum candidates 3，top-k 13/10/10

## Small GT 结果

| 模式 | train 平均正样本 | train 零正样本 | val 平均正样本 | val 零正样本 |
| --- | ---: | ---: | ---: | ---: |
| TAL | 3.362 | 26.58% | 3.720 | 21.75% |
| fixed | 3.721 | 17.91% | 4.040 | 16.03% |
| adaptive | 6.473 | 5.88% | 6.557 | 7.66% |

三组 precision、recall、mAP50 和 mAP50-95 均为 0。该结果只证明固定短训中的正样本覆盖变化，不是 APs 或 P1 结论。

## 复现

```powershell
$Repo = "F:\src\YOLO-Master"
$DataRoot = "F:\datasets\VisDrone"
$Model = "$Repo\ultralytics\cfg\models\master\v0_1\det\yolo-master-n.yaml"
$Subset = "F:\datasets\VisDrone-a2-mechanism-r4"
$Runs = "F:\yolo-runs\rhino-a2"

git -C $Repo checkout 52c2befa50706b9dff13b6e0813b19413d9f532d
foreach ($Mode in "tal", "fixed", "adaptive") {
  python scripts/run_a2_baseline.py `
    --repo $Repo --data-root $DataRoot --model $Model `
    --subset-dir $Subset --project $Runs --name "a2-v01n-$Mode-r4" `
    --train-count 64 --val-count 32 --seed 20260824 `
    --imgsz 800 --batch 4 --epochs 1 --stal-mode $Mode --no-mosaic
}
```

## 文件

- `*-full.log`：完整原始控制台日志
- `*-gt-assignment.csv`：逐 GT 的变换后面积档与最终正样本数
- `*-summary.json`：环境、协议、子集、聚合统计、恢复事件和产物哈希
- `*-results.csv`：训练器原始逐 epoch 指标
- `subset-manifest.json`：固定图像清单、面积分布和清单哈希
- `checksums.sha256`：上述证据的 SHA256

未上传 VisDrone 数据、预测 JSON、训练图片或 checkpoint。
