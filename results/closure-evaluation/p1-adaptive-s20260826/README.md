# Adaptive seed 20260826：120轮完成记录

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent validate。
- Origin Date：2026-09-11；Version Label：adaptive_seed3_completion_v1。
- Verification Status：ANALYZED。核对保存的checkpoint、参数、日志、预测和算术；本次未重训，也未重复整次评测。

训练于北京时间9月11日13:47结束，CPU评测于13:52完成。主checkpoint为第120轮last.pt，健康checkpoint记录的零起始epoch为119，CSV含完整1至120轮。其后队列自动启动pure TAL同seed，未重复启动训练。

| 指标（百分点） | adaptive | fixed同seed | 差值 |
| --- | ---: | ---: | ---: |
| DET AP† | 21.7494 | 21.7828 | −0.0334 |
| AP50† | 39.3220 | 39.6533 | −0.3314 |
| AP75† | 20.4308 | 20.6035 | −0.1727 |
| AR500† | 37.8461 | 38.2678 | −0.4217 |
| APs | 13.0384 | 12.6454 | +0.3930 |
| APm | 30.9957 | 30.8524 | +0.1433 |
| APl | 40.1542 | 40.8172 | −0.6630 |
| ARs@500 | 27.9221 | 28.0815 | −0.1595 |

†总体列来自官方MATLAB源码的Python移植；[源码逐项对照](../../../docs/closure-review/DET源码逐项对照-20260911.md)已提交，未声称成功运行MATLAB。面积列使用官方ignore过滤后的COCO-style补充口径，原图GT面积small<1024、medium[1024,9216)、large≥9216，maxDets500。

该seed的APs为正向变化，但未达到+1.0。不能单独把它作为P1达标证据，也不能把已排除的seed20260824重新混入正式三seed均值。本run缺少完整逐轮在线assigner统计，不能凭离线探针补造历史记录。P0的完整在线统计来自另一组原始acce839补跑。

## 原始证据

- `args.yaml`、`results.csv`：有效配置和120轮原始结果，与F盘运行文件SHA-256相同。
- `evaluation-manifest.json`：固定代码52c2bef、checkpoint哈希、CPU评测参数与评测脚本版本。
- `completion-verification.json`：健康checkpoint、实际optimizer momentum=0.9、启动脚本参数、与fixed的评测协议比较及精确差值。
- `training-log-manifest.json`：12份原始训练/恢复记录，共33191184字节；初始运行、关机前输出和恢复日志均保留，未截断或重编码。
- [原始日志目录](../../../logs/closure-evaluation/p1-adaptive-s20260826/)：含训练/恢复及CPU验证、转换和DET评测日志。
- `predictions-det-548.zip`、`predictions.json.gz`：解包后逐字节与评测输入核对；不包含原始图像或GT。
- `prediction-archive-manifest.json`：预测包和548份DET文件的SHA-256。

主checkpoint SHA-256：`b446a2768b72265b60736028cf699b79faa8a19b9e41eba48c51a03a966fc60f`。

归档/核验命令（工作目录为证据仓库）：

```powershell
& 'F:/conda-envs/yolo-master/python.exe' scripts/archive_p0_raw_logs.py --run p1-adaptive-s20260826
& 'F:/conda-envs/yolo-master/python.exe' scripts/package_p0_predictions.py --run p1-adaptive-s20260826
& 'F:/conda-envs/yolo-master/python.exe' scripts/verify_completed_seed3.py --mode adaptive
```

最后一条生成新核验记录，已有记录时拒绝覆盖。实际CPU评测入口仍是 `scripts/evaluate_completed_run.py --run p1-adaptive-s20260826 --threads 2`，由CPU队列调用，原日志已保存。

## 解释限制

11/11类方法学误判已检查：不把单seed当总体推断，不择优删除旧负结果，不用这一指标变化证明监督权重、候选质量或Mosaic的因果机制。未执行显著性检验或构造置信区间。Simpson、生态/选择/碰撞偏差、基率、回归均值、幸存者偏差、多重寻找、分析路径选择、相关当因果和反向因果均按上述范围检查；重点限制是协议差异、历史统计缺失和只完成一个新增配对，结论置信程度为CAUTION。
