# Pure TAL seed 20260826：120轮完成记录

北京时间9月13日08:03训练队列确认完成，08:07完整CPU评测结束。CSV包含连续1至120轮，健康checkpoint的零起始epoch为119，optimizer状态仍在。9月12日关机前暂停、9月13日恢复的记录与初次启动日志一并保留；恢复重跑未完成的epoch，不声称从中断batch逐位接续。CSV的时间列在恢复后重新计时，末行不能直接当作整次实验耗时。

| 指标（百分数；差值为百分点） | pure TAL | fixed同seed | 差值 |
| --- | ---: | ---: | ---: |
| DET AP† | 21.5635 | 21.7828 | −0.2193 |
| AP50† | 38.6339 | 39.6533 | −1.0194 |
| AP75† | 20.6763 | 20.6035 | +0.0728 |
| AR500† | 38.0658 | 38.2678 | −0.2021 |
| APs | 12.6504 | 12.6454 | +0.0050 |
| APm | 31.6636 | 30.8524 | +0.8111 |
| APl | 40.6959 | 40.8172 | −0.1214 |
| ARs@500 | 28.4458 | 28.0815 | +0.3642 |

†总体列来自官方MATLAB源码的Python移植；[源码逐项对照](../../../docs/closure-review/DET源码逐项对照-20260911.md)已提交，未声称成功运行MATLAB。面积列使用官方ignore过滤后的COCO-style补充口径：原图GT面积small<1024、medium[1024,9216)、large≥9216，maxDets=500。这不是VisDrone官方面积分档。

该seed的pure TAL与fixed的APs接近，不作统计等价或显著性判断。adaptive同seed相对fixed的ΔAPs为+0.3930；三模式结果不能单独证明P1达标。本run的120轮均没有完整在线assigner统计，不能通过固定checkpoint探针补造训练历史。完整在线统计来自独立的P0原始acce839补跑。

## 原始证据与复核

- `args.yaml`、`results.csv`：与运行目录原文件SHA-256一致。
- `evaluation-manifest.json`：第120轮last.pt哈希、冻结代码52c2bef、CPU评测参数和脚本精确版本。
- `completion-verification.json`：核对实际momentum=0.9、冻结参数、健康checkpoint、同seed fixed评测协议和八项精确差值。
- `training-log-manifest.json`：[7份训练/恢复原始记录](../../../logs/closure-evaluation/p1-tal-s20260826/training-original/)，共32904245字节；失败与恢复历史没有删除或重编码。
- [评测原始日志](../../../logs/closure-evaluation/p1-tal-s20260826/)：CPU推理、DET转换及评测输出。
- `predictions-det-548.zip`、`predictions.json.gz`：548张预测，不含原图或GT；逐文件、ZIP CRC及解压后JSON字节已经核验。
- `prediction-archive-manifest.json`与`publication-sha256.txt`：归档校验入口。

主checkpoint SHA-256：`d26aea2e395a71222cc8fa6fe932dbe0f10d9d7a0647242c12c7972e12c96e80`。权重本体保留在F盘，未随此目录上传。

```powershell
& 'F:/conda-envs/yolo-master/python.exe' scripts/archive_p0_raw_logs.py --run p1-tal-s20260826
& 'F:/conda-envs/yolo-master/python.exe' scripts/package_p0_predictions.py --run p1-tal-s20260826
& 'F:/conda-envs/yolo-master/python.exe' scripts/verify_completed_seed3.py --mode tal
```

最后一条在已有核验记录时拒绝覆盖。实际评测入口为`scripts/evaluate_completed_run.py --run p1-tal-s20260826 --threads 2`，此前已由CPU队列执行，本次不重复评测。

## 核验范围

Material Passport：academic-research-suite / experiment-agent validate；2026-09-13；tal_seed3_completion_v1；ANALYZED。检查保存的配置、checkpoint、日志、预测和算术，不包含新增训练或MATLAB运行。

本次按11类方法学误判检查解释边界：不跨协议混合seed、不将GT观测当独立实验重复、不择优删除负结果、不从单seed推断总体。分层汇总、选择/碰撞偏差、基率、回归均值、幸存者偏差、多重寻找、分析路径选择、相关当因果和反向因果均未被用于扩大结论；机制仍是待检验假设。未进行显著性检验或构造置信区间，解释等级为CAUTION。
