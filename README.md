# YOLO-Master A2 Admission Smoke

## 结题材料入口（2026-09-15）

**9月15日更新：**[第二轮 fixed 对照](docs/closure-review/P1第二轮fixed完成-20260915.md)已完成120轮、120份在线统计和548张正式评测，APs=13.0095。[本组官方MATLAB补评](docs/closure-review/P1第二轮fixed-MATLAB补评成功-20260915.md)已于09:24成功，四项总体指标与Python最大差约1.21×10⁻¹²个百分点；早先5201失败记录保留。作者已确认第二轮筛选获准，[1组fixed＋5组adaptive、各20轮的配置与规则](docs/closure-review/P1第二轮筛选冻结-20260915.md)已冻结，GPU/CPU队列已启动。P1仍未达标，不将短训或旧组拼接当作正式结论。

[A2研究报告](docs/closure-review/研究报告.md) · [P0结果与逐轮统计](results/closure-evaluation/p0-locked-s20260824-stats120/) · [原始训练日志](logs/closure-evaluation/p0-locked-s20260824-stats120/training-original/) · [功能代码](https://github.com/YilinWang121054/YOLO-Master/commit/2efc4d91d7363d65b6f38e7d47c585231bb98158)

9月15日下午，[第二轮筛选的fixed 20轮对照及五次评测已归档](docs/closure-review/P1第二轮筛选-fixed对照归档-20260915.md)，最后5轮平均APs为6.4417。此值只用于同预算筛选，不与120轮结果混比。首个adaptive已完成20轮训练、尚待评测，后续候选继续按原队列运行。

最终课题 [PR #274](https://github.com/Tencent/YOLO-Master/pull/274) 已于北京时间2026年9月11日01:53提交，标题为 `[犀牛鸟-A2]：Add configurable scale-aware TAL assignment for small objects`。当前为open，尚未合并；PR内的研究报告和证据链接固定到已核验提交ce40869，后续按审核意见追加修订。

原始锁定基线已完成120轮训练、完整548张验证及120份在线正样本统计。**9月13日晚已成功运行[官方 MATLAB DET](docs/closure-review/DET官方MATLAB实测对齐-20260913.md)**，AP/AP50/AP75/AR500与Python移植实现最大差异1.28×10⁻¹²个百分点，P0材料及实测对齐已补齐、待导师验收。[PR补充评论](https://github.com/Tencent/YOLO-Master/pull/274#issuecomment-5653695161)已发布。旧seed20260826三模式全部完成，adaptive−fixed ΔAPs为+0.3930个百分点，仍不满足P1。新的[fixed 120e＋逐轮统计对照](docs/closure-review/P1第二轮fixed完成-20260915.md)也已完成；[实批次FP32/AMP检查](results/p1-v2-amp-audit-20260913/)发现分配和梯度差异，继续调查。P1其余缺项、提升要求及P2未完成的状态均保留；下方smoke仅代表准入测试。

本仓库保存腾讯犀牛鸟开源人才计划 YOLO-Master 课题 A2 的准入 smoke、首轮探索实验和初步机制分析证据。内容只覆盖 A2；它不是上游功能 PR，也不包含 A3/F1 材料。

9月14日按林老师的合并前意见，补跑了默认配置、STAL和冲突测试，共24项通过；192组CPU分配对照的输出与PR基点完全相同。[复核证据](results/pr274-review-checks-20260914/)已公开，并已在[PR回复](https://github.com/Tencent/YOLO-Master/pull/274#issuecomment-5654468495)直接附上第三seed的三模式完整结果表。功能分支未修改，PR仍待维护者处理；这次兼容性检查不代表P1达标或CUDA/AMP完全等价。

9月13日新增[固定checkpoint的候选质量诊断](results/candidate-quality-20260913/)及[候选位置与质量分布图](results/candidate-quality-20260913/figures-v1/)：同一模型与16张图像下，adaptive增加的正样本数量没有转化为同比例的监督权重增量。图和局限已补入研究报告，原始候选记录、采集等价测试和独立汇总核验均保留；这是离线诊断，不是训练历史补录或P1达标证据。

## 结论

- 状态：`success`，进程退出码为 `0`
- 上游基线：[`acce839c7e895d6b179de7f7093fa879e237cc7b`](https://github.com/Tencent/YOLO-Master/commit/acce839c7e895d6b179de7f7093fa879e237cc7b)
- 最小任务：VisDrone2019-DET 确定性子集，训练 64 张、验证 32 张，训练 1 epoch
- 预算：`batch=2`、`imgsz=320`、`workers=0`、`seed=20260824`
- 模型：YOLO-Master EsMoE-N，输入权重 SHA256 `29e1b93f09b16c8cf7c402f36dcaafc19d4812155631ed45b769e941e4c88c32`
- 训练、验证、TaskAlignedAssigner 探针、指标落盘和恢复控制器均已跑通

## 准入材料索引

| 登记项 | 证据 |
| --- | --- |
| 环境安装 | [环境与安装](docs/environment.md) |
| 基线/最小任务 | 本页“结论”与 [summary.json](results/summary.json) |
| 复现命令 | 本页“复现”与 [run_a2_smoke.py](scripts/run_a2_smoke.py) |
| 配置文件 | [VisDrone-smoke.yaml](configs/VisDrone-smoke.yaml)、[args.yaml](results/args.yaml) |
| 完整日志 | [最终 smoke 原始日志](logs/a2-visdrone-smoke-acce839-final-v2-full.log) |
| 结果证据 | [results.csv](results/results.csv)、[summary.json](results/summary.json)、[checksums.sha256](results/checksums.sha256) |
| 设计说明 | [Assigner 与配置注入点](docs/assigner-and-config-entrypoints.md) |
| 风险与降级 | 本页“风险与降级” |
| 代码/方案链接 | 本仓库 README |

数据准备的成功日志为 [visdrone-prepare.log](logs/visdrone-prepare.log)。一次被中断复制的原始失败日志也作为审计记录保留在 [visdrone-prepare-failed-interrupted-copy.log](logs/visdrone-prepare-failed-interrupted-copy.log)；之后已从原始解压目录恢复损坏文件，并完成最终 smoke。

## 结果

`results.csv` 只产生一行有效 epoch 结果：

| 指标 | 值 |
| --- | ---: |
| train/box_loss | 2.86514 |
| train/cls_loss | 5.49839 |
| train/dfl_loss | 1.60813 |
| train/mixture_aux_loss | 2.97744 |
| precision | 0 |
| recall | 0 |
| mAP50 | 0 |
| mAP50-95 | 0 |
| val/box_loss | 3.34156 |
| val/cls_loss | 5.55581 |
| val/dfl_loss | 1.61092 |

这是链路 smoke，不是精度实验。输入权重原为 COCO 类别，训练时检测头被替换为 VisDrone 的 10 类，并且只训练 64 张图、1 epoch。因此 0 mAP 只能说明该极小预算下尚未形成有效检测精度，不能作为 A2 方案的精度结论。完整 P0 才应报告全验证集 mAP 以及小/中/大目标 AP。

运行时探针对 `TaskAlignedAssigner.forward()` 返回的 `fg_mask` 做只读采集：

| 阶段 | Assigner 调用 | 样本评估数 | 正样本总数 | 每样本均值 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 64 | 128 | 15928 | 124.4375 | 6 | 372 |
| val | 16 | 64 | 6879 | 107.484375 | 6 | 241 |

这些统计包含 recovery controller 对同一 epoch 的重试，不能解释为 2 个 epoch 或 2 个 assigner 分支。原始逐调用记录见 `summary.json`。

## 风险与降级

第一次 epoch 尝试在 AMP 下出现 `gradient_nonfinite=true`，同时 `loss_nonfinite=false`、`fitness_nonfinite=false`、`ema_nonfinite=false`。基线的 recovery controller 成功恢复，自动将 `amp` 从 `true` 降为 `false`，以 FP32 重跑同一 epoch，最终退出码为 0。日志中的两个 `1/1` 进度段分别是同一 epoch 的 AMP 失败尝试与 FP32 恢复尝试，不是两次独立实验。

- 风险：EsMoE-N 在本机 Windows/CUDA/PyTorch 组合下首轮 AMP 训练出现非有限梯度。
- 准入 smoke 的实际降级：由基线恢复控制器自动切换到 FP32 并成功完成。
- 后续 P0 建议：优先显式传入 `amp=False`，扩大数据和分辨率前先做短跑稳定性验证；保留 AMP/FP32、显存峰值和恢复事件记录。
- 资源风险：RTX 3060 Laptop GPU 只有 6 GB 显存；完整 VisDrone 训练应先从保守的 batch 和 imgsz 开始，再根据显存逐级上调。

## 复现

先准备完整 VisDrone2019-DET 的 YOLO 格式数据和 YOLO-Master EsMoE-N 权重。仓库不再分发数据集和 `.pt` 文件。

```powershell
$UpstreamRepo = "F:\src\YOLO-Master"
$DataRoot = "F:\datasets\VisDrone"
$ModelPath = "F:\yolo-assets\YOLO-Master-EsMoE-N.pt"
$SubsetDir = "F:\datasets\VisDrone-smoke"
$RunRoot = "F:\yolo-runs\rhino-admission"

git -C $UpstreamRepo checkout acce839c7e895d6b179de7f7093fa879e237cc7b
python scripts/run_a2_smoke.py `
  --repo $UpstreamRepo `
  --data-root $DataRoot `
  --model $ModelPath `
  --subset-dir $SubsetDir `
  --project $RunRoot `
  --name a2-visdrone-smoke-acce839-final-v2 `
  --train-count 64 `
  --val-count 32 `
  --seed 20260824 `
  --imgsz 320 `
  --batch 2
```

脚本会按固定种子重新生成 `train.txt`、`val.txt`、`VisDrone-smoke.yaml` 和 `subset-manifest.json`，并在训练结束后写出结构化 summary。已提交的配置是本次运行的原始配置，包含本机 `F:` 盘路径；换机器复现时由上述参数重新生成，不要手工替换证据文件。

## 数据与完整性

- 完整数据：6471 张训练图、548 张验证图
- smoke 子集：64 张训练图、32 张验证图
- 子集清单：[`subset-manifest.json`](manifests/subset-manifest.json)
- `train.txt` SHA256：`afa869832252b721f66bb79262d2cf965740e3fcedf0e49229484ec45883afff`
- `val.txt` SHA256：`f29170f7c067d0dfdd65c570cf04cecfbafd5a259d64d4b91d98ea96129e96d7`
- `VisDrone-smoke.yaml` SHA256：`1ce6fb8b8761e41082274b4749502894f1b4ac56a0ddfa8b0aaa6f3e40f038a9`

`results/checksums.sha256` 覆盖仓库中的原始日志、配置、清单、脚本和结果文件。权重文件未上传，其输入及输出 checkpoint 哈希记录在 `summary.json` 中。

## 仓库边界

本仓库只用于准入登记表的公开证据链接。smoke 过程中发现的 Windows UTF-8 validator 问题已通过独立分支提交为上游 [PR #235](https://github.com/Tencent/YOLO-Master/pull/235)，修复代码不与准入证据混合。

## 首轮 baseline

- 实验：[`a2-visdrone-baseline-r1`](docs/first-round-analysis.md)
- 设置：647 张训练子集、548 张全量验证、3 epoch、640 输入、batch 1、FP32
- 最终全局指标：precision 0.427、recall 0.085、mAP50 0.0558、mAP50-95 0.0264
- 机制信号：验证 small GT 平均 3.38 个正样本且 10.98% 空分配，medium/large 平均约 9.90/10.00；该结果只支持后续 STAL 消融假设，不是 AP 提升结论
- 原始日志、GT 级分配 CSV、summary 和 results.csv：[`results/baseline-r1/`](results/baseline-r1/)
- 给老师确认的问题：[`teacher-questions.md`](docs/teacher-questions.md)

## v0.1-N 三组机制 smoke

在导师确认实验口径后，使用同一固定 64/32 子集补充了 pure TAL、上游 fixed-stride STAL 和新增 adaptive STAL 三组对照。三组均为 v0.1-N、`imgsz=800`、batch 4、1 epoch、FP32、Mosaic off、seed `20260824`，代码对应 [`52c2bef`](https://github.com/YilinWang121054/YOLO-Master/commit/52c2befa50706b9dff13b6e0813b19413d9f532d)。

| 模式 | train small 平均正样本 | train small 零正样本 | val small 平均正样本 | val small 零正样本 |
| --- | ---: | ---: | ---: | ---: |
| pure TAL | 3.362 | 26.58% | 3.720 | 21.75% |
| fixed STAL | 3.721 | 17.91% | 4.040 | 16.03% |
| adaptive STAL | 6.473 | 5.88% | 6.557 | 7.66% |

adaptive 相比现有 fixed STAL，train/val 的 small 平均正样本分别增加 `2.752`/`2.518`，零正样本比例分别下降 `12.03`/`8.37` 个百分点。三组 1 epoch mAP 均为 0，因此这些数字只支持“正样本覆盖改善”的初步机制判断，不支持 `APs` 提升或 P1 达标结论。

完整报告见 [首轮实验与机制分析](docs/first-round-analysis.md)；三组原始日志、逐 GT CSV、summary、results.csv、固定子集清单和 SHA256 见 [`results/mechanism-r4/`](results/mechanism-r4/)。

社区同步已完成：[Issue #246 进展评论](https://github.com/Tencent/YOLO-Master/issues/246#issuecomment-5482342694)。smoke 中发现的 TAL 全零 IoU 候选冲突问题已作为独立 [PR #253](https://github.com/Tencent/YOLO-Master/pull/253) 提交上游。

## P1 seed 1 三组 120e 结果已完成

fixed STAL、adaptive STAL 和 pure TAL 的 seed `20260824` 均已完成完整 VisDrone train/val、120 epoch。主结果使用 epoch-120 `last.pt`；总体 AP/AP50/AP75/AR500 使用 VisDrone DET 官方算法 Python 移植，APs/APm/APl 使用经过官方 ignore-region 过滤的 COCO-style 补充口径。结果及协议限制见 [`docs/p1-seed1-comparison.md`](docs/p1-seed1-comparison.md) 和 [`results/p1-seed1-summary.json`](results/p1-seed1-summary.json)。

| 模式 | 官方 AP | 官方 AP50 | 官方 AP75 | 官方 AR500 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed STAL | 22.4055 | 40.5716 | 21.4056 | 39.0019 | 13.5427 | 31.9938 | 39.4703 |
| adaptive STAL | 22.4328 | 40.3226 | 21.4831 | 38.5890 | 13.3790 | 32.0548 | 40.5131 |
| pure TAL | 21.5406 | 38.7786 | 20.3948 | 37.8632 | 12.6167 | 31.2153 | 39.5915 |

adaptive 相对 fixed 的 seed-1 正式 `APs` 变化为 `-0.1637` 个百分点，pure TAL 为 `-0.9260` 个百分点。该 seed 不支持“adaptive 已提升”的结论，也不能单独判定 P1 是否达标；P1 仍需 3 个配对 seed 的平均 `ΔAPs >= 1.0` 且至少 2/3 seed 为正向提升。旧的 `coco-style-metrics.json` 仍保留 crowd-per-class 近似值，仅用于审计。

协议复核发现 pure TAL seed 1 的 `warmup_bias_lr=0.1`，而 fixed/adaptive 的 `optimizer=auto` 将其设为 `0.0`。因此 fixed-adaptive 主比较仍为同协议配对，但 pure TAL 当前只作初步对照；后续实验已统一显式冻结 `warmup_bias_lr=0.0`。

TAL 结果证据：[`official-det-metrics.json`](results/p1-tal-s20260824/official-det-metrics.json)、正式 [`coco-style-official-filter-metrics.json`](results/p1-tal-s20260824/coco-style-official-filter-metrics.json)、[`completion-manifest.json`](results/p1-tal-s20260824/completion-manifest.json)。旧的 [`coco-style-metrics.json`](results/p1-tal-s20260824/coco-style-metrics.json) 仅作为 `crowd-per-class` 审计近似保留。正式训练 assigner 统计仍需补齐，不能用 `mechanism-r4` 的 1 epoch 子集统计替代。

## P1 seed 2 三组 120e 结果已完成

fixed STAL、adaptive STAL 和 pure TAL 的 seed `20260825` 均已完成完整 VisDrone train/val、120 epoch；训练曾支持关机后的 checkpoint 恢复，当前无残留训练进程。总体指标使用官方 DET Python 移植，面积补充使用官方 ignore-region 过滤后的 COCO-style 结果。详细比较见 [`docs/p1-seed2-comparison.md`](docs/p1-seed2-comparison.md) 和 [`results/p1-seed2-summary.json`](results/p1-seed2-summary.json)。

| 模式 | 官方 AP | 官方 AP50 | 官方 AP75 | 官方 AR500 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed STAL | 21.7619 | 39.4406 | 20.7115 | 38.1235 | 13.0893 | 31.0922 | 39.4663 |
| adaptive STAL | 21.4915 | 38.9398 | 20.1943 | 37.7670 | 12.7558 | 30.4260 | 36.7081 |
| pure TAL | 21.5139 | 38.6953 | 20.3479 | 37.9989 | 12.7300 | 31.0032 | 42.0973 |

seed 2 的 adaptive-fixed 正式 `ΔAPs = -0.3335` 个百分点。与 seed 1 合并后，两个已完成 seed 的平均 `ΔAPs = -0.2486` 个百分点，且两个 seed 均非正向；这不是最终 P1 判定，因为 seed `20260826` 尚未运行。

每组的原始训练/验证/转换/评测日志和结构化结果均已归档在公开仓库的对应 `results/p1-*-s20260825/` 与 `logs/` 路径下。当前仍需：第三个配对 seed、正式训练期 assigner 正样本统计，以及按需进行 Mosaic-off 精简交互实验。
