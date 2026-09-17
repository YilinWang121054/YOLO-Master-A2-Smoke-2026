# 第二轮筛选：最小扩展候选仍未满足大目标精度约束

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / collect
- Origin Date: 2026-09-17
- Verification Status: ANALYZED（训练、五次评测、在线统计与归档哈希核验；不是独立复训或P1达标证明）
- Version Label: screen_v2_adaptive_minextent_complete_v1

第五组`s4-adaptive-minextent`的20轮训练于9月16日01:50完成。第16～18轮checkpoint已在当日评测，第19轮的首次评测被关机打断；9月17日开机后，队列自动完成第19、20轮评测，最后一次于15:55结束。五次成功评测均覆盖548张验证图像，训练和这些成功评测的退出码均为0。

这组最后五轮平均APs为6.7754，比同预算fixed提高0.3337个百分点。但APl下降2.5261个百分点，超过允许的0.5；large零正样本次数也从3次变为5次。因此仍不符合运行前冻结的筛选条件。最后一组尚未完成，不提前作出六组的最终选择。

## 最后五轮平均指标

差值为adaptive-minextent减fixed，单位为绝对百分点。

| 指标 | fixed | adaptive-minextent | 差值 |
| --- | ---: | ---: | ---: |
| DET AP | 13.2073 | 13.5746 | +0.3674 |
| AP50 | 25.4825 | 26.1392 | +0.6567 |
| AP75 | 12.0288 | 12.4164 | +0.3876 |
| AR500 | 28.8400 | 29.1821 | +0.3421 |
| APs | 6.4417 | 6.7754 | +0.3337 |
| APm | 18.6163 | 18.7791 | +0.1628 |
| APl | 24.5993 | 22.0732 | -2.5261 |
| ARs@500 | 18.9381 | 19.5738 | +0.6356 |

第16～20轮的APs依次为6.6371、6.8404、6.7268、6.7213、6.9513。这里按五轮平均比较，没有选择最高的一轮。

总体指标来自已与官方MATLAB实测对齐的Python DET实现，本候选未单独运行MATLAB。面积指标先按官方规则过滤ignore region，再使用本课题的COCO-style定义：原始验证GT面积small <1024、medium [1024,9216)、large ≥9216，IoU=.50:.95、maxDets=500。这不是VisDrone官方面积分档。

## 真实训练期零正样本统计

第16～20轮增强后、冲突消解后的GT观测合并计数如下。比例由合并分子与分母计算，不是各轮比例的简单平均。

| 面积档 | GT观测数 | fixed零正样本次数 | adaptive-minextent零正样本次数 | fixed比例 | adaptive-minextent比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 1,358,930 | 114,508 | 63,326 | 8.4263% | 4.6600% |
| medium | 221,618 | 46 | 45 | 0.0208% | 0.0203% |
| large | 12,722 | 3 | 5 | 0.0236% | 0.0393% |

small覆盖改善，但本次APs增幅有限，大目标精度损失仍然存在。large只多出2次零正样本观测，不能据此声称稳定退化；不过冻结规则要求各档比例不恶化，需要如实记为未满足。即使不考虑这项小计数差异，APl约束也已经失败。

本组将candidate_scale降到1.0，small top-k保持10；仍有adaptive的最小候选保障，不等于纯TAL。与scale=1.25、k10那组相比，继续缩小扩展系数并未在这次短训中改善APs或APl。这个观察不能直接确定是候选质量、监督权重还是GT冲突导致差异。

五个checkpoint不是五个独立seed，重复出现的GT观测也不是独立样本。这些数据只用于同预算筛选，不支持显著性、正式三seed提升或P1达标结论；也不与120轮训练的前20轮混比。

## 配置与原始证据

源码`52c2befa50706b9dff13b6e0813b19413d9f532d`；v0.1-N、完整6471/548、seed20260825、YAML初始化/pretrained=False、imgsz800、batch4/nbs64、workers0、FP32、显式MuSGD参数；Mosaic=1、close_mosaic=10，面积1024/9216、min_candidates=3、top-k=10/10/10，α/β不变。完整配置和执行脚本均在归档中。

20份真实在线统计覆盖32,360个训练batch，保留各档mean、P50/P90以及候选扩展、冲突消解前后的分布。`last.pt` SHA-256为 `dfdd5643618e558c74dc31ff8da6d3c4171b30808410bb6bbeabeb20fb5e4355`。

- [未舍入指标](../../results/p1-screen-v2-20260915/s4-adaptive-minextent/summary.json)与[同预算fixed](../../results/p1-screen-v2-20260915/s0-fixed/summary.json)
- [训练配置、逐轮指标和在线统计](../../results/p1-screen-v2-20260915/s4-adaptive-minextent/training/)
- [原始启动、退出和stdout/stderr日志](../../results/p1-screen-v2-20260915/s4-adaptive-minextent/original-logs/)
- [五次成功评测、指标和预测压缩包](../../results/p1-screen-v2-20260915/s4-adaptive-minextent/evaluation/)
- [105项文件记录与SHA-256](../../results/p1-screen-v2-20260915/s4-adaptive-minextent/archive-verification.json)
- [冻结配置和选择规则](P1第二轮筛选冻结-20260915.md)

第19轮旧评测没有可核实的退出码，保留原始日志和`interrupted.json`，没有补造成功记录。新评测在独立尝试目录完成。归档保留原始字节，预测gzip经过解压哈希核对；不上传数据图像、GT或权重二进制。

```powershell
& 'F:/conda-envs/yolo-master/python.exe' -X utf8 scripts/archive_p1_screen_case.py --case s4-adaptive-minextent
```

现在已有五组训练和25次评测完成，最后一组继续按冻结队列运行。P0材料与MATLAB对齐已提交、待导师验收；P1和P2仍未完成。
