# P1 前置检查：真实训练 batch 的 FP32 / AMP 差异

固定代码 52c2bef，fixed seed20260826 的同一个 last.pt；沿用先前哈希预选样本中的前4张 train 图像，batch=4、800像素。分别使用真实 Mosaic off/on 增强；同一个增强后 batch 在三模式下各做 FP32/FP16 autocast 前向、反向，模型处于 train 模式，每次恢复相同 state_dict 与 RNG，不执行 optimizer.step。配置在运行前提交于 756470d。

| Mosaic | 模式 | FP32/AMP 正样本数 | 前景 mask 不同数 | loss 相对差异 | 梯度相对 L2 差异 |
|---|---|---|---:|---:|---:|
| off | TAL | 1402 / 1405 | 19 | 0.2796% | 15.3002% |
| off | fixed | 1445 / 1447 | 10 | 0.4115% | 11.1067% |
| off | adaptive | 2226 / 2227 | 7 | 0.2518% | 10.7723% |
| on | TAL | 1874 / 1874 | 10 | 0.3967% | 10.4714% |
| on | fixed | 2199 / 2200 | 11 | 0.3950% | 9.9728% |
| on | adaptive | 3594 / 3596 | 12 | 0.3139% | 9.0247% |

六组 loss、梯度均有限，候选区域 mask 相同，但冲突前 top-k、最终 mask 或 GT 归属存在差异。正样本总数相同也不意味着选中了同一组网格。梯度差异超过预设5%提示阈值，不能写成 AMP 完全一致或安全性检查全部通过。正式第二轮暂时继续FP32，AMP差异需进一步定位，必要时补短训比较。

原始汇总见 [summary.json](summary.json)，分配 mask 的 NPZ 和六组独立计数核验见 [verification.json](verification.json)。梯度有限值由运行时断言检查；本次没有保存完整梯度张量用于独立重算。反向使用固定 loss scale=128，不代表完整训练中动态 GradScaler。增强后输入张量及其完整 SHA256 保存在 F 盘 diagnostics，未公开上传图像。

这是2个实际 batch的前置诊断，不是完整AMP训练对照，也不是原120轮统计的补录，不支持AP提升或因果结论。单checkpoint和小样本局限均保留。

复核时发现 P0 observer 测试与 P1 probe 测试在同一 pytest 进程中导入不同版本的 ultralytics，会发生模块缓存冲突。分开进程运行后分别4项和10项通过；这属于测试隔离问题，不能称为算法bug。
