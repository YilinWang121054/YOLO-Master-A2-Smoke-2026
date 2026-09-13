# P0 官方 DET：MATLAB 实测对齐

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run / runtime comparison
- Origin Date: 2026-09-13
- Verification Status: VERIFIED（限定为以下输入上的实际跨语言指标对齐，不是最终验收决定）
- Version Label: p0_matlab_alignment_v1

北京时间 2026 年 9 月 13 日 21:44，MATLAB R2024a Update 7 成功完成原始 P0 的 548 张验证集 DET 评测，退出码 0，stderr 为空。未改动官方工具源码。

| 指标 | 官方 MATLAB（%） | Python 源码移植（%） | 绝对差值（百分点） |
|---|---:|---:|---:|
| AP | 21.700296590048197 | 21.700296590049476 | 1.2790e-12 |
| AP50 | 39.36059518273365 | 39.36059518273381 | 1.6342e-13 |
| AP75 | 20.608374220452735 | 20.60837422045259 | 1.4566e-13 |
| AR500 | 38.21937074090097 | 38.2193707409014 | 4.2633e-13 |

四项总体指标最大差异为 1.28×10⁻¹² 个百分点，满足老师规定的 <0.01。AR1/10/100 也已逐项比较并满足同一容差。本次证明的是固定 P0 预测集上的数值对齐，不声称所有可能输入均已实测，也不把 P1 每组结果改称 MATLAB 直接输出。

## 可复核的输入与命令

- 原始 P0 代码 `acce839c7e895d6b179de7f7093fa879e237cc7b`；完整 train 120e 和统计见[既有 P0 材料](../../results/closure-evaluation/p0-locked-s20260824-stats120/)。
- P0 last.pt SHA256：`3d15a0b7f9562f18850766284c601394e970eac9607524bc2ad91c3ec8f4ab87`。
- 官方工具：`VisDrone/VisDrone2018-DET-toolkit@005445782213e20cb91bc50a597db3dd949e749a`。
- 所有 `.m` 文件逐字节对 Git blob 核验（仅允许 CRLF/LF 差别）；548 份 GT 和 DET 输入与 9 月 11 日的[输入哈希](../../results/det-source-alignment-20260911/input-sha256.json)一致。评测前后再次检查源文件和输入未变。
- Python 对照来自原始 P0 的 `det-source-metrics.json`，不是另一次 MATLAB 结果。

实际调用：

```powershell
& 'F:/conda-envs/yolo-master/python.exe' -X utf8 scripts/run_p0_matlab_alignment.py --output 'F:/YOLO-Master-A2-P1/diagnostics/p0-matlab-official-20260913'
```

完整 MATLAB 命令、脚本哈希、运行时间和七项精确差值见[comparison.json](../../results/p0-matlab-official-20260913/comparison.json)；[原始 stdout](../../results/p0-matlab-official-20260913/stdout.log)、[stderr](../../results/p0-matlab-official-20260913/stderr.log)、[MATLAB 原始 JSON](../../results/p0-matlab-official-20260913/matlab-metrics.json)均保留。复跑时必须使用新的输出目录，旧证据不可覆盖。

## 关于今天的启动故障

早先 batch 启动报 5201，Service Host 日志包含同名服务无法注册。按 MathWorks 官方服务重建流程，先将 ServiceHost/MATLABConnector 保留为备份；重建时出现 5202，两个官方签名的重装器均未能取得目标安装包。随后将原目录原样恢复，没有删除许可证或凭据。用户打开本机 MATLAB 后，再次 batch 调用成功。这个先后关系不足以认定故障只有一个根因，因此不把“重装成功”写成解决经过。

相关官方说明：[5201](https://www.mathworks.com/matlabcentral/answers/1815395-why-do-i-receive-error-5201-unable-to-access-services-required-to-run-matlab)、[Service Host 重装](https://www.mathworks.com/matlabcentral/answers/1815365-how-do-i-uninstall-and-reinstall-the-mathworks-service-host)。失败检查日志仍留在本机 diagnostics，账号及服务内部日志不公开。

## 登记建议

**P0 材料及官方 MATLAB 实测对齐已补齐、已提交，待导师验收；PR #274 已提交，待 review。**

原来的“评测对齐待补”已经不再是当前状态。P1 的同协议三 seed、完整在线统计、Mosaic 交互和提升指标仍需继续，不能由本次 P0 评测代替。
