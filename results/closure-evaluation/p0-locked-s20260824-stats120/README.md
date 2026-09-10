# P0原始基线评测证据

原始源码acce839，v0.1-N，VisDrone train/val=6471/548，120轮，主checkpoint为第120轮last.pt。配置、checkpoint与预测哈希见evaluation-manifest.json。

- `det-source-metrics.json`：MATLAB源码的Python审计移植结果，不是已完成的原MATLAB运行证明。
- `coco-area-metrics.json`：官方ignore规则源码移植过滤后，COCO-style半开面积分档补充。
- `assignment/`、`figures/`：120轮在线统计，曲线、源数据和QA说明。
- `training-log-manifest.json`：原始stdout/stderr及恢复元数据的归档位置和SHA256。
- `predictions-det-548.zip`：548个与验证图像同名的DET TXT，尚未做ignore过滤，可用于原官方工具独立复评。
- `predictions.json.gz`：原始推理JSON的无损压缩副本；解压SHA256应等于evaluation-manifest.json中的predictions_sha256。
- `prediction-archive-manifest.json`：两个压缩包及每份TXT的校验值。已核查548个文件名与原始val注释文件名集合完全相同。

图像和GT不随归档发布，请从VisDrone原发布方获取完整DET val。在有可用MATLAB许可的环境中，可使用仓库 `scripts/run_visdrone_official_matlab.m` 调用原devkit工具；输入为原验证集目录及解压后的DET目录。此包装脚本本身不构成成功运行的证明。

证据目录通过.gitattributes关闭Git文本换行转换，原始回车和ANSI终端控制符不清洗。136份关键文件（12份日志/元数据、120份统计、报告、3份实际评测脚本）已核查Git索引与记录字节一致。旧提交继续保留；本次没有重新生成或裁剪训练日志。
