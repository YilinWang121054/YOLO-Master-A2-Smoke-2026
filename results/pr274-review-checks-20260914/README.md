# PR #274 合并前复核（2026-09-14）

回应林老师 9 月 13 日 23:02 的[两项确认](https://github.com/Tencent/YOLO-Master/pull/274#issuecomment-5654062468)：默认配置兼容性，以及第三 seed 的结果表。此次只运行 CPU 检查，没有更改功能分支或中断正在进行的 fixed 训练。

## 版本与执行

公开 PR head 为 `2efc4d91d7363d65b6f38e7d47c585231bb98158`，完整 tree 为 `f1ae8461c9108b89d41fee6df4fd95023ab031c1`。本地测试提交 `6f558609882d9cc39379f88c912c9c3c5ec515d6` 的 tree 与之相同；同时核对两个 checkout 的 tracked 文件未修改。比较基点为 PR 的直接上游 `7bfbfd374a6b44720f98366e088f3a962e16321d`，不是不含 #253 修复的原始 P0 基线 acce839。

```powershell
& 'F:/conda-envs/yolo-master/python.exe' -X utf8 scripts/check_pr274_fixed_compatibility.py
```

[运行清单](manifest.json)记录了源码版本、命令和 runner SHA256。脚本在空的独立工作目录执行，隔离 pytest 的权重清理钩子，设置 CPU 2 线程及 `CUDA_VISIBLE_DEVICES=-1`，不占用训练 GPU。输出存在时拒绝覆盖。此前两次同结果的格式检查过程保留在本机 F 盘 diagnostics，不覆盖或改写原日志。

## 检查结果

- `tests/test_default_config_integrity.py`：5 项通过。
- `tests/test_stal_assignment.py`：18 项通过。
- `tests/test_tal_conflict_resolution.py`：1 项通过。
- 合计 24 项，无失败、错误或跳过。原始 [stdout](stdout.log)、[stderr](stderr.log) 和 [JUnit XML](pytest.xml) 保留。
- 原默认 YAML 的 332 个配置项值均未变；新增 8 个 `stal_*` 项，默认 `stal_mode=fixed`。
- 使用 8 个确定性随机种子、随机框/密集重叠框/极小及阈值边界框/空 GT、三组 top-k，分别测试隐式默认与显式 fixed：192 组对照的 960 个输出张量与上游基点逐元素相同，非空 GT 的候选 mask 也相同。原始逐组记录见 [summary.json](summary.json)。

这证明上述 CPU FP32 回归样例的兼容性，不等于穷尽所有输入或证明实际 CUDA/AMP 训练完全等价。此前真实增强 batch 的 AMP 分配/梯度差异仍保留，不被这次小规模单测覆盖。测试 runner 的 Ruff 检查和格式检查通过。

## 第三 seed

seed `20260826` 的 fixed、adaptive、pure TAL 均已完成 120 轮及 548 张验证，三行完整表已经在[研究报告](../../docs/closure-review/研究报告.md)中。此次再次运行报告核验，9 个实验行及 P0 表共 95 个数值单元格均与保存的 JSON 一致，见 [report-table-verification.json](report-table-verification.json)。

该核验不重跑训练或评测。旧实验缺逐轮在线统计、探索 seed 的优化器差异等限制仍然保留；第三 seed adaptive−fixed APs 为 +0.3930 个百分点，不代表满足正式 P1 条件。
