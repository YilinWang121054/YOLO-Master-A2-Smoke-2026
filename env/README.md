# A2 实验环境

Windows；Intel i7-11800H；RTX 3060 Laptop GPU 6 GB。

训练运行时：`F:/conda-envs/yolo-master/python.exe`，Python 3.11.15、PyTorch 2.5.1+cu121；框架源码版本和依赖实测见 `results/closure-audit-20260909.json`。训练源码使用冻结的 A2 分支，P0 原始基线另用 `YOLO-Master-baseline` 的 acce839。

不需要为了结题更新 PyTorch 或模型版本。CPU 采集器烟测和审阅测试设置 CUDA_VISIBLE_DEVICES=-1，可与 GPU 训练并行。正式训练仍固定 batch=4、workers=0、FP32；nbs=64，不能为赶进度临时改 batch。

MATLAB R2024a 已安装，但 2026-09-09 的运行检查报 MathWorks 5201。官方 devkit 文件本地已存在，严格对齐仍缺一次成功的原工具运行。
