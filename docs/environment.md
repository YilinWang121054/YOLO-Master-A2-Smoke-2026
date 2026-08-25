# 环境与安装

## 实际运行环境

| 项目 | 版本 |
| --- | --- |
| OS | Windows 10.0.26200, 64-bit |
| Python | 3.11.15 |
| YOLO-Master / ultralytics | 8.4.101，editable source install |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| CUDA runtime | 12.1 |
| cuDNN | 9.1.0 |
| NVIDIA driver | 546.30 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB |
| NumPy | 2.4.6 |
| OpenCV | 5.0.0 |
| PyYAML | 6.0.3 |

## 安装方式

下面是与本次 smoke 相符的 Windows/Conda 安装方式。PyTorch CUDA wheel 应先安装，再把锁定基线的源码装为 editable package。

```powershell
conda create --prefix F:\conda-envs\yolo-master python=3.11 -y
conda activate F:\conda-envs\yolo-master

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/Tencent/YOLO-Master.git F:\src\YOLO-Master
git -C F:\src\YOLO-Master checkout acce839c7e895d6b179de7f7093fa879e237cc7b
python -m pip install -e F:\src\YOLO-Master
```

验证关键版本：

```powershell
python -c "import torch, torchvision, ultralytics; print(torch.__version__); print(torchvision.__version__); print(ultralytics.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本次结果依赖 NVIDIA GPU。其他 CUDA、驱动或 PyTorch 组合可能不会复现同一个 AMP recovery 事件，因此复现报告应同时记录这些版本和 `amp` 最终状态。
