# 候选质量诊断图

本图版根据9月13日已公开的固定checkpoint探针记录制作，没有重新推理或训练。输入版本为`ecdd72de51d257f267417d69a1496a6b49df4a45`；两张图共享同一fixed checkpoint和16张预选train图像。图级源数据、输入/输出SHA-256与选例规则见[figure-manifest.json](figure-manifest.json)，原始统计见[上一级说明](../README.md)。

## 图1 | 同一预测下的正样本增量与质量分布

![同一预测下的正样本增量与质量分布](candidate-quality.png)

a，每个蓝点对应一个small GT，横纵轴分别为adaptive相对fixed的最终正样本数量变化、归一化target score之和变化。使用全部1321个small GT，点重叠处未抖动或删减；两个坐标单位不同，不应以y=x判断比例。b、c，adaptive最终选中的8404个small正样本的经验累积分布，灰色实线为属于同一GT的fixed候选集合的5530个位置，橙色虚线为真正新增的2874个位置。两组互斥且覆盖全部8404个位置；“属于fixed候选”不等于“被fixed最终选中”。b表示归一化target score，c表示普通IoU，而非实际排序使用的clamped CIoU。零值和重复值均保留，没有平滑。

small按本次LetterBox后实际FP32面积<1024定义，不是原始验证图像的APs分档。本图为单checkpoint的描述性统计，不提供独立seed重复、误差条、置信区间或显著性检验。源数据：[GT配对差值](paired-small-gt.csv)、[已选候选](selected-small-candidates.csv)。矢量版：[SVG](candidate-quality.svg)、[PDF](candidate-quality.pdf)。

## 图2 | 预先选定的小目标候选位置

![预先选定的小目标候选位置](candidate-geometry.png)

a–c依次为pure TAL、fixed STAL、adaptive STAL，使用相同预测、坐标范围和0–1色标。灰色空心圆表示候选位置，彩色实心圆表示最终正样本，颜色为其实际target score；黑色虚线框是进入assigner但尚未候选扩展的GT。图例中的叉号表示冲突前已选、冲突后丢失的位置，本例没有此类位置。坐标为800×800 LetterBox输入像素，y轴向下，没有显示原图像素。

选例规则在绘图前确定：manifest第一张图像`0000084_01436_d_0000004.jpg`中索引最小的small GT，即GT 3，不依据改善幅度筛选。在这个例子中，pure TAL和fixed均为16个候选、10个最终正样本；adaptive为42个候选、13个最终正样本。此例用于说明几何变化，不代表全体GT的冲突或质量变化。源数据：[三模式候选坐标](geometry-example-candidates.csv)。矢量版：[SVG](candidate-geometry.svg)、[PDF](candidate-geometry.pdf)。

## 复现

从证据仓库根目录运行，输出目录必须尚不存在：

```powershell
python scripts/plot_candidate_quality.py --source results/candidate-quality-20260913 --output <新的输出目录>
```

Python 3.11.15、NumPy 2.4.6、matplotlib 3.11.1。脚本先核对原始记录哈希，再生成图与三份CSV。图1为183×130 mm，图2为183×86 mm，完整PNG为600 dpi，逐面板预览为300 dpi；SVG/PDF保留文字。详细检查见[QA.md](QA.md)。PDF创建时间等元数据可能随复绘变化，应核对源数据和图形内容，不以跨次PDF文件哈希相同作为要求。

这组图显示的是固定预测下的分配行为，不是120轮训练统计的补录；不能单独证明长训APs变化的原因，也不用于认定P1或P2达标。
