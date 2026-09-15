# 图面与数据核查

- Backend：已保存的 Python，复用本项目原始 P0 绘图程序的相同数据结构，只改实验标签，不改数据计算。
- 问题：本组训练中，小/中/大目标冲突后平均正样本数如何随 epoch 变化？本图只描述单组轨迹，不证明 adaptive 优越。
- 映射：epoch → 横轴；after_conflict.mean → 纵轴；small/medium/large → 三条线。全 120 轮、194160 batch，无数据排除。均值由每轮 GT 观测计算，不是 seed 均值。
- 版式：单面板定量曲线，183×95 mm；SVG/PDF 保留文字，PNG 600 dpi。PDF 字号审计共 19 个文字段，最小 7 pt，无小于 5 pt 的字形。
- 图面检查：标题、图例、轴标签和说明无裁切/碰撞；三种颜色辅以不同点形。medium 和 large 接近是原始数据的特征，不为视觉区分修改数值或偏移曲线。
- Source preflight：18 PASS、2 WARN、0 FAIL。PNG 而无 TIFF 的提醒不影响本 GitHub 报告；“无 seed 误差条”提醒已核查，本图只有一个 seed，不能编造不确定性区间。
- 图中所有 120 个点对应完整 JSON；P50/P90、零正样本比例和四阶段分布保留在源 CSV/JSON。末 10 轮关闭 Mosaic 的原协议不等同独立交互实验。

文字更新沿用术语：fixed=现有固定 stride STAL；APs=COCO-style small AP@[.50:.95]、maxDets500；GT观测=真实增强后分配器输入。新完成记录替换“fixed仍在训练”的旧状态，历史九行结果及负结果不删。正文仅增加本组结果和局限，全部哈希及日志放在证据目录。
