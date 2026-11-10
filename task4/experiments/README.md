# 任务四训练 Notebook

本目录保留最终六成员配方及其必要训练阶段和配置对照。每个实验含 Notebook、Python 源码和 Kaggle 配置。

| 阶段 | Notebook |
| --- | --- |
| 延长 U-Net | [unet-extended](2026task4-unet-extended/training.ipynb) |
| Little W-Net 四折 | [little-wnet-cv](2026task4-little-wnet-cv/training.ipynb) |
| 局部上下文四折 | [context-local](2026task4-context-local/training.ipynb) |
| 上下文全量拟合 | [context-full20](2026task4-context-full20/training.ipynb) |
| FIVES 迁移四折 | [fives-transunet](2026task4-fives-transunet/training.ipynb) |
| FIVES 全量拟合 | [fives-full20](2026task4-fives-full20/training.ipynb) |
| Retina 小块训练 | [retina-patch](2026task4-retina-patch/training.ipynb) |
| Retina 96 像素细化 | [retina-refine](2026task4-retina-refine/training.ipynb) |
| Retina 全量拟合 | [retina-full](2026task4-retina-full/training.ipynb) |

基线 U-Net 入口位于 [notebook/vessel_training.ipynb](../notebook/vessel_training.ipynb)。融合阶段为：上下文成员按 0.30/0.70 与 Little W-Net 概率混合；再按 0.75/0.25 融合 FIVES；最后与 Retina 按 0.50/0.50 融合，并以 0.44 阈值生成掩膜。基线、延长 U-Net 与 Little W-Net 的名义权重分别为 1.40625%、4.21875% 和 5.625%。
