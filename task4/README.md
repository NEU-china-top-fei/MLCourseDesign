# 任务四 眼底血管分割

使用 20 张训练眼底图学习血管分割，预测 20 张测试图。评价指标为 Dice，提交格式为 20 行 Id,Predicted 的 RLE 编码。最终六成员融合正式公榜 Dice 为 0.85318。

## 融合方案

六成员名义权重为：残差 U-Net 1.40625%、延长 U-Net 4.21875%、Little W-Net 5.625%、上下文模型 26.25%、FIVES 迁移模型 12.5%、Retina 模型 50%，阈值为 0.44。

融合按阶段执行：Little W-Net 与上下文全量模型按 0.30/0.70 混合；再按 0.75/0.25 加入 FIVES；最后按 0.50/0.50 与 Retina 融合。概率中间结果采用 float16 存储，最终阈值计算使用 float32。

## 训练入口

| 成员或训练阶段 | Notebook |
| --- | --- |
| 残差 U-Net 基线 | [vessel_training.ipynb](notebook/vessel_training.ipynb) |
| 延长 U-Net | [unet-extended](experiments/2026task4-unet-extended/training.ipynb) |
| Little W-Net 四折 | [little-wnet-cv](experiments/2026task4-little-wnet-cv/training.ipynb) |
| 局部上下文四折 | [context-local](experiments/2026task4-context-local/training.ipynb) |
| 上下文全量拟合 | [context-full20](experiments/2026task4-context-full20/training.ipynb) |
| FIVES 迁移四折 | [fives-transunet](experiments/2026task4-fives-transunet/training.ipynb) |
| FIVES 全量拟合 | [fives-full20](experiments/2026task4-fives-full20/training.ipynb) |
| Retina 小块训练 | [retina-patch](experiments/2026task4-retina-patch/training.ipynb) |
| Retina 96 像素细化 | [retina-refine](experiments/2026task4-retina-refine/training.ipynb) |
| Retina 全量拟合 | [retina-full](experiments/2026task4-retina-full/training.ipynb) |

## 数据与评价

标签文件扩展名为 .jpg，内容为二值 GIF89a 图像。按官方流程双线性缩放至 512×512 后取大于 0 的像素作为目标。四折在原图级划分，每折 15 张训练、5 张验证。最终 Retina 与上下文模型使用全部 20 张训练图拟合。RLE 按行展开，起点从 1 开始。
