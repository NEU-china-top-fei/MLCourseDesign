# 任务一 植物图像分类

任务使用 500 张训练图像识别五种植物。V2 方案正式公榜 Accuracy 为 0.9841。

## 训练入口

- [V2 完整 Notebook](main/plant_kaggle_v2.ipynb)

## 模型与训练

DINOv2-Large 冻结特征与五折 RBF SVM 完成五分类，草类样本由 DINOv2-Base 和 ConvNeXt-Tiny 二分类专家处理。每个专家增加 64 维受限残差头，使用零初始化、梯度隔离与 KL 约束，logit 修正范围为 ±0.1。

两专家等权融合，以 0.25 为阈值，再结合 RootSIFT 与 RANSAC 几何检索。Notebook 中列出依赖、参数、数据路径和运行步骤。
