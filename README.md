# 机器学习课程设计源码

本仓库收录四项任务的训练 Notebook、配套 Python 源码、Kaggle 运行配置和说明文档。Notebook 按 Kaggle 环境整理，运行前请阅读各任务说明与[运行指南](docs/运行指南.md)。

## 任务入口

| 任务 | 方法 | 训练入口 |
| --- | --- | --- |
| 任务一 植物分类 | DINOv2 特征与草类专家 | [V2 Notebook](task1/main/plant_kaggle_v2.ipynb) |
| 任务二 表情识别 | 迁移学习与概率融合 | [任务二说明](task2/README.md) |
| 任务三 中央凹定位 | V25 父模型与 V26 方案 B | [V26 Notebook](task3/kaggle_v26/train.ipynb) |
| 任务四 血管分割 | 六成员概率融合 | [任务四说明](task4/README.md) |

## 目录

- task1/：植物分类训练入口与说明。
- task2/：表情识别各训练阶段 Notebook 与源码。
- task3/：V25/V26 Notebook、所需基础引擎与设计说明。
- task4/：最终融合成员及必要对照的训练 Notebook。
- docs/：运行、结果、来源和目录说明。
