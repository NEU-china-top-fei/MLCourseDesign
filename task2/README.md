# 任务二 面部表情识别

输入为 48×48 灰度人脸，训练集 28,275 张，测试集 7,065 张，识别六类表情。评价指标为 Accuracy。

## 实验分支

| 阶段 | Notebook | 方法 | 公榜准确率 |
| --- | --- | --- | --- |
| 基线 | [ResNet18](notebook/fer_gpu.ipynb) | ImageNet 初始化与分组验证 | 0.702 |
| 骨干改进 | [ConvNeXt-Tiny](notebook_improved/fer_improved.ipynb) | 分层学习率、EMA 与正则化 | 0.718 |
| 残差骨干 | [ResNet50](notebook_resnet50/fer_resnet50.ipynb) | 分组验证 | 0.716 |
| 分组集成 | [三折多模型](notebook_cv/fer_cv.ipynb) | ConvNeXt-Tiny 与 ResNet50 | 0.729 |
| 迁移学习 | [多骨干融合](notebook_transfer/fer_transfer.ipynb) | ConvNeXt V2 Tiny、EfficientNetV2-S 与旧集成 | 0.738 |
| 注意力对照 | [局部注意力](notebook_attention/fer_attention.ipynb) | ConvNeXt V2 空间池化注意力 | 与基线持平 |

## 数据划分与训练

相同像素内容的图像放在同一分组中进行分层交叉验证，避免重复样本分散到训练与验证两侧。迁移学习分支使用 192 像素输入、AdamW、预热、EMA、mixup 与早停，并比较原图、水平翻转和轻微中心裁剪。最终融合权重为旧集成 0.50、ConvNeXt V2 0.25、EfficientNetV2 0.25，正式公榜准确率为 0.738。

注意力分支在 ConvNeXt V2 上增加两个空间池化头，并与相同训练预算的继续微调对照。最终采用迁移学习融合方案。

## 运行

每个实验目录包含 Notebook、Python 源码和 Kaggle 配置。配置文件记录竞赛数据、父任务输出和模型附件。运行前请阅读[运行指南](../docs/运行指南.md)。

预训练来源：[ConvNeXt V2 模型卡](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)、[EfficientNetV2 模型卡](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k)。
