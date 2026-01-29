# 需求文档：VQ‑VAE 压缩金融热图 + XGBoost 分类

## 背景与目标
- 将金融热图（WebDataset 中的 `input.npy`）压缩为潜变量特征。
- 使用 XGBoost 在潜变量上进行监督分类（2 类或 3 类）。
- 目标是评估“VQ‑VAE 表示 + XGBoost”在金融热图分类任务上的有效性。

## 现状与约束
- 项目基于 PyTorch，已有 VQ‑VAE 与 PixelCNN 实现。
- 数据集位于 `data/`，为外部导入，不纳入版本控制。
- WebDataset 格式见 `VQVAE_DATASET_GUIDE.md`。

## 范围（Scope）
1) 数据读取
   - 支持 WebDataset 分片：`data/train-*.tar.gz`, `data/val-*.tar.gz`, `data/test-*.tar.gz`
   - 读取 `input.npy`，shape `(2, 101, 160)`，dtype 多为 `float16`
   - 标签字段从样本键中读取（待明确）

2) 表示学习（VQ‑VAE）
   - 训练仅使用训练集，严格避免时间泄漏
   - 训练结束后对 train/val/test 进行编码并导出潜变量

3) 特征构建（用于 XGBoost）
   - 支持至少一种特征形式：
     - `z_e`（量化前连续潜变量）flatten
     - 或 `z_q`（量化后）flatten
     - 或 码本索引统计（bag‑of‑codes）
   - 允许配置特征方案（便于对比实验）

4) 监督分类（XGBoost）
   - 基于潜变量特征训练分类器
   - 使用时间顺序切分或给定 train/val/test 分片

## 非目标（Non‑Goals）
- 不在此阶段优化 VQ‑VAE 架构或做大规模超参搜索
- 不引入 PixelCNN 训练流程
- 不实现在线/实时推理系统

## 评估指标
- 分类指标：Accuracy、F1（宏/微）、AUC（若为二分类）
- 对比基线：
  - 直接使用原始热图训练轻量 CNN（可选，若时间允许）
  - 不量化的简单自编码器（可选，若时间允许）

## 交付物
- 数据读取与标签解析代码
- VQ‑VAE 训练与编码导出脚本
- XGBoost 训练脚本与评估输出
- 简要实验记录（参数与结果）

## 风险与注意事项
- 潜变量可能不保留标签相关信息，需实证验证
- 数据非平稳，必须采用时间顺序切分
- 必须避免归一化/统计信息泄漏
- 类别不平衡需在 XGBoost 中处理（如 `scale_pos_weight`）

## 需要确认的问题（Open Questions）
1) 任务标签字段名（2 类/3 类）具体是哪一个？
2) train/val/test 的时间切分规则是什么？是否已有固定分片？
3) 优先使用哪种潜变量特征：`z_e` / `z_q` / 码本统计？
4) 是否需要保存潜变量为文件（如 `.npy` / `.npz`）？
5) 评估指标的首要目标是什么（如 F1 / AUC / Accuracy）？

