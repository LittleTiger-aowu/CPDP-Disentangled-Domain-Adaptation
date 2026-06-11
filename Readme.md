# MM-CPDP：基于共享–私有表示学习与域对抗迁移的跨项目缺陷预测框架

本仓库为论文提出的跨项目缺陷预测（Cross-Project Defect Prediction, CPDP）框架的官方实现，核心包括：

* 多模态代码表示学习
* 共享–私有特征解耦
* 域对抗迁移学习
* 加性间隔边界优化
* 严格 zero-shot CPDP 实验协议

---

# 📌 项目简介

跨项目缺陷预测（CPDP）的核心难点在于：

* 不同软件项目之间存在明显分布偏移；
* 代码语义与项目特有模式容易发生表示混杂；
* 固定阈值迁移条件下决策边界容易退化。

为解决上述问题，本文提出一种基于：

* CodeBERT 语义表示
* 结构先验建模
* 共享–私有特征解耦
* 域对抗适配
* 加性间隔分类优化

的多模态跨项目缺陷预测框架。

实验基于 AEEEM 数据集，并采用严格 Leave-One-Project-Out（LOOCV）协议进行评估。

---

# ✨ 方法特点

## 1. 多模态代码表示学习

联合建模：

* CodeBERT 语义特征
* 结构先验信息
* 块级上下文传播

提升跨项目代码表示能力。

---

## 2. 共享–私有特征解耦

通过：

* Shared 分支
* Private 分支
* 正交约束

分离：

* 跨项目稳定模式
* 项目特有噪声

减轻负迁移问题。

---

## 3. 域对抗迁移学习

采用：

* Gradient Reversal Layer（GRL）
* Domain Discriminator
* Feature Alignment

实现源域与目标域共享表示对齐。

---

## 4. 加性间隔边界优化

采用 AM-Softmax 优化分类边界，

提升：

* MCC
* F1
* 阈值迁移稳定性

尤其适用于严格 zero-shot CPDP 场景。

---

# 🏗️ 总体框架

整体流程包括：

```text
源码输入
    ↓
TPSM 结构块划分
    ↓
CodeBERT 语义编码
    ↓
结构先验提取
    ↓
多模态融合
    ↓
共享–私有特征解耦
    ↓
共享表示 Z_sh
    ↓
GRL + 域判别器
    ↓
AM-Softmax 分类器
    ↓
跨项目缺陷预测
```

损失函数：

```text
L = L_cls + λ_ortho·L_ortho + λ_dom·L_domain
```

---

# 📂 项目结构

```text
MM-CPDP/
├── src/
│   ├── ch3/                 # 第三章：表示学习模块 ⭐
│   ├── ch4/                 # 第四章：跨项目迁移模块 ⭐
│   ├── rep/                 # 多模态表示组件
│   └── models/              # GRL / Domain Head / AM-Softmax
│
├── data/processed/          # 预处理数据
├── outputs/                 # 实验输出
├── Dataset/                 # AEEEM 数据集
├── environment.yml          # Conda 环境
└── README.md
```

---

# 📊 数据集

实验基于公开 AEEEM 数据集。

包含五个 Java 开源项目：

* Eclipse JDT Core
* Equinox
* Mylyn
* PDE UI
* Lucene

实验协议：

* Leave-One-Project-Out（LOOCV）
* Target-domain unlabeled
* Fixed-threshold transfer

用于严格跨项目缺陷预测评估。

---

# 🔧 环境配置

## 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate LineDef_env
```

---

## 2. 手动安装 PyTorch Geometric（可选）

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

---

## 3. 关键依赖

| 库               | 版本     |
| --------------- | ------ |
| Python          | 3.9    |
| PyTorch         | 2.0    |
| torch-geometric | 2.4    |
| transformers    | latest |
| scikit-learn    | 1.3    |
| pandas          | 1.3    |

---

# 📦 数据准备

## 1. 构建统一数据集

```bash
python src/ch3/prepare_all_data.py \
    --input-dir Dataset/BugPrediction/src \
    --output-parquet data/processed/all_files.parquet
```

---

## 2. 构建项目词汇表

```bash
python src/ch3/prepare_ubd_class_parquet.py \
    --input-dir Dataset/BugPrediction/src \
    --output-vocab data/processed/project_vocab.json
```

---

# 🚀 第三章：表示学习模块（Ch3）

## 1. 训练表示模型

```bash
python src/ch3/train_representation.py \
    --data-parquet data/processed/all_files.parquet \
    --project-vocab data/processed/project_vocab.json \
    --output-dir outputs/ch3_ckpt \
    --epochs 3 \
    --batch-size 4 \
    --d-h 256 \
    --d-sh 128 \
    --d-pr 128 \
    --lambda-pr 1.0 \
    --lambda-ortho 0.1 \
    --codebert-path /path/to/CodeBert \
    --use-amp 1 \
    --grad-accum-steps 4
```

---

## 2. 关键参数说明

| 参数                   | 含义            |
| -------------------- | ------------- |
| `--d-h`              | 隐藏层维度         |
| `--d-sh`             | Shared 子空间维度  |
| `--d-pr`             | Private 子空间维度 |
| `--lambda-pr`        | 项目监督损失权重      |
| `--lambda-ortho`     | 正交约束权重        |
| `--use-amp`          | 混合精度训练        |
| `--grad-accum-steps` | 梯度累积步数        |

---

## 3. 提取共享表示

```bash
python src/ch3/dump_representations.py \
    --model-path outputs/ch3_ckpt/checkpoint.pt \
    --data-parquet data/processed/all_files.parquet \
    --output-dump outputs/ch3_repre \
    --batch-size 8
```

输出：

```text
outputs/ch3_repre/repr.npz
outputs/ch3_repre/meta.jsonl
```

---

## 4. 表示质量分析

### 有效秩与 CORAL 距离

```bash
python src/ch3/compute_repr_metrics.py \
    --dump-dir outputs/ch3_repre \
    --output-file outputs/ch3_metrics.json
```

---

### t-SNE 可视化

```bash
python src/ch3/plot_tsne.py \
    --dump-dir outputs/ch3_repre \
    --output-dir outputs/ch3_tsne_figs
```

---

## 5. 消融实验

```bash
python src/ch3/run_ablation_suite.py \
    --data-parquet data/processed/all_files.parquet \
    --project-vocab data/processed/project_vocab.json \
    --output-dir outputs/ch3_ablation \
    --epochs 3
```

---

## 6. 留一法交叉验证

```bash
python src/ch3/run_loocv_ch3_pretrain.py \
    --data-parquet data/processed/all_files.parquet \
    --project-vocab data/processed/project_vocab.json \
    --output-dir outputs/ch3_loocv \
    --epochs 3
```

---

# 🚀 第四章：跨项目迁移模块（Ch4）

## 1. 训练 CPDP 模型

```bash
python src/ch4/train_cpdp_adapt_cached.py \
    --dump-dir outputs/ch3_repre \
    --source-projects "lucene-2.4,Eclipse_JDT_Core-3.4" \
    --target-project "Mylyn-3.1" \
    --output-dir outputs/ch4_cpdp \
    --d-sh 128 \
    --dom-hidden 64 \
    --lambda-dom 0.5 \
    --lr-head 1e-3 \
    --epochs 50 \
    --batch-size 32 \
    --threshold-mode mcc \
    --cached 1
```

---

## 2. 关键参数说明

| 参数                 | 含义       |
| ------------------ | -------- |
| `--lambda-dom`     | 域对抗损失权重  |
| `--dom-hidden`     | 域判别器维度   |
| `--threshold-mode` | 阈值选择策略   |
| `--cached`         | 是否使用缓存表示 |

---

## 3. LOOCV 实验

```bash
python src/ch4/run_loocv_cpdp.py \
    --dump-dir outputs/ch3_repre \
    --output-dir outputs/ch4_loocv \
    --epochs 50 \
    --batch-size 32
```

---

## 4. 多随机种子实验

```bash
python src/ch4/run_all_methods_all_seeds.py \
    --dump-dir outputs/ch3_repre \
    --output-dir outputs/ch4_all_methods \
    --seeds "1337,42,1234,5678,9999"
```

---

## 5. 消融实验

```bash
python src/ch4/run_ablation_all_projects.py \
    --dump-dir outputs/ch3_repre \
    --output-dir outputs/ch4_ablation \
    --epochs 50
```

包括：

* 无 GRL
* 无域判别器
* 无正交约束
* 无 GCN
* 仅语义特征

---

# 📈 评估指标

采用：

* ROC-AUC
* PR-AUC
* MCC
* F1
* Precision / Recall

其中：

* ROC-AUC 用于衡量排序能力；
* MCC 与 F1 用于评估固定阈值迁移稳定性。

---

# 🔁 可复现实验

本仓库提供：

* 完整训练脚本
* 消融实验
* 多随机种子实验
* LOOCV 配置
* 表示缓存
* 统计分析脚本

用于复现实验结果。

---

# 🛠️ 常见问题

## 1. 显存不足

建议：

* 减小 batch-size
* 增加 grad-accum-steps
* 启用 AMP
* 冻结 CodeBERT

---

## 2. CodeBERT 路径配置

```bash
git clone https://huggingface.co/microsoft/codebert-base
```

训练时指定：

```bash
--codebert-path /path/to/codebert-base
```

---

## 3. 随机种子复现

```bash
--seed 1337
```

建议：

```bash
--seeds "1337,42,1234,5678,9999"
```

---


---

# 📄 License

本项目仅用于学术研究。

---

# 👥 联系方式

如有问题欢迎提交 Issue。
