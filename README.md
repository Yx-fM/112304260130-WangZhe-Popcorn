# 机器学习实验：基于 Word2Vec 的影评情感预测

## 1. 学生信息

- **姓名**: 王喆
- **学号**: 112304260130
- **班级**: 数据1231

> 注意：请将学号姓名填写正确，否则本次实验提交无效！

---

## 2. 实验简介

本实验基于 IMDb 电影评论数据，使用 **Word2Vec 将文本转换为向量表示**，结合 **机器学习模型** 完成情感二分类任务，并将结果提交到 Kaggle 竞赛平台进行评测。

实验主要任务：
- 文本预处理（HTML 清洗、分词）
- Word2Vec 词向量训练与表示
- 机器学习分类器训练与预测
- Bag of Centroids 特征融合与 XGBoost 优化

数据集：
- 训练集：25,000 条 labeled 评论
- 无标签数据：50,000 条 unlabeled 评论（用于 Word2Vec 训练）
- 测试集：25,000 条评论

---

## 3. 实验与提交信息

- **竞赛名称**: Bag of Words Meets Bags of Popcorn
- **竞赛链接**: https://www.kaggle.com/competitions/word2vec-nlp-tutorial
- **提交文件**: `submissions/part4_fused_final.csv`

- **GitHub 仓库地址**: https://github.com/你的用户名/112304260130-WangZhe-Popcorn

---

## 4. Kaggle 成绩

请填写你提交到 Kaggle 的成绩：

- **Public Score**: 0.XXXX
- **Private Score**: 0.XXXX（可选）
- **排名**: （可选，如果排名很靠前可以填写）

### 各方法成绩对比

| 方法 | 验证集准确率 | Kaggle Public Score |
|------|-------------|---------------------|
| Part 1 (BoW Baseline) | ~86-87% | 待填写 |
| Part 3 (Word2Vec + RF) | ~88-89% | 0.86680 |
| Part 4 (特征融合+XGBoost) | CV AUC ~0.90-0.93 | 0.87948 |

---

## 5. 实验方法

### Part 1: Bag of Words 基线

使用词袋模型 + 随机森林建立基线。

**核心代码**: `src/models/train_bow.py`

**关键参数**:
- 特征数：5000
- 随机森林树数：100
- 最大深度：10

### Part 2: 句子分词与数据准备

使用 NLTK 进行句子分词，为 Word2Vec 训练准备数据。

**核心代码**: `src/preprocessing/prepare_sentences.py`

**输出**:
- `data/processed/train_sentences.pkl`
- `data/processed/all_sentences.pkl`

### Part 3: Word2Vec 词向量训练

使用 Gensim 训练 Word2Vec 模型，采用词向量平均方法生成句子表示。

**核心代码**: `src/models/train_word2vec.py`

**参数**:
- 向量维度：300
- 最小词频：40
- 上下文窗口：10
- 训练算法：Skip-gram (sg=1)

### Part 4: 特征融合与 XGBoost 优化

融合多种特征并使用 XGBoost 进行预测。

**核心代码**: `run_part4_final.py`

**特征**:
- BoW (5000 维)
- Word2Vec 平均向量 (300 维)
- Bag of Centroids (5000 维)

**优化策略**:
- 5 折交叉验证
- L1+L2 正则化
- 特征采样 (subsample=0.8, colsample_bytree=0.8)
- 树深度限制 (max_depth=6)

---

## 6. 使用方法

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行 Part 1 (BoW Baseline)
cd src/models
python train_bow.py

# 3. 运行 Part 2 (准备数据)
cd ../preprocessing
python prepare_sentences.py

# 4. 运行 Part 3 (Word2Vec)
cd ../models
python train_word2vec.py

# 5. 运行 Part 4 (特征融合)
cd ../../
python run_part4_final.py
```

### Jupyter Notebook

```bash
jupyter notebook
```

然后依次执行：
- `notebooks/01_part1_bag_of_words.ipynb`
- `notebooks/02_part2_word_vectors.ipynb`
- `notebooks/03_part3_word2vec.ipynb`

---

## 7. 项目结构

```
112304260130-WangZhe-Popcorn/
├── README.md                    # 实验报告
├── requirements.txt             # 依赖列表
├── QUICKSTART.md                # 快速开始指南
├── run_part3.py                 # Part 3 运行脚本
├── run_part3_gpu.py             # GPU 加速版本
├── run_part4_final.py           # Part 4 最终版本
├── run_part4_v2.py              # Part 4 变体
├── notebooks/                   # Jupyter Notebooks
│   ├── 01_part1_bag_of_words.ipynb
│   ├── 02_part2_word_vectors.ipynb
│   └── 03_part3_word2vec.ipynb
├── src/                         # 源代码
│   ├── models/
│   │   ├── train_bow.py
│   │   └── train_word2vec.py
│   └── preprocessing/
│       └── prepare_sentences.py
├── data/                        # 数据目录
│   ├── labeledTrainData.tsv
│   ├── unlabeledTrainData.tsv
│   ├── testData.tsv
│   └── processed/
├── models/                      # 训练好的模型
│   └── word2vec_model.bin
└── submissions/                 # Kaggle 提交文件
    ├── bow_*.csv
    ├── word2vec_xgb_*.csv
    └── part4_fused_*.csv
```

---

## 8. 依赖

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- nltk >= 3.8.0
- gensim >= 4.3.0
- xgboost
- jupyter >= 1.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

---

## 9. 常见问题

### Q1: NLTK 下载失败
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Q2: BeautifulSoup 报错
```bash
pip install beautifulsoup4
```

### Q3: 内存不足
- Part 1: 减少 `max_features` 参数（如改为 3000）
- Part 3: 减少 Word2Vec 的 `vector_size`（如改为 100）

---

## 10. 参考资料

1. Kaggle 竞赛页面：https://www.kaggle.com/competitions/word2vec-nlp-tutorial
2. Word2Vec 原论文：Mikolov et al. (2013) "Efficient Estimation of Word Representations in Vector Space"
3. 实验模板文件：`readme_机器学习实验 2 模板.md`
