# 快速开始指南

## 实验环境配置

### 1. 安装依赖

```bash
cd Q:\All_Items\PythonProjects\Popcorn

# 创建虚拟环境（可选但推荐）
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import pandas, sklearn, nltk, gensim, bs4; print('✓ 所有依赖已安装')"
```

---

## 执行流程

### Part 1: Bag of Words（词袋模型）- Baseline

**目标**: 快速建立 baseline，预期准确率 ~86-87%

#### 方法 A: Jupyter Notebook（推荐）

1. 打开 Jupyter:
   ```bash
   jupyter notebook
   ```
2. 打开 `notebooks/01_part1_bag_of_words.ipynb`
3. 按顺序执行所有单元格

#### 方法 B: Python 脚本

```bash
cd src/models
python train_bow.py
```

**输出**:
- 验证集准确率：约 86-87%
- 提交文件：`../submissions/bow_YYYYMMDD_HHMMSS.csv`

---

### Part 2: 句子分词与数据准备

**目标**: 为 Word2Vec 训练准备句子级别的数据

#### 方法 A: Jupyter Notebook

1. 打开 `notebooks/02_part2_word_vectors.ipynb`
2. 执行所有单元格

#### 方法 B: Python 脚本

```bash
cd src/preprocessing
python prepare_sentences.py
```

**输出**:
- `../data/processed/train_sentences.pkl`
- `../data/processed/all_sentences.pkl`

---

### Part 3: Word2Vec 词向量训练与分类

**目标**: 使用深度学习词向量提升准确率，预期 ~88-89%

#### 方法 A: Jupyter Notebook

1. 打开 `notebooks/03_part3_word2vec.ipynb`
2. 执行所有单元格

**包含两个阶段**:
- **Part 3A**: 训练 Word2Vec 模型 (300 维向量)
- **Part 3B**: 词向量平均 + 随机森林分类

#### 方法 B: Python 脚本

```bash
cd src/models
python train_word2vec.py
```

**注意**: Part 3 依赖 Part 2 的输出文件，请先运行 Part 2！

**输出**:
- 训练好的模型：`../models/word2vec_model.bin`
- 提交文件：`../submissions/word2vec_YYYYMMDD_HHMMSS.csv`

---

## 完整运行流程

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
```

---

## Kaggle 提交流程

1. 访问竞赛页面：https://www.kaggle.com/competitions/word2vec-nlp-tutorial

2. 点击 "Submit Predictions"

3. 上传 `submissions/` 目录下的 CSV 文件

4. 查看成绩（Public Score）

5. 记录成绩到实验报告

### 预期成绩对比

| 方法 | 验证集准确率 | Kaggle Public Score |
|------|-------------|---------------------|
| Part 1 (BoW) | ~86-87% | ~85-87% |
| Part 3 (Word2Vec) | ~88-89% | ~87-89% |

---

## 文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `notebooks/01_part1_*.ipynb` | Part 1 Jupyter Notebook |
| `notebooks/02_part2_*.ipynb` | Part 2 Jupyter Notebook |
| `notebooks/03_part3_*.ipynb` | Part 3 Jupyter Notebook |
| `src/models/train_bow.py` | Part 1 Python 脚本 |
| `src/models/train_word2vec.py` | Part 3 Python 脚本 |
| `src/preprocessing/prepare_sentences.py` | Part 2 Python 脚本 |
| `submissions/` | Kaggle 提交文件 |
| `data/processed/` | 处理后的数据 |
| `models/` | 训练好的模型 |

---

## 常见问题

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

### Q4: Part 3 报错找不到 Part 2 的数据
确保先运行 Part 2，生成 `data/processed/` 目录下的 pickle 文件。

---

## 下一步

完成 Part 1-3 后：
- ✅ 获得 BoW baseline 成绩（~87%）
- ✅ 获得 Word2Vec 成绩（~88-89%）
- ✅ 掌握 NLP 基础流程
- ⏭️ 继续 Part 4 (选做): 特征融合/深度学习
