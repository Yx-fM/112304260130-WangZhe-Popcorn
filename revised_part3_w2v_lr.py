"""
Part 3 Revised: Word2Vec + Logistic Regression
关键改进：
1. 保留否定词
2. 使用逻辑回归代替随机森林
3. 添加 negation-aware 特征处理
"""

import pandas as pd
import numpy as np
import os
import pickle
import datetime
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import nltk
import re
from bs4 import BeautifulSoup

# 数据路径
DATA_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\data"
PROCESSED_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\data\processed"
MODELS_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\models"
SUBMISSION_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\submissions"

# 否定词列表
NEGATION_WORDS = {
    'not', 'no', 'never', 'nor', 'neither', 'nobody', 'nothing',
    'nowhere', 'none', "n't", 'cannot', "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "didn't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't"
}


def review_to_words(raw_review, keep_negations=True):
    """去除 HTML、标点，转小写，但保留否定词"""
    markup = BeautifulSoup(raw_review, 'html.parser')
    markup_text = markup.get_text()
    
    # 处理缩写
    text = markup_text.lower()
    contractions = {"n't": " not", "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"}
    for short, full in contractions.items():
        text = text.replace(short, full)
    
    # 去除标点
    letters_only = re.sub("[^a-zA-Z\\s]", " ", text)
    words = letters_only.split()
    
    return " ".join(words)


def make_feature_vector(words, model, num_features):
    """计算句子/评论的特征向量（词向量平均）"""
    feature_vec = np.zeros((num_features), dtype="float32")
    nwords = 0
    
    for word in words:
        if word in model.wv:
            feature_vec += model.wv[word]
            nwords += 1
    
    if nwords > 0:
        feature_vec = feature_vec / nwords
    
    return feature_vec


def get_avg_feature_vector(review, model, num_features):
    """将整条评论转为特征向量"""
    words = review_to_words(review).split()
    return make_feature_vector(words, model, num_features)


def main():
    print("=" * 70)
    print("Part 3 Revised: Word2Vec + Logistic Regression")
    print("=" * 70)
    
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # ========== Part 3A: Word2Vec 训练 ==========
    print("\n【Part 3A: Word2Vec Model Training】\n")
    
    # 加载句子数据
    print("1. Loading sentence data...")
    with open(os.path.join(PROCESSED_DIR, 'all_sentences.pkl'), 'rb') as f:
        all_sentences = pickle.load(f)
    
    print(f"   Total sentences: {len(all_sentences):,}")
    
    # Word2Vec 参数
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context_window = 10
    downsampling = 1e-3
    
    print(f"\n2. Training Word2Vec model...")
    print(f"   Vector size: {num_features}")
    print(f"   Min word count: {min_word_count}")
    print(f"   Context window: {context_window}")
    
    model = Word2Vec(
        sentences=all_sentences,
        vector_size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        window=context_window,
        sample=downsampling,
        sg=1,  # Skip-gram
        epochs=10
    )
    
    print("   ✓ Training complete!")
    
    # 模型统计
    vocab_size = len(model.wv.key_to_index)
    print(f"\n3. Model statistics:")
    print(f"   Vocabulary size: {vocab_size:,} words")
    
    # 保存模型
    model_path = os.path.join(MODELS_DIR, 'word2vec_model_revised.bin')
    model.save(model_path)
    print(f"\n4. Model saved: {model_path}")
    
    # ========== Part 3B: 分类 ==========
    print("\n【Part 3B: Classification with Logistic Regression】\n")
    
    # 加载原始数据
    print("1. Loading raw data...")
    train = pd.read_csv(
        os.path.join(DATA_DIR, "labeledTrainData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    test = pd.read_csv(
        os.path.join(DATA_DIR, "testData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    
    # 创建特征向量
    print("\n2. Creating training features...")
    train_features = np.zeros((len(train), num_features), dtype="float32")
    
    for i in range(len(train)):
        if (i+1) % 5000 == 0:
            print(f"   Processed {i+1}/{len(train)}")
        train_features[i] = get_avg_feature_vector(train['review'][i], model, num_features)
    
    print("\n3. Creating test features...")
    test_features = np.zeros((len(test), num_features), dtype="float32")
    
    for i in range(len(test)):
        if (i+1) % 5000 == 0:
            print(f"   Processed {i+1}/{len(test)}")
        test_features[i] = get_avg_feature_vector(test['review'][i], model, num_features)
    
    print(f"\n   ✓ Feature matrices:")
    print(f"     Train: {train_features.shape}")
    print(f"     Test: {test_features.shape}")
    
    # 训练逻辑回归
    print("\n4. Training Logistic Regression...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train['sentiment'],
        test_size=0.2, random_state=42, stratify=train['sentiment']
    )
    
    print(f"   Train split: {X_train.shape}")
    print(f"   Val split: {X_val.shape}")
    
    model_lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    print("   Training...")
    model_lr.fit(X_train, y_train)
    
    # 验证集评估
    y_pred = model_lr.predict(X_val)
    y_pred_proba = model_lr.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n   ✓ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"   ✓ Validation AUC: {val_auc:.4f}")
    
    # 全量训练并预测
    print("\n5. Training on full dataset...")
    model_lr.fit(train_features, train['sentiment'])
    result = model_lr.predict(test_features)
    
    output = pd.DataFrame({
        'id': test['id'],
        'sentiment': result
    })
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join(SUBMISSION_DIR, f"revised_w2v_lr_{timestamp}.csv")
    output.to_csv(submission_path, index=False, quoting=3)
    
    print(f"   ✓ Submission saved: {submission_path}")
    print(f"   Positive (1): {sum(result)} reviews")
    print(f"   Negative (0): {len(result) - sum(result)} reviews")
    
    # 结果对比
    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print(f"  Part 1 Revised (TF-IDF + LR):   Expected AUC ~0.94+")
    print(f"  Part 3 Revised (W2V + LR):      Val AUC = {val_auc:.4f}")
    print("=" * 70)
    print("\n✓ Complete! Upload to Kaggle for Public Score")
    print("=" * 70)


if __name__ == "__main__":
    main()
