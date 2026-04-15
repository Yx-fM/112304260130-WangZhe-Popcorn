"""
Part 1 Revised: Bag of Words with Proper Preprocessing
关键改进：
1. 保留否定词 (not, no, never, nor)
2. 使用 n-grams (1-2 grams) 捕捉短语如 "not good", "very bad"
3. 使用逻辑回归（简单模型效果更好）
4. TF-IDF 特征加权
"""

import pandas as pd
import numpy as np
import re
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import datetime

# 否定词列表 - 千万不要删除！
NEGATION_WORDS = {
    'not', 'no', 'never', 'nor', 'neither', 'nobody', 'nothing',
    'nowhere', 'none', "n't", 'cannot', "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "didn't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't"
}

# 常用停用词 - 但要排除否定词
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'dare', 'ought', 'used', 'right', 'there', 'here', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'we', 'it', 'they',
    'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'then', 'once', 'if', 'because', 'as', 
    'until', 'while', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out',
    'off', 'over', 'under', 'again', 'further', 'any', 'many', 'much',
    'am', 'its', 'your', 'his', 'her', 'our', 'their', 'my', 'me', 'him',
    'us', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself',
    'ourselves', 'themselves', 'yourselves', 's', 't', 'd', 'll', 've', 're', 'm'
}


def review_to_words(raw_review, keep_negations=True):
    """
    将原始评论转换为干净的单词序列
    
    关键改进：
    1. 保留否定词
    2. 处理缩写（如 don't -> do not）
    3. 不删除标点（先保留，用于后续处理）
    """
    # 1. 去除 HTML 标签
    markup = BeautifulSoup(raw_review, 'html.parser')
    markup_text = markup.get_text()
    
    # 2. 处理常见缩写 - 重要！把 didn't 展开为 did not
    contractions = {
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'s": " is",
        "'m": " am",
        "'re": " are"
    }
    
    text = markup_text.lower()
    for short, full in contractions.items():
        text = text.replace(short, full)
    
    # 3. 去除标点符号（但已经处理了缩写）
    letters_only = re.sub("[^a-zA-Z\\s]", " ", text)
    
    # 4. 分割成单词
    words = letters_only.split()
    
    # 5. 移除停用词，但保留否定词
    if keep_negations:
        words = [w for w in words if w not in STOP_WORDS or w in NEGATION_WORDS]
    else:
        words = [w for w in words if w not in STOP_WORDS]
    
    return " ".join(words)


def main():
    # 数据路径
    DATA_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\data"
    SUBMISSION_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\submissions"
    
    # 确保提交目录存在
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Part 1 Revised: Proper Preprocessing + Logistic Regression")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n【Step 1】Loading data...")
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
    
    # 2. 文本清洗
    print("\n【Step 2】Text preprocessing...")
    print("   Key improvements:")
    print("   - Keeping negation words (not, no, never)")
    print("   - Expanding contractions (don't -> do not)")
    print("   - Using n-grams (1-2 grams) for phrases")
    
    print("\n   Cleaning training data...")
    clean_train_reviews = [review_to_words(review) for review in train['review']]
    
    print("   Cleaning test data...")
    clean_test_reviews = [review_to_words(review) for review in test['review']]
    print("   ✓ Cleaning complete!")
    
    # 3. TF-IDF 特征提取（使用 n-grams）
    print("\n【Step 3】TF-IDF Feature Extraction with n-grams...")
    
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),  # 使用 1-2 grams，捕捉 "not good", "very bad"
        max_features=10000,   # 增加特征数
        min_df=2,
        max_df=0.85,          # 去掉太常见的词
        sublinear_tf=True,    # 使用对数 TF 缩放
        use_idf=True,         # 使用 IDF 加权
        smooth_idf=True,
        norm='l2'             # L2 归一化
    )
    
    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
    test_data_features = vectorizer.transform(clean_test_reviews).toarray()
    
    print(f"   Train features: {train_data_features.shape}")
    print(f"   Test features: {test_data_features.shape}")
    
    # 查看一些重要的 n-gram 特征
    feature_names = vectorizer.get_feature_names_out()
    negation_features = [f for f in feature_names if 'not' in f][:10]
    print(f"\n   Sample negation-related features: {negation_features}")
    
    # 4. 训练逻辑回归模型（简单模型，效果好）
    print("\n【Step 4】Training Logistic Regression...")
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        train_data_features, train['sentiment'],
        test_size=0.2, random_state=42, stratify=train['sentiment']
    )
    
    print(f"   Train split: {X_train.shape}")
    print(f"   Val split: {X_val.shape}")
    
    # 逻辑回归 - 简单但强大
    model = LogisticRegression(
        C=1.0,              # 正则化强度
        max_iter=1000,      # 增加迭代次数
        solver='lbfgs',     # 适合多分类和大型数据集
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    print("   Training...")
    model.fit(X_train, y_train)
    
    # 5. 验证集评估
    print("\n【Step 5】Validation...")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"   Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"   AUC: {val_auc:.4f}")
    
    # 6. 全量训练并预测
    print("\n【Step 6】Training on full dataset...")
    model.fit(train_data_features, train['sentiment'])
    
    print("   Predicting test set...")
    result = model.predict(test_data_features)
    result_proba = model.predict_proba(test_data_features)[:, 1]
    
    # 保存提交文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 二值预测提交
    submission_path = os.path.join(SUBMISSION_DIR, f"revised_lr_{timestamp}.csv")
    output = pd.DataFrame({
        'id': test['id'],
        'sentiment': result
    })
    output.to_csv(submission_path, index=False, quoting=3)
    print(f"   ✓ Submission saved: {submission_path}")
    
    # 概率提交（可能有用）
    proba_path = os.path.join(SUBMISSION_DIR, f"revised_lr_proba_{timestamp}.csv")
    pd.DataFrame({
        'id': test['id'],
        'sentiment_proba': result_proba
    }).to_csv(proba_path, index=False, quoting=3)
    
    # 统计
    print(f"\n   Positive (1): {sum(result)} reviews")
    print(f"   Negative (0): {len(result) - sum(result)} reviews")
    
    # 7. 结果汇总
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"   Previous (RandomForest): AUC ~0.87")
    print(f"   Revised (LogisticRegression+TF-IDF+ngrams):")
    print(f"      - Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"      - Val AUC: {val_auc:.4f}")
    print(f"\n   Expected Kaggle AUC: 0.94+")
    print("=" * 70)
    
    if val_auc >= 0.94:
        print("\n✓ Excellent! Validation AUC reaches 0.94+")
    elif val_auc >= 0.90:
        print("\n⚠ Good, but may need tuning for 0.94+")
    else:
        print("\n⚠ May need further optimization")
    
    print("\n【Next】Upload to Kaggle and check Public Score!")
    print("=" * 70)


if __name__ == "__main__":
    main()
