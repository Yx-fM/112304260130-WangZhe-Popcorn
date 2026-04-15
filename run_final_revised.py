# -*- coding: utf-8 -*-
"""
Final Revised: Complete Pipeline with Proper Preprocessing
目标：Kaggle AUC 0.94+

关键改进（根据老师提示）：
1. ✅ 保留否定词 (not, no, never, nor) - 不删除！
2. ✅ 使用 n-grams (1-3 grams) 捕捉短语 "not good", "very bad", "not great"
3. ✅ 使用逻辑回归等简单模型，而不是复杂的随机森林/XGBoost
4. ✅ 展开缩写 (don't -> do not)
5. ✅ TF-IDF 加权而不是简单的词袋

模型选择：
- 逻辑回归 (主模型) - 简单、快速、效果好
- 可选：线性 SVM、Ridge Regression 作为对比
"""

import sys
import pandas as pd
import numpy as np
import re
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
DATA_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\data"
SUBMISSION_DIR = r"Q:\All_Items\PythonProjects\112304260130-Popcorn\submissions"

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 否定词列表 - 绝对不要删除！
NEGATION_WORDS = frozenset([
    'not', 'no', 'never', 'nor', 'neither', 'nobody', 'nothing',
    'nowhere', 'none', "n't", 'cannot', "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "didn't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", 'wont',
    'wouldnt', 'shouldnt', 'couldnt', 'didnt', 'doesnt', 'isnt',
    'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'dont'
])

# 停用词 - 已排除否定词
STOP_WORDS = frozenset([
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
])


def preprocess_review(raw_review):
    """
    关键预处理步骤：
    1. 去除 HTML
    2. 展开缩写（don't -> do not）- 保留否定词
    3. 转小写
    4. 去除标点
    5. 移除停用词（但保留否定词）
    """
    # 1. 去除 HTML
    soup = BeautifulSoup(raw_review, 'html.parser')
    text = soup.get_text()
    
    # 2. 展开缩写 - 关键！保留否定含义
    contractions = {
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'s": " is",
        "'m": " am",
        "ca": " can",
        "'ve": " have"
    }
    
    text = text.lower()
    for short, full in contractions.items():
        text = text.replace(short, full)
    
    # 3. 去除标点
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 4. 分割并移除停用词（保留否定词）
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS or w in NEGATION_WORDS]
    
    return " ".join(words)


def cross_validate(model, X, y, n_splits=5):
    """5 折交叉验证"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        
        # 检查是否支持 predict_proba
        try:
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, y_pred_proba)
        except AttributeError:
            # 如果不支持，用 decision_function 代替
            y_scores = model.decision_function(X_val)
            fold_auc = roc_auc_score(y_val, y_scores)
        
        y_pred = model.predict(X_val)
        fold_acc = accuracy_score(y_val, y_pred)
        
        aucs.append(fold_auc)
        accuracies.append(fold_acc)
    
    return np.mean(aucs), np.std(aucs), np.mean(accuracies)


def main():
    print("=" * 80)
    print("FINAL REVISED: Proper Preprocessing + Simple Models")
    print("Target: Kaggle AUC 0.94+")
    print("=" * 80)
    
    # ============ Step 1: 加载数据 ============
    print("\n【Step 1】Loading data...")
    train = pd.read_csv(
        os.path.join(DATA_DIR, "labeledTrainData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    test = pd.read_csv(
        os.path.join(DATA_DIR, "testData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    y_train = train['sentiment'].values
    
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    
    # ============ Step 2: 文本预处理 ============
    print("\n【Step 2】Preprocessing reviews...")
    print("   Key improvements:")
    print("   [OK] Keeping negation words (not, no, never)")
    print("   [OK] Expanding contractions (don't -> do not)")
    print("   [OK] Using n-grams (1-3) for phrases like 'not good', 'very bad'")
    print("   [OK] TF-IDF weighting")
    
    print("\n   Cleaning training data...")
    clean_train = [preprocess_review(r) for r in train['review']]
    
    print("   Cleaning test data...")
    clean_test = [preprocess_review(r) for r in test['review']]
    print("   ✓ Complete!")
    
    # ============ Step 3: TF-IDF 特征提取 ============
    print("\n【Step 3】TF-IDF Feature Extraction with n-grams...")
    
    # 尝试不同的 n-gram 范围
    best_auc = 0
    best_ngram = (1, 2)
    
    for ngram_range in [(1, 2), (1, 3)]:
        print(f"\n   Testing ngram_range={ngram_range}...")
        
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            max_features=15000,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        X_train = vectorizer.fit_transform(clean_train).toarray()
        X_test = vectorizer.transform(clean_test).toarray()
        
        print(f"      Features: {X_train.shape[1]}")
        
        # 快速验证
        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1, random_state=42)
        mean_auc, std_auc, mean_acc = cross_validate(model, X_train, y_train, n_splits=5)
        
        print(f"      CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_ngram = ngram_range
            best_vectorizer = vectorizer
            best_X_train = X_train
            best_X_test = X_test
    
    print(f"\n   [OK] Best ngram_range: {best_ngram}")
    print(f"   [OK] Best CV AUC: {best_auc:.4f}")
    
    X_train_final = best_X_train
    X_test_final = best_X_test
    
    # 显示一些重要的 n-gram 特征
    feature_names = best_vectorizer.get_feature_names_out()
    negation_features = [f for f in feature_names if 'not' in f][:15]
    print(f"\n   Sample negation features: {negation_features}")
    
    # ============ Step 4: 训练多个简单模型 ============
    print("\n【Step 4】Training Simple Models...")
    
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1, random_state=42,
            class_weight='balanced'
        ),
        'Ridge Classifier': RidgeClassifier(
            alpha=1.0, random_state=42, class_weight='balanced'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        mean_auc, std_auc, mean_acc = cross_validate(model, X_train_final, y_train, n_splits=5)
        results[name] = {'auc': mean_auc, 'std': std_auc, 'acc': mean_acc}
        print(f"      CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        print(f"      CV Accuracy: {mean_acc:.4f}")
    
    # 选择最佳模型
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    print(f"\n   ✓ Best model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
    
    # ============ Step 5: 全量训练并提交 ============
    print("\n【Step 5】Training final model on full dataset...")
    
    final_model = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1, random_state=42,
        class_weight='balanced'
    )
    final_model.fit(X_train_final, y_train)
    
    print("   Predicting test set...")
    y_pred = final_model.predict(X_test_final)
    y_pred_proba = final_model.predict_proba(X_test_final)[:, 1]
    
    # 保存提交文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    submission_path = os.path.join(SUBMISSION_DIR, f"final_revised_lr_{timestamp}.csv")
    output = pd.DataFrame({
        'id': test['id'],
        'sentiment': y_pred
    })
    output.to_csv(submission_path, index=False, quoting=3)
    print(f"   ✓ Submission saved: {submission_path}")
    
    print(f"\n   Positive (1): {sum(y_pred)} reviews")
    print(f"   Negative (0): {len(y_pred) - sum(y_pred)} reviews")
    
    # ============ Step 6: 结果汇总 ============
    print("\n" + "=" * 80)
    print("Final Results Summary")
    print("=" * 80)
    
    for name, res in results.items():
        print(f"  {name:25s}: CV AUC = {res['auc']:.4f} (+/- {res['std']:.4f})")
    
    print("-" * 80)
    print(f"\n  Best Model: {best_model_name}")
    print(f"  Final CV AUC: {results[best_model_name]['auc']:.4f}")
    print(f"\n  Expected Kaggle AUC: 0.94+ (target achieved if CV AUC >= 0.94)")
    print("=" * 80)
    
    if results[best_model_name]['auc'] >= 0.94:
        print("\n✓ SUCCESS! CV AUC reaches 0.94+ - Kaggle should pass!")
    elif results[best_model_name]['auc'] >= 0.92:
        print("\n⚠ Close! May achieve 0.94+ on Kaggle")
    else:
        print("\n⚠ Below target. Consider:")
        print("   - Increasing max_features")
        print("   - Trying different n-gram ranges")
        print("   - Adding more features (sentiment lexicons)")
    
    print("\n【Next】Upload to Kaggle and check Public Score!")
    print("=" * 80)


if __name__ == "__main__":
    main()
