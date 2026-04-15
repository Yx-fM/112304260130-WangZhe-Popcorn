"""
Part 4: 优化版 - 快速特征提取 + 强正则化
目标：Kaggle AUC 0.94+
"""

import pandas as pd
import numpy as np
import os
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
import xgboost as xgb
import re
from bs4 import BeautifulSoup
from scipy.sparse import hstack, csr_matrix

# ============ 配置 ============
DATA_DIR = r'Q:\All_Items\PythonProjects\Popcorn\data'
MODELS_DIR = r'Q:\All_Items\PythonProjects\Popcorn\models'
SUBMISSION_DIR = r'Q:\All_Items\PythonProjects\Popcorn\submissions'

NUM_CENTROIDS = 2000   # 减少聚类数加速
BOW_FEATURES = 8000    # 增加 BoW 特征
W2V_DIM = 300

print('='*70)
print('Part 4: Optimized Feature Fusion')
print('='*70)

# ============ 文本清洗 ============
def review_to_words(r):
    return ' '.join(re.sub('[^a-zA-Z]', ' ', BeautifulSoup(r, 'html.parser').get_text()).lower().split())

print('\n【Step 1】Loading data...')
train = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), header=0, delimiter='\t', quoting=3)
test = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), header=0, delimiter='\t', quoting=3)
y_train = train['sentiment'].values

clean_train = [review_to_words(r) for r in train['review']]
clean_test = [review_to_words(r) for r in test['review']]
print(f'  Train: {len(clean_train)}, Test: {len(clean_test)}')

# ============ 特征 1: BoW ============
print('\n【Step 2】BoW features...')
bow_vectorizer = CountVectorizer(max_features=BOW_FEATURES, min_df=2, max_df=0.8, ngram_range=(1,2))
bow_train = bow_vectorizer.fit_transform(clean_train)
bow_test = bow_vectorizer.transform(clean_test)
print(f'  BoW: {bow_train.shape[1]} features')

# ============ 特征 2: Word2Vec 平均 ============
print('\n【Step 3】Word2Vec features...')
model = Word2Vec.load(os.path.join(MODELS_DIR, 'word2vec_model.bin'))

def make_w2v_features(reviews, model):
    """快速词向量平均"""
    result = []
    for r in reviews:
        words = r.split()
        vec = np.zeros(model.vector_size, dtype='float32')
        count = 0
        for w in words:
            if w in model.wv:
                vec += model.wv[w]
                count += 1
        result.append(vec / count if count > 0 else vec)
    return np.array(result)

w2v_train = make_w2v_features(clean_train, model)
w2v_test = make_w2v_features(clean_test, model)
print(f'  W2V: {w2v_train.shape[1]} dimensions')

# ============ 特征 3: Bag of Centroids (优化版) ============
print('\n【Step 4】Bag of Centroids...')

# 先用一部分词训练 K-means
print('  Training K-means...')
word_vecs = model.wv.vectors.astype('float64')  # 转 float64
kmeans = KMeans(n_clusters=NUM_CENTROIDS, random_state=42, n_init=5, max_iter=20)
kmeans.fit(word_vecs)

# 批量预测所有词的 centroid ID（更高效）
print('  Creating word-to-centroid mapping...')
all_words = list(model.wv.key_to_index.keys())
all_vecs = np.array([model.wv.vectors[model.wv.key_to_index[w]] for w in all_words], dtype='float64')
centroid_ids = kmeans.predict(all_vecs)
word_to_centroid = {word: int(cid) for word, cid in zip(all_words, centroid_ids)}
print(f'  Mapped {len(word_to_centroid)} words to {NUM_CENTROIDS} centroids')

def make_centroid_features(reviews, word_to_centroid):
    """快速 centroid 特征"""
    result = []
    for r in reviews:
        words = r.split()
        hist = np.zeros(NUM_CENTROIDS, dtype='float32')
        count = 0
        for w in words:
            if w in word_to_centroid:
                hist[word_to_centroid[w]] += 1
                count += 1
        if count > 0:
            hist = hist / count
        result.append(hist)
    return np.array(result)

print('  Creating centroid features...')
centroid_train = make_centroid_features(clean_train, word_to_centroid)
centroid_test = make_centroid_features(clean_test, word_to_centroid)
print(f'  Centroids: {NUM_CENTROIDS} clusters')

# ============ 特征融合 ============
print('\n【Step 5】Fusing features...')
# BoW 是稀疏矩阵，需要转换
w2v_sparse = csr_matrix(w2v_train)
centroid_sparse = csr_matrix(centroid_train)
X_train_fused = hstack([bow_train, w2v_sparse, centroid_sparse])

w2v_sparse_test = csr_matrix(w2v_test)
centroid_sparse_test = csr_matrix(centroid_test)
X_test_fused = hstack([bow_test, w2v_sparse_test, centroid_sparse_test])

print(f'  Fused features: {X_train_fused.shape[1]}')

# ============ XGBoost 训练 ============
print('\n【Step 6】Training XGBoost with strong regularization...')

# 强力防止过拟合的参数
xgb_params = {
    'n_estimators': 500,
    'max_depth': 4,           # 更浅的树
    'learning_rate': 0.03,    # 更小的学习率
    'subsample': 0.7,         # 更强的采样
    'colsample_bytree': 0.7,
    'min_child_weight': 10,   # 更大的最小权重
    'gamma': 0.2,
    'reg_alpha': 0.5,         # 更强的 L1 正则
    'reg_lambda': 2.0,        # 更强的 L2 正则
    'tree_method': 'hist',
    'n_jobs': -1,
    'eval_metric': 'auc',
    'random_state': 42
}

# 5 折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('  Running 5-fold CV...')
auc_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_fused, y_train)):
    X_tr = X_train_fused[train_idx]
    X_val = X_train_fused[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, verbose=False)
    
    y_pred_proba = model_xgb.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(fold_auc)
    print(f'  Fold {fold+1}: AUC = {fold_auc:.4f}')

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
print(f'\n  CV Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})')

# ============ 全量训练 ============
print('\n【Step 7】Training on full dataset...')
final_model = xgb.XGBClassifier(**xgb_params)
final_model.fit(X_train_fused, y_train, verbose=False)

# 预测
print('\n【Step 8】Generating submission...')
test_pred_proba = final_model.predict_proba(X_test_fused)[:, 1]
test_pred = (test_pred_proba > 0.5).astype(int)

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission_path = os.path.join(SUBMISSION_DIR, f'part4_fused_{ts}.csv')
pd.DataFrame({'id': test['id'], 'sentiment': test_pred}).to_csv(submission_path, index=False, quoting=3)

print(f'  Saved: {submission_path}')
print(f'  Positive: {sum(test_pred)}, Negative: {len(test_pred)-sum(test_pred)}')

# ============ 结果汇总 ============
print('\n' + '='*70)
print('Results Summary')
print('='*70)
print(f'  Part 1 (BoW):                  AUC ~0.85-0.87')
print(f'  Part 3 (W2V + XGB):            AUC = 0.8668 (Kaggle)')
print(f'  Part 4 (Fusion + Centroids):   CV AUC = {mean_auc:.4f}')
print('-'*70)
print(f'  Target: AUC 0.94+')
print('='*70)

if mean_auc >= 0.94:
    print('\n✓ CV AUC 达到 0.94+！Kaggle 有望及格！')
elif mean_auc >= 0.92:
    print('\n⚠ CV AUC 接近 0.92，Kaggle 预计 0.90-0.93')
else:
    print('\n⚠ CV AUC 较低，可能需要进一步调参')

print('\nNext: Upload to Kaggle!')
print('='*70)
