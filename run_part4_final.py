"""
Part 4: 进阶方法 - Bag of Centroids + 特征融合 + XGBoost 调参
目标：Kaggle AUC 0.94+
"""

import pandas as pd
import numpy as np
import os
import pickle
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from gensim.models import Word2Vec
import xgboost as xgb
import re
from bs4 import BeautifulSoup

# ============ 配置 ============
DATA_DIR = r'Q:\All_Items\PythonProjects\Popcorn\data'
MODELS_DIR = r'Q:\All_Items\PythonProjects\Popcorn\models'
SUBMISSION_DIR = r'Q:\All_Items\PythonProjects\Popcorn\submissions'

NUM_CENTROIDS = 5000   # K-means 聚类数
BOW_FEATURES = 5000    # BoW 特征数
W2V_DIM = 300          # Word2Vec 维度

print('='*70)
print('Part 4: Bag of Centroids + Feature Fusion + XGBoost Tuning')
print('='*70)

# ============ 数据加载和预处理 ============
def review_to_words(r):
    """文本清洗"""
    return ' '.join(re.sub('[^a-zA-Z]', ' ', BeautifulSoup(r, 'html.parser').get_text()).lower().split())

print('\n【Step 1】Loading data...')
train = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), header=0, delimiter='\t', quoting=3)
test = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), header=0, delimiter='\t', quoting=3)
y_train = train['sentiment'].values

# 文本清洗
print('  Cleaning reviews...')
clean_train = [review_to_words(r) for r in train['review']]
clean_test = [review_to_words(r) for r in test['review']]

# ============ 特征 1: Bag of Words ============
print('\n【Step 2】Extracting BoW features...')
bow_vectorizer = CountVectorizer(max_features=BOW_FEATURES, min_df=2, max_df=0.8)
bow_train = bow_vectorizer.fit_transform(clean_train).toarray()
bow_test = bow_vectorizer.transform(clean_test).toarray()
print(f'  BoW Train: {bow_train.shape}, Test: {bow_test.shape}')

# ============ 特征 2: Word2Vec 平均 ============
print('\n【Step 3】Loading Word2Vec model...')
model = Word2Vec.load(os.path.join(MODELS_DIR, 'word2vec_model.bin'))
wv_matrix = model.wv.vectors
word2idx = model.wv.key_to_index

def words_to_vec(words):
    """词向量平均"""
    indices = [word2idx[w] for w in words if w in word2idx]
    if len(indices) == 0:
        return np.zeros(W2V_DIM, dtype='float32')
    return wv_matrix[indices].mean(axis=0)

print('  Creating Word2Vec features...')
w2v_train = np.array([words_to_vec(r.split()) for r in clean_train])
w2v_test = np.array([words_to_vec(r.split()) for r in clean_test])
print(f'  W2V Train: {w2v_train.shape}, Test: {w2v_test.shape}')

# ============ 特征 3: Bag of Centroids ============
print('\n【Step 4】Training K-means for Bag of Centroids...')
print(f'  Clustering {len(wv_matrix)} words into {NUM_CENTROIDS} centroids...')

kmeans = MiniBatchKMeans(n_clusters=NUM_CENTROIDS, batch_size=100, random_state=42, n_init=10)
kmeans.fit(wv_matrix)

def review_to_centroids(review, kmeans_model):
    """将评论转换为 centroids 直方图"""
    words = review.split()
    centroid_ids = []
    for w in words:
        if w in word2idx:
            vec = wv_matrix[word2idx[w]]
            centroid_id = kmeans_model.predict([vec])[0]
            centroid_ids.append(centroid_id)
    
    # 统计每个 centroid 的词频
    hist = np.zeros(NUM_CENTROIDS, dtype='float32')
    if len(centroid_ids) > 0:
        for cid in centroid_ids:
            hist[cid] += 1
        # 归一化
        hist = hist / len(centroid_ids)
    return hist

print('  Creating centroid features for train set...')
centroid_train = np.array([review_to_centroids(r, kmeans) for r in clean_train])
print('  Creating centroid features for test set...')
centroid_test = np.array([review_to_centroids(r, kmeans) for r in clean_test])
print(f'  Centroid Train: {centroid_train.shape}, Test: {centroid_test.shape}')

# ============ 特征融合 ============
print('\n【Step 5】Fusing all features...')
X_train_fused = np.hstack([bow_train, w2v_train, centroid_train])
X_test_fused = np.hstack([bow_test, w2v_test, centroid_test])
print(f'  Fused Train: {X_train_fused.shape}, Test: {X_test_fused.shape}')
print(f'  Features: BoW({BOW_FEATURES}) + W2V({W2V_DIM}) + Centroids({NUM_CENTROIDS})')

# ============ 交叉验证 + XGBoost 调参 ============
print('\n【Step 6】Training XGBoost with Cross-Validation...')

# 防止过拟合的 XGBoost 参数
xgb_params = {
    'n_estimators': 300,
    'max_depth': 6,          # 降低深度防止过拟合
    'learning_rate': 0.05,   # 降低学习率
    'subsample': 0.8,        # 行采样
    'colsample_bytree': 0.8, # 列采样
    'min_child_weight': 5,   # 叶子节点最小权重
    'gamma': 0.1,            # 分裂最小损失减少
    'reg_alpha': 0.1,        # L1 正则
    'reg_lambda': 1.0,       # L2 正则
    'tree_method': 'hist',
    'n_jobs': -1,
    'eval_metric': 'auc',
    'random_state': 42
}

# 5 折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

print('  Running 5-fold CV...')
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_fused, y_train)):
    X_tr, X_val = X_train_fused[train_idx], X_train_fused[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, verbose=False)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
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

# 预测测试集
print('\n【Step 8】Generating submission...')
test_pred_proba = final_model.predict_proba(X_test_fused)[:, 1]
test_pred = final_model.predict(X_test_fused)

# 保存提交文件
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission_path = os.path.join(SUBMISSION_DIR, f'part4_fused_{ts}.csv')
output = pd.DataFrame({'id': test['id'], 'sentiment': test_pred})
output.to_csv(submission_path, index=False, quoting=3)

# 保存概率文件（可选，用于集成）
proba_path = os.path.join(SUBMISSION_DIR, f'part4_proba_{ts}.csv')
pd.DataFrame({'id': test['id'], 'sentiment_proba': test_pred_proba}).to_csv(proba_path, index=False, quoting=3)

print(f'  Submission saved: {submission_path}')
print(f'  Probabilities saved: {proba_path}')
print(f'  Positive: {sum(test_pred)}, Negative: {len(test_pred)-sum(test_pred)}')

# ============ 结果汇总 ============
print('\n' + '='*70)
print('Part 4 Results Summary')
print('='*70)
print(f'  Bag of Words (Part 1):         AUC ~0.85-0.87')
print(f'  Word2Vec + XGBoost (Part 3):   AUC = 0.8668 (Kaggle)')
print(f'  Part 4 (Fusion + Centroids):   CV AUC = {mean_auc:.4f} (+/- {std_auc:.4f})')
print('-'*70)
print(f'  Expected Kaggle AUC: 0.90-0.93 (target: 0.94+)')
print('='*70)

if mean_auc >= 0.94:
    print('\n✓ 恭喜！CV AUC 达到 0.94+，Kaggle 有望及格！')
elif mean_auc >= 0.90:
    print('\n⚠ CV AUC 在 0.90+，Kaggle 预计 0.88-0.91，可能需要进一步优化')
else:
    print('\n⚠ CV AUC 较低，需要调整参数或增加特征')

print('\n【下一步】上传到 Kaggle 查看 Public Score！')
print('='*70)
