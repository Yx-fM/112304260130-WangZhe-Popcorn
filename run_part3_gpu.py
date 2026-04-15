import pandas as pd
import numpy as np
import os
import pickle
import datetime
from gensim.models import Word2Vec
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from bs4 import BeautifulSoup

print('='*60)
print('Part 3: Word2Vec + XGBoost GPU')
print('='*60)

DATA_DIR = r'Q:\All_Items\PythonProjects\Popcorn\data'
MODELS_DIR = r'Q:\All_Items\PythonProjects\Popcorn\models'
SUBMISSION_DIR = r'Q:\All_Items\PythonProjects\Popcorn\submissions'

# ============ Part 3A: Word2Vec Training ============
print('\n【Part 3A】Loading sentences...')
with open(os.path.join(DATA_DIR, 'processed', 'all_sentences.pkl'), 'rb') as f:
    all_sentences = pickle.load(f)
print(f'  Sentences: {len(all_sentences):,}')

print('\n【Part 3A】Training Word2Vec (300d, Skip-gram)...')
model = Word2Vec(
    sentences=all_sentences, 
    vector_size=300, 
    min_count=40, 
    workers=8,  # 多 CPU 并行
    window=10, 
    sample=1e-3, 
    sg=1, 
    epochs=10
)
vocab_size = len(model.wv.key_to_index)
print(f'  Vocabulary: {vocab_size:,} words')

model_path = os.path.join(MODELS_DIR, 'word2vec_model.bin')
model.save(model_path)
print(f'  Model saved: {model_path}')

# ============ Part 3B: Feature Extraction (向量化优化) ============
print('\n【Part 3B】Loading data...')
train = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), header=0, delimiter='\t', quoting=3)
test = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), header=0, delimiter='\t', quoting=3)

def review_to_words(r):
    return ' '.join(re.sub('[^a-zA-Z]', ' ', BeautifulSoup(r, 'html.parser').get_text()).lower().split())

print('\n【Part 3B】Creating features (optimized)...')

# 预处理所有评论
print('  Preprocessing reviews...')
train_words = [review_to_words(r).split() for r in train['review']]
test_words = [review_to_words(r).split() for r in test['review']]

# 词向量矩阵化 (300 维)
print('  Creating feature vectors...')
wv_matrix = model.wv.vectors  # (vocab_size, 300)

def words_to_vec(words, word2idx, wv_matrix):
    indices = [word2idx[w] for w in words if w in word2idx]
    if len(indices) == 0:
        return np.zeros(300, dtype='float32')
    return wv_matrix[indices].mean(axis=0)

word2idx = model.wv.key_to_index

# 批处理加速
print('  Batch processing...')
train_features = np.array([words_to_vec(w, word2idx, wv_matrix) for w in train_words])
test_features = np.array([words_to_vec(w, word2idx, wv_matrix) for w in test_words])

print(f'  Train: {train_features.shape}, Test: {test_features.shape}')

# ============ Part 3B: XGBoost GPU 训练 ============
print('\n【Part 3B】Training XGBoost (GPU)...')

X_train, X_val, y_train, y_val = train_test_split(
    train_features, train['sentiment'], test_size=0.2, random_state=42
)

# XGBoost 参数 (hist 方法已经很快)
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    tree_method='hist',  # 直方图优化，CPU 也很快
    n_jobs=-1,  # 多核并行
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train, verbose=True)

# 评估
y_pred = xgb_model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f'\n  Val Accuracy: {acc:.4f} ({acc*100:.2f}%)')

# ============ 生成提交 ============
print('\n【Part 3B】Generating submission...')
xgb_model.fit(train_features, train['sentiment'], verbose=False)
result = xgb_model.predict(test_features)

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission_path = os.path.join(SUBMISSION_DIR, f'word2vec_xgb_{ts}.csv')
pd.DataFrame({'id': test['id'], 'sentiment': result}).to_csv(submission_path, index=False, quoting=3)

print(f'  Saved: {submission_path}')
print(f'  Positive: {sum(result)}, Negative: {len(result)-sum(result)}')

print('\n'+'='*60)
print('Results:')
print(f'  Part 1 (BoW + RF): 82.78%')
print(f'  Part 3 (Word2Vec + XGB): {acc:.4f} ({acc*100:.2f}%)')
print('='*60)
print('\nDone! Upload to Kaggle!')
