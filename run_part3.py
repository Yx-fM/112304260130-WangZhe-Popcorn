import pandas as pd
import numpy as np
import os
import pickle
import datetime
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from bs4 import BeautifulSoup

DATA_DIR = r'Q:\All_Items\PythonProjects\Popcorn\data'
MODELS_DIR = r'Q:\All_Items\PythonProjects\Popcorn\models'
SUBMISSION_DIR = r'Q:\All_Items\PythonProjects\Popcorn\submissions'

print('='*60)
print('Part 3: Word2Vec')
print('='*60)

print('\n[Part 3A] Loading sentences...')
with open(os.path.join(DATA_DIR, 'processed', 'all_sentences.pkl'), 'rb') as f:
    all_sentences = pickle.load(f)
print(f'  Sentences: {len(all_sentences):,}')

print('\n[Part 3A] Training Word2Vec (300d, Skip-gram)...')
model = Word2Vec(sentences=all_sentences, vector_size=300, min_count=40, workers=4, window=10, sample=1e-3, sg=1, epochs=10)
vocab_size = len(model.wv.key_to_index)
print(f'  Vocabulary: {vocab_size:,} words')
model.save(os.path.join(MODELS_DIR, 'word2vec_model.bin'))
print('  Model saved!')

print('\n[Part 3B] Loading data...')
train = pd.read_csv(os.path.join(DATA_DIR, 'labeledTrainData.tsv'), header=0, delimiter='\t', quoting=3)
test = pd.read_csv(os.path.join(DATA_DIR, 'testData.tsv'), header=0, delimiter='\t', quoting=3)

def review_to_words(r):
    return ' '.join(re.sub('[^a-zA-Z]', ' ', BeautifulSoup(r, 'html.parser').get_text()).lower().split())

def make_vec(review):
    words = review_to_words(review).split()
    vec = np.zeros(300, dtype='float32')
    n = sum(1 for w in words if w in model.wv)
    if n > 0:
        for w in words:
            if w in model.wv: vec += model.wv[w]
        vec /= n
    return vec

print('\n[Part 3B] Creating features...')
train_features = np.array([make_vec(r) for r in train['review']])
test_features = np.array([make_vec(r) for r in test['review']])
print(f'  Train: {train_features.shape}, Test: {test_features.shape}')

print('\n[Part 3B] Training classifier...')
X_train, X_val, y_train, y_val = train_test_split(train_features, train['sentiment'], test_size=0.2, random_state=42)
forest = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1)
forest.fit(X_train, y_train)
acc = accuracy_score(y_val, forest.predict(X_val))
print(f'  Val Accuracy: {acc:.4f} ({acc*100:.2f}%)')

print('\n[Part 3B] Generating submission...')
forest.fit(train_features, train['sentiment'])
result = forest.predict(test_features)
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
pd.DataFrame({'id': test['id'], 'sentiment': result}).to_csv(
    os.path.join(SUBMISSION_DIR, f'word2vec_{ts}.csv'), index=False, quoting=3
)
print(f'  Saved: word2vec_{ts}.csv')

print('\n'+'='*60)
print('Results:')
print(f'  Part 1 (BoW): 82.78%')
print(f'  Part 3 (Word2Vec): {acc:.4f} ({acc*100:.2f}%)')
print('='*60)
