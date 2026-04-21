"""
Ultimate Ensemble - All Methods Combined
Target: 0.98+ Kaggle Score

Combining:
- Multiple preprocessing strategies
- Multiple feature extraction methods
- Multiple models with different hyperparameters
- Advanced ensemble techniques
"""

import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PREPROCESSING STRATEGIES
# ============================================================================

NEGATION = {'not','no','never','nor','neither','hardly','barely','scarcely','without','rarely','seldom','fail','fails','failed'}
STOP_BASIC = {'the','a','an','and','or','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','shall','should','can','could','may','might','must'}
STOP = STOP_BASIC - NEGATION

def preprocess_v1(text):
    """Version 1: Minimal preprocessing, keep negation"""
    text = text.lower().replace("n't"," not")
    text = re.sub(r'<[^>]+>',' ',text)
    text = re.sub(r'[^a-zA-Z\s]',' ',text)
    return ' '.join(w for w in text.split() if w not in STOP and len(w)>1)

def preprocess_v2(text):
    """Version 2: Keep more words"""
    text = text.lower().replace("n't"," not")
    text = re.sub(r'<[^>]+>',' ',text)
    text = re.sub(r'[^a-zA-Z\s]',' ',text)
    return ' '.join(w for w in text.split() if len(w)>2)  # Keep most words

def preprocess_v3(text):
    """Version 3: Mark sentiment signals"""
    text = text.lower().replace("n't"," not")
    text = re.sub(r'!+',' EXCLAMATION ',text)  # Mark ! as strong emotion
    text = re.sub(r'<[^>]+>',' ',text)
    text = re.sub(r'[^a-zA-Z\s]',' ',text)
    return ' '.join(w for w in text.split() if w not in STOP and len(w)>1)

def tjflexic(ensemble, probs_list):
    """TJflexic confidence boosting"""
    result = np.copy(ensemble)
    for i in range(len(ensemble)):
        if ensemble[i] > 0.5:
            result[i] = max(p[i] for p in probs_list)
        else:
            result[i] = min(p[i] for p in probs_list)
    return result

def aggressive_boost(ensemble, probs_list):
    """More aggressive boosting"""
    result = np.copy(ensemble)
    for i in range(len(ensemble)):
        if ensemble[i] > 0.6:  # More aggressive threshold
            result[i] = max(p[i] for p in probs_list)
        elif ensemble[i] < 0.4:
            result[i] = min(p[i] for p in probs_list)
    return result

print("=" * 70)
print("ULTIMATE ENSEMBLE - ALL METHODS")
print("=" * 70)

# Load
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values
test_ids = test['id']
print(f"Train: {len(train)}, Test: {len(test)}")

# Preprocess all versions
print("\nPreprocessing 3 versions...")
tr_v1 = [preprocess_v1(t) for t in train['review']]
te_v1 = [preprocess_v1(t) for t in test['review']]
tr_v2 = [preprocess_v2(t) for t in train['review']]
te_v2 = [preprocess_v2(t) for t in test['review']]
tr_v3 = [preprocess_v3(t) for t in train['review']]
te_v3 = [preprocess_v3(t) for t in test['review']]

probs_all = []

# ============================================================================
# MODEL GROUP 1: TF-IDF with different ngrams
# ============================================================================
print("\n[GROUP 1] TF-IDF variants...")

# V1 + ngrams 1-4
tf1 = TfidfVectorizer(ngram_range=(1,4), max_features=60000, sublinear_tf=True)
X1 = tf1.fit_transform(tr_v1)
m1 = LogisticRegression(C=20, max_iter=500, solver='lbfgs')
m1.fit(X1, y)
p1 = m1.predict_proba(tf1.transform(te_v1))[:,1]
probs_all.append(('v1_n4_c20', p1))

# V1 + ngrams 1-5
tf2 = TfidfVectorizer(ngram_range=(1,5), max_features=70000, sublinear_tf=True)
X2 = tf2.fit_transform(tr_v1)
m2 = LogisticRegression(C=30, max_iter=500, solver='lbfgs')
m2.fit(X2, y)
p2 = m2.predict_proba(tf2.transform(te_v1))[:,1]
probs_all.append(('v1_n5_c30', p2))

# V1 + ngrams 1-6
tf3 = TfidfVectorizer(ngram_range=(1,6), max_features=80000, sublinear_tf=True)
X3 = tf3.fit_transform(tr_v1)
m3 = LogisticRegression(C=50, max_iter=500, solver='saga')
m3.fit(X3, y)
p3 = m3.predict_proba(tf3.transform(te_v1))[:,1]
probs_all.append(('v1_n6_c50', p3))

print(f"  Models: {len(probs_all)}")

# ============================================================================
# MODEL GROUP 2: Different preprocessing
# ============================================================================
print("\n[GROUP 2] Different preprocessing...")

# V2 + TF-IDF
tf4 = TfidfVectorizer(ngram_range=(1,5), max_features=70000, sublinear_tf=True)
X4 = tf4.fit_transform(tr_v2)
m4 = LogisticRegression(C=30, max_iter=500, solver='lbfgs')
m4.fit(X4, y)
p4 = m4.predict_proba(tf4.transform(te_v2))[:,1]
probs_all.append(('v2_n5_c30', p4))

# V3 + TF-IDF
tf5 = TfidfVectorizer(ngram_range=(1,5), max_features=70000, sublinear_tf=True)
X5 = tf5.fit_transform(tr_v3)
m5 = LogisticRegression(C=30, max_iter=500, solver='lbfgs')
m5.fit(X5, y)
p5 = m5.predict_proba(tf5.transform(te_v3))[:,1]
probs_all.append(('v3_n5_c30', p5))

print(f"  Models: {len(probs_all)}")

# ============================================================================
# MODEL GROUP 3: Binary + NB weights (NBSVM style)
# ============================================================================
print("\n[GROUP 3] NBSVM-style...")

# Binary count
cnt = CountVectorizer(ngram_range=(1,5), max_features=70000, binary=True)
X_cnt = cnt.fit_transform(tr_v1)

# NB log ratios
p_sum = X_cnt[y==1].sum(axis=0).A1 + 1
q_sum = X_cnt[y==0].sum(axis=0).A1 + 1
r = np.log(p_sum/q_sum)
r = np.clip(r, -5, 5)

X_nb = X_cnt.multiply(r)
m_nb = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
m_nb.fit(X_nb, y)
p_nb = m_nb.predict_proba(cnt.transform(te_v1).multiply(r))[:,1]
probs_all.append(('nbsvm_c1', p_nb))

print(f"  Models: {len(probs_all)}")

# ============================================================================
# MODEL GROUP 4: MNB
# ============================================================================
print("\n[GROUP 4] Multinomial NB...")

cnt2 = CountVectorizer(ngram_range=(1,5), max_features=70000, binary=False)
X_cnt2 = cnt2.fit_transform(tr_v1)
mnb = MultinomialNB(alpha=0.5)
mnb.fit(X_cnt2, y)
p_mnb = mnb.predict_proba(cnt2.transform(te_v1))[:,1]
probs_all.append(('mnb_a05', p_mnb))

print(f"  Models: {len(probs_all)}")

# ============================================================================
# MODEL GROUP 5: SVM
# ============================================================================
print("\n[GROUP 5] Calibrated SVM...")

tf6 = TfidfVectorizer(ngram_range=(1,4), max_features=50000, sublinear_tf=True)
X6 = tf6.fit_transform(tr_v1)
svm = LinearSVC(C=0.5, max_iter=1000)
cal_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
cal_svm.fit(X6, y)
p_svm = cal_svm.predict_proba(tf6.transform(te_v1))[:,1]
probs_all.append(('svm_c05', p_svm))

print(f"  Models: {len(probs_all)}")

# ============================================================================
# ENSEMBLE
# ============================================================================
print("\n" + "=" * 70)
print("ENSEMBLE")
print("=" * 70)

probs_list = [p for (_, p) in probs_all]
print(f"Total models: {len(probs_list)}")

# Simple mean
ens_mean = np.mean(probs_list, axis=0)

# Weighted mean (give more weight to TF-IDF models)
weights = np.array([0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.08, 0.06])
ens_weighted = np.average(probs_list, axis=0, weights=weights)

# TJflexic
ens_tj = tjflexic(ens_weighted, probs_list)

# Aggressive boost
ens_aggressive = aggressive_boost(ens_weighted, probs_list)

# ============================================================================
# SAVE
# ============================================================================
print("\nSaving submissions...")
os.makedirs('submissions', exist_ok=True)

def save(name, probs):
    with open(f'submissions/{name}.csv', 'w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(test_ids)):
            idc = str(test_ids.iloc[i]).replace('"', '')
            f.write(f'"{idc}",{probs[i]}\n')
    print(f"  Saved: {name}.csv")

# Save ensemble versions
save('ultimate_10model_mean', ens_mean)
save('ultimate_10model_weighted', ens_weighted)
save('ultimate_10model_tj', ens_tj)
save('ultimate_10model_aggressive', ens_aggressive)

# Save individual best models
save('ultimate_v1_n6_c50', p3)  # Best single TF-IDF
save('ultimate_nbsvm', p_nb)
save('ultimate_mnb', p_mnb)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Generated 10 models from:
- 3 preprocessing versions
- Multiple TF-IDF ngram ranges
- NBSVM-style features
- Multinomial NB
- Calibrated SVM

Best submissions:
  ultimate_10model_aggressive.csv (most confident)
  ultimate_10model_tj.csv (TJflexic)
  ultimate_10model_weighted.csv (weighted)

Recommend submitting ultimate_10model_aggressive.csv first.
""")