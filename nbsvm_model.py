"""NBSVM - Naive Bayes Support Vector Machine
Proven technique for text classification, used by many Kaggle winners"""
import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

print("Loading...")
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values
ids = test['id']

# Preprocessing
def prep(t):
    t = str(t).lower()
    t = t.replace("n't"," not").replace("'t"," not")
    t = re.sub(r'<[^>]+>',' ',t)
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    return ' '.join(w for w in t.split() if len(w)>1)

print("Preprocessing...")
tr = [prep(t) for t in train['review']]
te = [prep(t) for t in test['review']]

# NBSVM uses binarized counts + Naive Bayes log-count ratios
print("Creating binary count features...")
cv = CountVectorizer(ngram_range=(1,4), max_features=80000, binary=True, min_df=1)
X_cnt = cv.fit_transform(tr)
Xt_cnt = cv.transform(te)

# Compute Naive Bayes log-count ratios
print("Computing NB log-count ratios...")
alpha = 1.0  # smoothing
p = alpha + X_cnt[y==1].sum(axis=0)
q = alpha + X_cnt[y==0].sum(axis=0)
p = np.asarray(p).ravel()
q = np.asarray(q).ravel()
r = np.log((p / p.sum()) / (q / q.sum()))

# Create NB features: X_nb = X.multiply(r)
# For sparse matrix, multiply each row by r
print("Creating NB features...")
def nb_features(X, r):
    # X: sparse matrix, r: 1D array
    # Multiply each row by r element-wise
    return X.multiply(r)

X_nb = nb_features(X_cnt, r)
Xt_nb = nb_features(Xt_cnt, r)

# Also add TF-IDF features
print("Adding TF-IDF features...")
tfidf = TfidfVectorizer(ngram_range=(1,5), max_features=60000, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(tr)
Xt_tfidf = tfidf.transform(te)

# Combine NB and TF-IDF
X = hstack([X_nb, X_tfidf]).tocsr()
Xt = hstack([Xt_nb, Xt_tfidf]).tocsr()
print(f"Features: {X.shape[1]}")

# Train models
probs = []

print("Training LR C=10...")
m1 = LogisticRegression(C=10, max_iter=500, solver='lbfgs')
m1.fit(X, y)
probs.append(m1.predict_proba(Xt)[:,1])

print("Training LR C=30...")
m2 = LogisticRegression(C=30, max_iter=500, solver='lbfgs')
m2.fit(X, y)
probs.append(m2.predict_proba(Xt)[:,1])

print("Training LR C=50...")
m3 = LogisticRegression(C=50, max_iter=500, solver='lbfgs')
m3.fit(X, y)
probs.append(m3.predict_proba(Xt)[:,1])

# Ensemble
ens = np.mean(probs, axis=0)

# TJflexic
tj = np.copy(ens)
for i in range(len(ens)):
    if ens[i] > 0.5:
        tj[i] = max(p[i] for p in probs)
    else:
        tj[i] = min(p[i] for p in probs)

# Save
os.makedirs('submissions',exist_ok=True)

def save(name, arr):
    with open(f'submissions/{name}.csv','w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(ids)):
            idc = str(ids.iloc[i]).replace('"','')
            f.write(f'"{idc}",{arr[i]}\n')
    print(f"Saved: {name}.csv ({arr.min():.4f}-{arr.max():.4f})")

save('nbsvm_ensemble', ens)
save('nbsvm_tjflexic', tj)
save('nbsvm_c30', probs[1])

print("\nDone! Submit nbsvm_tjflexic.csv")