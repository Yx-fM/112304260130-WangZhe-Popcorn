"""Aggressive parameters for higher AUC"""
import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack

print("Loading...")
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values
ids = test['id']

# Aggressive preprocessing - preserve negation context
def prep(t):
    t = str(t).lower()
    # Expand negation contractions
    t = t.replace("n't"," not")
    t = t.replace("'t"," not")
    t = t.replace("cannot"," can not")
    t = t.replace("won't"," will not")
    t = t.replace("shan't"," shall not")
    # Keep negation words separate
    for neg in ['not','no','never','nor','neither','hardly','barely','scarcely','without']:
        t = t.replace(neg, f" {neg} ")
    # Remove HTML
    t = re.sub(r'<[^>]+>',' ',t)
    # Keep letters only
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    # Remove very short words except negation
    NEG = {'not','no','nor','never','neither','hardly','barely','scarcely'}
    words = [w for w in t.split() if (len(w)>2 or w in NEG)]
    return ' '.join(words)

print("Preprocessing...")
tr = [prep(t) for t in train['review']]
te = [prep(t) for t in test['review']]

# Aggressive TF-IDF: 1-7 ngrams, 150000 features
print("TF-IDF 1-7 ngrams...")
tf1 = TfidfVectorizer(ngram_range=(1,7), max_features=150000, sublinear_tf=True, min_df=1)
X1 = tf1.fit_transform(tr)
Xt1 = tf1.transform(te)
print(f"Features 1-7: {X1.shape[1]}")

# Also create 1-5 for comparison
print("TF-IDF 1-5 ngrams...")
tf2 = TfidfVectorizer(ngram_range=(1,5), max_features=120000, sublinear_tf=True, min_df=1)
X2 = tf2.fit_transform(tr)
Xt2 = tf2.transform(te)
print(f"Features 1-5: {X2.shape[1]}")

probs = []

# Model 1: LR on 1-7 ngrams
print("LR C=30 on 1-7...")
m1 = LogisticRegression(C=30, max_iter=500, solver='lbfgs')
m1.fit(X1, y)
probs.append(m1.predict_proba(Xt1)[:,1])

# Model 2: LR C=100 on 1-7
print("LR C=100 on 1-7...")
m2 = LogisticRegression(C=100, max_iter=500, solver='lbfgs')
m2.fit(X1, y)
probs.append(m2.predict_proba(Xt1)[:,1])

# Model 3: SGDClassifier (hinge loss = SVM)
print("SGDClassifier on 1-7...")
sgd = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000)
sgd.fit(X1, y)
probs.append(sgd.predict_proba(Xt1)[:,1])

# Model 4: LinearSVC calibrated
print("LinearSVC calibrated on 1-5...")
svc = CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=5000), method='sigmoid', cv=3)
svc.fit(X2, y)
probs.append(svc.predict_proba(Xt2)[:,1])

# Model 5: LR on 1-5 with C=200
print("LR C=200 on 1-5...")
m5 = LogisticRegression(C=200, max_iter=500, solver='lbfgs')
m5.fit(X2, y)
probs.append(m5.predict_proba(Xt2)[:,1])

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
    arr = np.clip(arr, 0.0, 1.0)
    with open(f'submissions/{name}.csv','w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(ids)):
            idc = str(ids.iloc[i]).replace('"','')
            f.write(f'"{idc}",{arr[i]}\n')
    print(f"Saved: {name}.csv ({arr.min():.4f}-{arr.max():.4f})")

save('aggressive_ensemble', ens)
save('aggressive_tjflexic', tj)

# Also save individual best models
save('aggressive_lr30', probs[0])
save('aggressive_lr100', probs[1])

print("\nDone! Submit aggressive_tjflexic.csv")