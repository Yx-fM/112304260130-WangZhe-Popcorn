"""
Grid Search for Best Parameters
Quick optimization to find best C and ngram combinations
"""

import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values

# Minimal preprocessing
STOP = {'the','a','an','and','or','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','shall','should','can','could','may','might','must'} - {'not','no','never','nor','neither','hardly','barely'}

def p(t):
    t = t.lower().replace("n't"," not")
    t = re.sub(r'<[^>]+>',' ',t)
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    return ' '.join(w for w in t.split() if w not in STOP and len(w)>1)

print("Preprocessing...")
tr = [p(t) for t in train['review']]
te = [p(t) for t in test['review']]

# Quick grid search on subset
print("\nGrid search (subset 5000 samples)...")
tr_small = tr[:5000]
y_small = y[:5000]

best_config = None
best_auc = 0.0

configs = [
    {'ngram': (1,4), 'feat': 50000, 'C': 10},
    {'ngram': (1,4), 'feat': 60000, 'C': 20},
    {'ngram': (1,5), 'feat': 60000, 'C': 10},
    {'ngram': (1,5), 'feat': 70000, 'C': 30},
    {'ngram': (1,5), 'feat': 80000, 'C': 50},
    {'ngram': (1,6), 'feat': 80000, 'C': 30},
]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for cfg in configs:
    tf = TfidfVectorizer(ngram_range=cfg['ngram'], max_features=cfg['feat'], sublinear_tf=True)
    X = tf.fit_transform(tr_small)
    
    scores = []
    for tr_i, val_i in skf.split(X, y_small):
        m = LogisticRegression(C=cfg['C'], max_iter=300, solver='lbfgs')
        m.fit(X[tr_i], y_small[tr_i])
        auc = roc_auc_score(y_small[val_i], m.predict_proba(X[val_i])[:,1])
        scores.append(auc)
    
    avg = np.mean(scores)
    if avg > best_auc:
        best_auc = avg
        best_config = cfg
    print(f"  ngram={cfg['ngram']}, feat={cfg['feat']}, C={cfg['C']}: AUC={avg:.4f}")

print(f"\nBest: ngram={best_config['ngram']}, feat={best_config['feat']}, C={best_config['C']}")

# Train with best config on full data
print("\nTraining with best config...")
tf_best = TfidfVectorizer(ngram_range=best_config['ngram'], max_features=best_config['feat'], sublinear_tf=True)
X_full = tf_best.fit_transform(tr)
X_test = tf_best.transform(te)

# Multiple C values around best
probs_list = []
for c in [best_config['C'], best_config['C']*2, best_config['C']*3]:
    m = LogisticRegression(C=c, max_iter=500, solver='saga')
    m.fit(X_full, y)
    probs_list.append(m.predict_proba(X_test)[:,1])

# Ensemble
ens = np.mean(probs_list, axis=0)

# TJflexic
tj = np.copy(ens)
for i in range(len(ens)):
    if ens[i] > 0.5:
        tj[i] = max(p[i] for p in probs_list)
    else:
        tj[i] = min(p[i] for p in probs_list)

# Save
os.makedirs('submissions',exist_ok=True)
ids = test['id']

def save(name, probs):
    with open(f'submissions/{name}.csv','w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(ids)):
            idc = str(ids.iloc[i]).replace('"','')
            f.write(f'"{idc}",{probs[i]}\n')
    print(f"Saved: {name}.csv")

save('grid_best_ensemble', tj)
save('grid_best_mean', ens)

print("\nDone!")