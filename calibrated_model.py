"""Ultra-optimized TF-IDF with calibration and advanced features"""
import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

print("Loading...")
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values
ids = test['id']

# Ultra preprocessing - keep all sentiment signals
def prep(t):
    t = str(t).lower()
    # Preserve negation patterns
    t = t.replace("n't"," not")
    t = t.replace("'t"," not")  
    t = t.replace("never"," never ")
    t = t.replace("no"," no ")
    # Remove HTML but keep text
    t = re.sub(r'<[^>]+>',' ',t)
    # Keep only letters and spaces
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    # Remove very short words but keep negation
    NEG = {'not','no','nor','never','neither','hardly','barely','scarcely'}
    words = [w for w in t.split() if len(w)>1 or w in NEG]
    return ' '.join(words)

print("Preprocessing...")
tr = [prep(t) for t in train['review']]
te = [prep(t) for t in test['review']]

# Best config: 1-5 ngrams, 100000 features, sublinear_tf
print("TF-IDF...")
tf = TfidfVectorizer(
    ngram_range=(1,5),
    max_features=100000,
    sublinear_tf=True,
    min_df=1,
    max_df=0.95
)
X = tf.fit_transform(tr)
Xt = tf.transform(te)
print(f"Features: {X.shape[1]}")

# Train multiple models with different C values
probs = []

for C in [10, 30, 50, 100, 200]:
    print(f"Training C={C}...")
    base = LogisticRegression(C=C, max_iter=500, solver='lbfgs')
    calibrated = CalibratedClassifierCV(base, method='sigmoid', cv=3)
    calibrated.fit(X, y)
    probs.append(calibrated.predict_proba(Xt)[:,1])

# Ensemble
ens = np.mean(probs, axis=0)

# TJflexic
tj = np.copy(ens)
for i in range(len(ens)):
    if ens[i] > 0.5:
        tj[i] = max(p[i] for p in probs)
    else:
        tj[i] = min(p[i] for p in probs)

# Median ensemble (more robust)
med = np.median(probs, axis=1)

# Save
os.makedirs('submissions',exist_ok=True)

def save(name, arr):
    with open(f'submissions/{name}.csv','w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(ids)):
            idc = str(ids.iloc[i]).replace('"','')
            f.write(f'"{idc}",{arr[i]}\n')
    print(f"Saved: {name}.csv ({arr.min():.4f}-{arr.max():.4f})")

save('calibrated_ensemble', ens)
save('calibrated_tjflexic', tj)

print("\nDone! Submit calibrated_tjflexic.csv")