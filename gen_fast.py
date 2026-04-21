"""Fastest submission generator"""
import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values

# Minimal preprocessing
STOP = {'the','a','an','and','or','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','shall','should','can','could','may','might','must'}

def p(t):
    t = t.lower().replace("n't"," not")
    t = re.sub(r'<[^>]+>',' ',t)
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    return ' '.join(w for w in t.split() if w not in STOP and len(w)>1)

print("Prep...")
tr = [p(t) for t in train['review']]
te = [p(t) for t in test['review']]

# Single optimized model
print("TFIDF 50000...")
tf = TfidfVectorizer(ngram_range=(1,4), max_features=50000, sublinear_tf=True)
X = tf.fit_transform(tr)
Xt = tf.transform(te)

print("LR C=10...")
m = LogisticRegression(C=10, max_iter=500, solver='lbfgs')
m.fit(X, y)
probs = m.predict_proba(Xt)[:,1]

# Save
os.makedirs('submissions',exist_ok=True)
ids = test['id']
with open('submissions/fast_single.csv','w') as f:
    f.write('"id","sentiment"\n')
    for i in range(len(ids)):
        idc = str(ids.iloc[i]).replace('"','')
        f.write(f'"{idc}",{probs[i]}\n')
print("Saved: fast_single.csv")
print("Done!")