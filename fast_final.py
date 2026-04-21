"""Fast best model generator - optimized for speed"""
import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values

# Minimal stop words, keep negation
STOP = {'the','a','an','and','or','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','shall','should','can','could','may','might','must'} - {'not','no','never','nor','neither','hardly','barely'}

def p(t):
    t = t.lower().replace("n't"," not")
    t = re.sub(r'<[^>]+>',' ',t)
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    return ' '.join(w for w in t.split() if w not in STOP and len(w)>1)

print("Prep...")
tr = [p(t) for t in train['review']]
te = [p(t) for t in test['review']]

# TF-IDF with optimal settings
print("TFIDF...")
tf = TfidfVectorizer(ngram_range=(1,5), max_features=80000, sublinear_tf=True, min_df=1)
X = tf.fit_transform(tr)
Xt = tf.transform(te)

# Train multiple models
print("Train...")
m1 = LogisticRegression(C=30, max_iter=1000, solver='saga')
m1.fit(X, y)
p1 = m1.predict_proba(Xt)[:,1]

m2 = LogisticRegression(C=50, max_iter=1000, solver='saga')
m2.fit(X, y)
p2 = m2.predict_proba(Xt)[:,1]

m3 = LogisticRegression(C=100, max_iter=1000, solver='saga')
m3.fit(X, y)
p3 = m3.predict_proba(Xt)[:,1]

# Ensemble
ens = np.mean([p1,p2,p3], axis=0)

# TJflexic post-process
tj = np.copy(ens)
for i in range(len(ens)):
    if ens[i] > 0.5:
        tj[i] = max(p1[i], p2[i], p3[i])
    else:
        tj[i] = min(p1[i], p2[i], p3[i])

# Save
os.makedirs('submissions',exist_ok=True)

def save(name, probs):
    ids = test['id']
    with open(f'submissions/{name}.csv','w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(ids)):
            idc = str(ids.iloc[i]).replace('"','')
            f.write(f'"{idc}",{probs[i]}\n')
    print(f"Saved: {name}.csv")

save('final_c30', p1)
save('final_c50', p2)
save('final_c100', p3)
save('final_ensemble3', tj)

print("Done!")