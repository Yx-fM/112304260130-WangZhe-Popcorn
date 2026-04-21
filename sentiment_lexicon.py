"""Sentiment lexicon-enhanced TF-IDF"""
import numpy as np, pandas as pd, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

print("Loading...")
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
y = train['sentiment'].values
ids = test['id']

# Positive and negative sentiment words
POSITIVE = {
    'good','great','excellent','amazing','wonderful','fantastic','brilliant','superb','awesome',
    'love','loved','loving','like','liked','enjoy','enjoyed','best','perfect','beautiful',
    'happy','happily','excited','exciting','fun','funny','hilarious','entertaining','interesting',
    'recommend','recommended','worth','worthwhile','satisfying','satisfied','pleased','pleasing',
    'impressive','impressed','outstanding','remarkable','incredible','marvelous','terrific'
}

NEGATIVE = {
    'bad','terrible','awful','horrible','poor','worst','badly','waste','wasted','boring',
    'disappoint','disappointed','disappointing','dislike','disliked','hate','hated','hating',
    'annoy','annoyed','annoying','frustrate','frustrated','frustrating','angry','angrily',
    'sad','sadly','unhappy','miserable','dull','stupid','ridiculous','absurd','pathetic',
    'fail','failed','failure','mess','messy','confuse','confused','confusing','mediocre'
}

NEGATION = {'not','no','never','nor','neither','hardly','barely','scarcely'}

def prep(t):
    t = str(t).lower()
    t = t.replace("n't"," not").replace("'t"," not")
    t = re.sub(r'<[^>]+>',' ',t)
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    return ' '.join(w for w in t.split() if len(w)>1)

print("Preprocessing...")
tr = [prep(t) for t in train['review']]
te = [prep(t) for t in test['review']]

# TF-IDF features
print("TF-IDF...")
tf = TfidfVectorizer(ngram_range=(1,5), max_features=80000, sublinear_tf=True)
X_tfidf = tf.fit_transform(tr)
Xt_tfidf = tf.transform(te)

# Sentiment lexicon features
print("Sentiment features...")
def sentiment_features(texts):
    features = []
    for t in texts:
        words = set(t.split())
        pos_count = len([w for w in words if w in POSITIVE])
        neg_count = len([w for w in words if w in NEGATIVE])
        negation_count = len([w for w in words if w in NEGATION])
        # Negation flips sentiment
        effective_pos = pos_count - negation_count if negation_count > 0 else pos_count
        effective_neg = neg_count + negation_count if negation_count > 0 else neg_count
        features.append([pos_count, neg_count, negation_count, effective_pos, effective_neg])
    return csr_matrix(np.array(features))

X_sent = sentiment_features(tr)
Xt_sent = sentiment_features(te)

# Combine
X = hstack([X_tfidf, X_sent]).tocsr()
Xt = hstack([Xt_tfidf, Xt_sent]).tocsr()
print(f"Features: {X.shape[1]}")

# Train models
probs = []

for C in [10, 30, 50]:
    print(f"Training C={C}...")
    m = LogisticRegression(C=C, max_iter=300, solver='lbfgs')
    m.fit(X, y)
    probs.append(m.predict_proba(Xt)[:,1])

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

save('sentiment_lexicon_ensemble', ens)
save('sentiment_lexicon_tjflexic', tj)

print("\nDone! Submit sentiment_lexicon_tjflexic.csv")