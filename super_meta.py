"""Super Meta-Ensemble: Combine ALL best submissions"""
import numpy as np, pandas as pd, os

print("Loading all submissions...")
ids = None
predictions = []

# All high-quality submissions
files = [
    'submissions/best_ensemble.csv',           # Kaggle 0.96854 (verified)
    'submissions/nbsvm_tjflexic.csv',          # NBSVM
    'submissions/word_char_tjflexic.csv',      # Word+Char
    'submissions/calibrated_tjflexic.csv',     # Calibrated
    'submissions/sentiment_lexicon_tjflexic.csv', # Lexicon
    'submissions/super_ensemble.csv',          # High C
    'submissions/meta_ensemble_tj.csv',        # Meta-ensemble
]

weights = [0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10]  # best_ensemble gets highest weight

for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if ids is None:
            ids = df['id']
        p = df['sentiment'].values
        predictions.append(p)
        print(f"  {f}: {p.min():.4f}-{p.max():.4f}")

print(f"\nLoaded {len(predictions)} submissions")

# Weighted ensemble
ens_weighted = np.zeros(len(predictions[0]))
for i, p in enumerate(predictions):
    ens_weighted += weights[i] * p

# TJflexic super
tj = np.copy(ens_weighted)
for i in range(len(ens_weighted)):
    if ens_weighted[i] > 0.5:
        tj[i] = max(p[i] for p in predictions)
    else:
        tj[i] = min(p[i] for p in predictions)

# Geometric mean (for probabilities)
geo = np.ones(len(predictions[0]))
for p in predictions:
    geo *= p
geo = geo ** (1.0 / len(predictions))

# Rank averaging
ranks = []
for p in predictions:
    rank = pd.Series(p).rank().values
    ranks.append(rank)
rank_avg = np.mean(ranks, axis=0)
rank_scaled = rank_avg / len(predictions[0])  # Scale to 0-1

# Save
os.makedirs('submissions',exist_ok=True)

def save(name, arr):
    # Clip to valid probability range
    arr = np.clip(arr, 0.0, 1.0)
    with open(f'submissions/{name}.csv','w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(ids)):
            idc = str(ids.iloc[i]).replace('"','')
            f.write(f'"{idc}",{arr[i]}\n')
    print(f"Saved: {name}.csv ({arr.min():.4f}-{arr.max():.4f})")

save('super_meta_weighted', ens_weighted)
save('super_meta_tjflexic', tj)
save('super_meta_geometric', geo)
save('super_meta_rank', rank_scaled)

print("\nDone! Submit super_meta_tjflexic.csv for best results!")
print("This combines 7 high-quality submissions with weighted averaging + TJflexic.")