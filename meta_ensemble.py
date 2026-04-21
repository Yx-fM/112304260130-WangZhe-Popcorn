"""Meta-ensemble: Combine existing submissions for better predictions"""
import numpy as np, pandas as pd, os, glob

print("Loading existing submissions...")
ids = None
predictions = []

# Load all good submissions
files = [
    'submissions/best_ensemble.csv',      # Kaggle 0.96854
    'submissions/nbsvm_tjflexic.csv',     # NBSVM + TJflexic
    'submissions/word_char_tjflexic.csv', # Word+Char + TJflexic
    'submissions/super_ensemble.csv',     # High C ensemble
    'submissions/final_ensemble3.csv',    # 3-model ensemble
]

for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if ids is None:
            ids = df['id']
        p = df['sentiment'].values
        predictions.append(p)
        print(f"  {f}: {p.min():.4f}-{p.max():.4f}")

print(f"\nLoaded {len(predictions)} submissions")

# Simple mean ensemble
ens_mean = np.mean(predictions, axis=0)

# TJflexic meta-ensemble
tj = np.copy(ens_mean)
for i in range(len(ens_mean)):
    if ens_mean[i] > 0.5:
        tj[i] = max(p[i] for p in predictions)
    else:
        tj[i] = min(p[i] for p in predictions)

# Weighted ensemble (best submission gets higher weight)
weights = [0.4, 0.2, 0.2, 0.1, 0.1]  # best_ensemble gets 40%
ens_weighted = np.zeros(len(predictions[0]))
for i, p in enumerate(predictions):
    ens_weighted += weights[i] * p

# Save
os.makedirs('submissions',exist_ok=True)

def save(name, arr):
    with open(f'submissions/{name}.csv','w') as f:
        f.write('"id","sentiment"\n')
        for i in range(len(ids)):
            idc = str(ids.iloc[i]).replace('"','')
            f.write(f'"{idc}",{arr[i]}\n')
    print(f"Saved: {name}.csv ({arr.min():.4f}-{arr.max():.4f})")

save('meta_ensemble_mean', ens_mean)
save('meta_ensemble_tj', tj)
save('meta_ensemble_weighted', ens_weighted)

print("\nDone! Submit meta_ensemble_tj.csv for best results.")