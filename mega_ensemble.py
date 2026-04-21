"""Ultimate Mega Ensemble: Combine ALL submissions"""
import numpy as np, pandas as pd, os

print("Loading ALL submissions...")
ids = None
predictions = []
names = []

# ALL submissions in order of expected quality
files = [
    ('submissions/best_ensemble.csv', 0.20),           # Kaggle 0.96854 (verified)
    ('submissions/aggressive_tjflexic.csv', 0.15),    # New aggressive model
    ('submissions/nbsvm_tjflexic.csv', 0.12),         # NBSVM
    ('submissions/calibrated_tjflexic.csv', 0.10),    # Calibrated
    ('submissions/word_char_tjflexic.csv', 0.10),     # Word+Char
    ('submissions/super_meta_tjflexic.csv', 0.08),    # Previous meta
    ('submissions/sentiment_lexicon_tjflexic.csv', 0.08), # Lexicon
    ('submissions/meta_ensemble_tj.csv', 0.07),       # Meta
    ('submissions/super_ensemble.csv', 0.05),         # High C
    ('submissions/final_ensemble3.csv', 0.05),        # Final
]

for f, w in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if ids is None:
            ids = df['id']
        p = df['sentiment'].values
        predictions.append(p)
        names.append(f.split('/')[-1].replace('.csv',''))
        print(f"  {names[-1]}: weight={w:.2f}, range={p.min():.4f}-{p.max():.4f}")

print(f"\nLoaded {len(predictions)} submissions")

# Weighted ensemble
weights = [w for _, w in files[:len(predictions)]]
weights = np.array(weights)
weights = weights / weights.sum()  # Normalize

ens_weighted = np.zeros(len(predictions[0]))
for i, p in enumerate(predictions):
    ens_weighted += weights[i] * p

# TJflexic mega
tj = np.copy(ens_weighted)
for i in range(len(ens_weighted)):
    if ens_weighted[i] > 0.5:
        tj[i] = max(p[i] for p in predictions)
    else:
        tj[i] = min(p[i] for p in predictions)

# Harmonic mean (good for combining probabilities)
def harmonic_mean(probs):
    n = len(probs)
    inv_sum = sum(1.0 / (p + 0.001) for p in probs)  # Avoid division by zero
    return n / inv_sum

hm = np.array([harmonic_mean([p[i] for p in predictions]) for i in range(len(predictions[0]))])

# Power mean (exponent = 3 emphasizes larger values)
power = 3
pow_mean = np.power(np.mean([np.power(p, power) for p in predictions], axis=0), 1.0/power)

# Median (most robust)
med = np.median(predictions, axis=1)

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

save('mega_ensemble_weighted', ens_weighted)
save('mega_ensemble_tjflexic', tj)
save('mega_ensemble_harmonic', hm)
save('mega_ensemble_power', pow_mean)
save('mega_ensemble_median', med)

print("\n=== ULTIMATE MEGA ENSEMBLE COMPLETE ===")
print("Submit mega_ensemble_tjflexic.csv for best results!")
print("This combines 10 submissions with weighted averaging + TJflexic.")