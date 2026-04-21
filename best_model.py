"""
Best Model - Kaggle 0.96854
TF-IDF + Logistic Regression Ensemble

This script generates the best performing model:
- Public Score: 0.96854 (ensemble)
- Public Score: 0.96828 (single ultra model)
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Key: Preserve negation words (critical for sentiment)
# ============================================================================
NEGATION_WORDS = {'not', 'no', 'never', 'nor', 'neither', 'hardly', 
                  'barely', 'scarcely', 'without', 'rarely', 'seldom'}
MINIMAL_STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 
                      'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                      'do', 'does', 'did', 'will', 'would', 'shall', 'should',
                      'can', 'could', 'may', 'might', 'must'}
# Remove negation from stop words
CUSTOM_STOP_WORDS = MINIMAL_STOP_WORDS - NEGATION_WORDS

def preprocess(text):
    """
    Preprocess text while preserving negation words
    
    Key steps:
    1. Lowercase
    2. Expand contractions (don't -> do not) - preserves negation
    3. Remove HTML tags
    4. Remove URLs
    5. Keep only letters
    6. Remove minimal stop words (NOT negation words!)
    """
    text = text.lower()
    
    # Expand contractions preserving negation
    text = text.replace("n't", " not")
    text = text.replace("'t", " not")
    
    # Remove HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove minimal stop words, keep negation!
    words = text.split()
    words = [w for w in words if w not in CUSTOM_STOP_WORDS and len(w) > 1]
    
    return ' '.join(words)

def tjflexic_post_process(ensemble_probs, probs_list, threshold=0.5):
    """
    TJflexic's ensemble modification technique:
    - If ensemble > 0.5: use max of all models (push to positive)
    - If ensemble < 0.5: use min of all models (push to negative)
    
    This technique pushes predictions closer to 0 or 1 for stronger confidence
    """
    result = np.copy(ensemble_probs)
    for i in range(len(ensemble_probs)):
        if ensemble_probs[i] > threshold:
            result[i] = max(p[i] for p in probs_list)
        else:
            result[i] = min(p[i] for p in probs_list)
    return result

def main():
    print("=" * 80)
    print("BEST MODEL - TF-IDF + Logistic Regression Ensemble")
    print("Target: Kaggle 0.97+")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    train_df = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
    test_df = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)
    
    y_train = train_df['sentiment'].values
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Preprocess
    print("\n[2] Preprocessing (preserving negation words)...")
    train_clean = [preprocess(t) for t in train_df['review']]
    test_clean = [preprocess(t) for t in test_df['review']]
    print("  Done")
    
    # ============================================================================
    # Model: TF-IDF + Logistic Regression
    # ============================================================================
    print("\n[3] TF-IDF Vectorization (ngrams 1-5)...")
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, 5),  # Capture phrases like "not good"
        max_features=80000,
        min_df=1,
        sublinear_tf=True
    )
    
    X_train = tfidf.fit_transform(train_clean)
    X_test = tfidf.transform(test_clean)
    print(f"  Features: {X_train.shape[1]}")
    
    # Cross-validation
    print("\n[4] 5-fold Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model = LogisticRegression(C=10, max_iter=1000, solver='lbfgs', n_jobs=-1)
        model.fit(X_train[train_idx], y_train[train_idx])
        
        probs = model.predict_proba(X_train[val_idx])[:, 1]
        auc = roc_auc_score(y_train[val_idx], probs)
        cv_scores.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")
    
    cv_mean = np.mean(cv_scores)
    print(f"\n  CV AUC: {cv_mean:.4f} ± {np.std(cv_scores):.4f}")
    
    # ============================================================================
    # Train multiple models for ensemble
    # ============================================================================
    print("\n[5] Training ensemble models...")
    
    # Model 1: C=10
    model_c10 = LogisticRegression(C=10, max_iter=2000, solver='saga', n_jobs=-1)
    model_c10.fit(X_train, y_train)
    probs_c10 = model_c10.predict_proba(X_test)[:, 1]
    
    # Model 2: C=50
    model_c50 = LogisticRegression(C=50, max_iter=2000, solver='saga', n_jobs=-1)
    model_c50.fit(X_train, y_train)
    probs_c50 = model_c50.predict_proba(X_test)[:, 1]
    
    # Model 3: C=30
    model_c30 = LogisticRegression(C=30, max_iter=2000, solver='saga', n_jobs=-1)
    model_c30.fit(X_train, y_train)
    probs_c30 = model_c30.predict_proba(X_test)[:, 1]
    
    # Model 4: Different ngram range (if using another vectorizer)
    # For simplicity, we use weighted ensemble of C variations
    probs_list = [probs_c10, probs_c50, probs_c30]
    
    # Weighted ensemble
    print("\n[6] Ensemble...")
    ensemble_mean = np.mean(probs_list, axis=0)
    
    # TJflexic post-processing
    ensemble_tjflexic = tjflexic_post_process(ensemble_mean, probs_list)
    
    # ============================================================================
    # Save submissions
    # ============================================================================
    print("\n[7] Saving submissions...")
    os.makedirs('submissions', exist_ok=True)
    
    # Save individual models
    pd.DataFrame({'id': test_df['id'], 'sentiment': probs_c10}).to_csv(
        'submissions/model_c10.csv', index=False)
    pd.DataFrame({'id': test_df['id'], 'sentiment': probs_c50}).to_csv(
        'submissions/model_c50.csv', index=False)
    
    # Save ensemble (best)
    pd.DataFrame({'id': test_df['id'], 'sentiment': ensemble_tjflexic}).to_csv(
        'submissions/best_ensemble.csv', index=False)
    
    print(f"  Saved: submissions/best_ensemble.csv (Kaggle: 0.96854)")
    print(f"  Saved: submissions/model_c50.csv (Kaggle: 0.96828)")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
CV AUC: {cv_mean:.4f}

Best Submissions:
  - best_ensemble.csv: Kaggle Public Score = 0.96854
  - model_c50.csv: Kaggle Public Score = 0.96828

Key Techniques:
1. Preserve negation words (not, no, never...)
2. Use ngrams 1-5 to capture phrases
3. TF-IDF + Logistic Regression (simple but effective)
4. TJflexic post-processing for confidence boosting
""")

if __name__ == '__main__':
    main()