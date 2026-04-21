"""
RoBERTa Quick Fine-tuning for 0.98+ Kaggle Score
Uses smaller batch size and max_length for faster training
"""

import os
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Suppress tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
MAX_LEN = 128  # Reduced for speed
BATCH_SIZE = 16  # Smaller batch
EPOCHS = 2  # Fewer epochs
LR = 2e-5

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy())
    return np.array(probs)

print("=" * 70)
print("RoBERTa QUICK FINE-TUNING")
print("=" * 70)

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data
print("\nLoading...")
train = pd.read_csv('data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', sep='\t', quoting=3)

train_clean = [clean_text(t) for t in train['review']]
test_clean = [clean_text(t) for t in test['review']]
y = train['sentiment'].values

print(f"Train: {len(train)}, Test: {len(test)}")

# Tokenizer
print("\nLoading RoBERTa tokenizer...")
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
except:
    print("  Using mirror...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Cross-validation
print("\n3-fold CV (quick)...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

cv_scores = []
all_probs = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_clean, y)):
    print(f"\n  Fold {fold+1}...")
    
    # Split
    tr_texts = [train_clean[i] for i in tr_idx]
    tr_labels = y[tr_idx]
    val_texts = [train_clean[i] for i in val_idx]
    val_labels = y[val_idx]
    
    # Datasets
    tr_ds = ReviewDataset(tr_texts, tr_labels, tokenizer, MAX_LEN)
    val_ds = ReviewDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # Model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR)
    
    # Train
    for epoch in range(EPOCHS):
        loss = train_epoch(model, tr_loader, optimizer, device)
        print(f"    Epoch {epoch+1}: loss={loss:.4f}")
    
    # Predict
    val_probs = predict(model, val_loader, device)
    auc = roc_auc_score(val_labels, val_probs)
    cv_scores.append(auc)
    print(f"    AUC: {auc:.4f}")
    
    # Clean up
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\nCV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Final training on full data
print("\nFinal training...")
full_ds = ReviewDataset(train_clean, y, tokenizer, MAX_LEN)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ds = ReviewDataset(test_clean, [0]*len(test_clean), tokenizer, MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    loss = train_epoch(model, full_loader, optimizer, device)
    print(f"  Epoch {epoch+1}: loss={loss:.4f}")

# Predict
probs = predict(model, test_loader, device)

# Save
os.makedirs('submissions', exist_ok=True)
with open('submissions/roberta_quick.csv', 'w') as f:
    f.write('"id","sentiment"\n')
    for i in range(len(test)):
        idc = str(test['id'].iloc[i]).replace('"', '')
        f.write(f'"{idc}",{probs[i]}\n')

print("\nSaved: roberta_quick.csv")
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
CV AUC: {np.mean(cv_scores):.4f}

If CV AUC > 0.97, this model should help reach 0.98+ on Kaggle.
Submit roberta_quick.csv to verify.
""")