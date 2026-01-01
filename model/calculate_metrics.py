#!/usr/bin/env python3
"""Calculate business metrics and optimal thresholds"""

import asyncio
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import FraudMLP
from model.train import load_training_data
from service.config import settings
import joblib

async def calculate_business_metrics():
    print("=" * 70)
    print("BUSINESS METRICS CALCULATION")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    X, y, tx_ids = await load_training_data(limit=50000)
    
    # Time-based split
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"✅ Loaded {len(X)} transactions")
    print(f"   Test set: {len(X_test)} ({y_test.sum()} fraud)")
    
    # Load scaler and model
    scaler = joblib.load('models/scaler.pkl')
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    model = FraudMLP(input_dim=X.shape[1])
    checkpoint = torch.load('models/fraud_model_v1.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.FloatTensor(X_val_scaled))).numpy()
        test_probs = torch.sigmoid(model(torch.FloatTensor(X_test_scaled))).numpy()
    
    print("\n📊 BUSINESS METRICS:")
    print("=" * 70)
    
    # 1. Recall @ 1% FPR
    legit_scores = test_probs[y_test == 0]
    fraud_scores = test_probs[y_test == 1]
    
    fpr_target = 0.01
    threshold_1pct_fpr = np.percentile(legit_scores, 100 * (1 - fpr_target))
    recall_at_1pct_fpr = (fraud_scores >= threshold_1pct_fpr).sum() / len(fraud_scores)
    
    print(f"\n📍 Recall @ 1% FPR: {recall_at_1pct_fpr:.2%}")
    print(f"   Threshold: {threshold_1pct_fpr:.4f}")
    print(f"   → Catches {recall_at_1pct_fpr:.1%} of fraud with only 1% false positives")
    
    # 2. Precision @ top 1% review rate
    review_rate = 0.01
    threshold_review_test = np.percentile(test_probs, 100 * (1 - review_rate))
    flagged = test_probs >= threshold_review_test
    precision_at_review = y_test[flagged].sum() / flagged.sum() if flagged.sum() > 0 else 0
    
    print(f"\n📍 Precision @ 1% review rate: {precision_at_review:.2%}")
    print(f"   Threshold: {threshold_review_test:.4f}")
    print(f"   → Of top 1% flagged, {precision_at_review:.1%} are actually fraud")
    
    # 3. Calculate decision thresholds from validation set
    T_review = float(np.percentile(val_probs, 99))  # Top 1% → review
    T_decline = float(np.percentile(val_probs, 99.9))  # Top 0.1% → decline
    
    print(f"\n🎯 DECISION THRESHOLDS (from validation set):")
    print(f"   Approve:  < {T_review:.4f}")
    print(f"   Review:   >= {T_review:.4f} (top 1%)")
    print(f"   Decline:  >= {T_decline:.4f} (top 0.1%)")
    
    # Apply to test set
    decisions_test = np.where(test_probs >= T_decline, 'decline',
                             np.where(test_probs >= T_review, 'review', 'approve'))
    
    print(f"\n📊 Test Set Distribution:")
    print(f"{'Decision':<10} {'Count':<8} {'%':<8} {'Fraud':<8} {'Precision'}")
    print("-" * 50)
    for decision in ['approve', 'review', 'decline']:
        mask = decisions_test == decision
        count = mask.sum()
        pct = count / len(decisions_test) * 100
        fraud_count = y_test[mask].sum()
        precision = fraud_count / count if count > 0 else 0
        print(f"{decision.capitalize():<10} {count:<8} {pct:>5.1f}%   {fraud_count:<8} {precision:>6.1%}")
    
    # 4. Standard PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, test_probs)
    
    print(f"\n📈 Standard Metrics:")
    print(f"   PR-AUC:  {pr_auc:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    # Save thresholds
    threshold_info = {
        'T_approve_max': float(T_review),
        'T_review': float(T_review),
        'T_decline': float(T_decline),
        'recall_at_1pct_fpr': float(recall_at_1pct_fpr),
        'precision_at_1pct_review': float(precision_at_review),
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc)
    }
    
    with open('models/thresholds.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)
    
    print(f"\n✅ Saved thresholds to models/thresholds.json")
    
    return threshold_info

if __name__ == '__main__':
    asyncio.run(calculate_business_metrics())
