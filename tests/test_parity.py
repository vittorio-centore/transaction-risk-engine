#!/usr/bin/env python3
"""
API vs Offline Parity Test
Verifies that API scoring matches offline model scoring
"""

import asyncio
import httpx
import torch
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, '/Users/vittorioc/transaction-risk')

from model.model import FraudMLP
from service.features import compute_all_features
from sqlalchemy import select
from db.models import Transaction, db

async def test_parity():
    """Test if API matches offline scoring"""
    
    print("=" * 70)
    print("API vs OFFLINE PARITY TEST")
    print("=" * 70)
    
    # Load model and scaler (offline)
    print("\n📦 Loading model and scaler...")
    checkpoint = torch.load('models/fraud_model_v1.pt', map_location='cpu')
    scaler = joblib.load('models/scaler.pkl')
    model = FraudMLP(input_dim=38)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Get 200 random transactions from database
    print("\n📂 Loading 200 validation transactions...")
    async with db.async_session() as session:
        query = select(Transaction).limit(200).offset(10000)  # skip first 10k (training)
        result = await session.execute(query)
        transactions = result.scalars().all()
    
    print(f"✅ Loaded {len(transactions)} transactions")
    
    # Score offline
    print("\n🔬 Scoring OFFLINE...")
    offline_scores = []
    api_scores = []
    
    for i, tx in enumerate(transactions[:50]):  # test first 50 for speed
        if i % 10 == 0:
            print(f"   Processing {i}/50...")
        
        # Compute features offline
        v_features = {f'v{j}': float(getattr(tx, f'v{j}')) for j in range(1, 29)}
        feature_vector = await compute_all_features(
            user_id=tx.user_id,
            merchant_id=tx.merchant_id,
            amount=float(tx.amount),
            timestamp=tx.timestamp,
            v_features=v_features,
            exclude_tx_id=tx.transaction_id
        )
        
        # Scale and score offline
        feature_scaled = scaler.transform([feature_vector])[0]
        with torch.no_grad():
            offline_prob = torch.sigmoid(model(torch.FloatTensor([feature_scaled]))).item()
        offline_scores.append(offline_prob)
        
        # Score via API
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    "http://localhost:8000/score",
                    json={
                        "user_id": tx.user_id,
                        "merchant_id": tx.merchant_id,
                        "amount": float(tx.amount)
                    }
                )
                api_prob = response.json()['fraud_score']
                api_scores.append(api_prob)
            except Exception as e:
                print(f"   ⚠️  API error for tx {tx.transaction_id}: {e}")
                api_scores.append(None)
    
    # Remove None values
    valid_pairs = [(o, a) for o, a in zip(offline_scores, api_scores) if a is not None]
    offline_scores = [p[0] for p in valid_pairs]
    api_scores = [p[1] for p in valid_pairs]
    
    # Compare
    print(f"\n📊 RESULTS (n={len(offline_scores)}):")
    print(f"{'Metric':<30} {'Offline':<15} {'API':<15} {'Match?'}")
    print("-" * 70)
    
    offline_arr = np.array(offline_scores)
    api_arr = np.array(api_scores)
    
    print(f"{'Mean score':<30} {offline_arr.mean():<15.6f} {api_arr.mean():<15.6f}")
    print(f"{'Std score':<30} {offline_arr.std():<15.6f} {api_arr.std():<15.6f}")
    print(f"{'Min score':<30} {offline_arr.min():<15.6f} {api_arr.min():<15.6f}")
    print(f"{'Max score':<30} {offline_arr.max():<15.6f} {api_arr.max():<15.6f}")
    
    # Most important: max absolute difference
    abs_diff = np.abs(offline_arr - api_arr)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    print(f"\n🎯 PARITY METRICS:")
    print(f"   Max absolute diff: {max_diff:.10f}")
    print(f"   Mean absolute diff: {mean_diff:.10f}")
    print(f"   Median absolute diff: {np.median(abs_diff):.10f}")
    
    # Show some examples
    print(f"\n📋 SAMPLE COMPARISONS:")
    print(f"{'Offline':<20} {'API':<20} {'Diff':<20}")
    print("-" * 60)
    for i in range(min(10, len(offline_scores))):
        diff = abs(offline_scores[i] - api_scores[i])
        print(f"{offline_scores[i]:<20.10f} {api_scores[i]:<20.10f} {diff:<20.10f}")
    
    # Verdict
    print(f"\n{'='*70}")
    if max_diff < 1e-6:
        print("✅ PARITY PASS: API matches offline (max diff < 1e-6)")
    elif max_diff < 1e-4:
        print("⚠️  PARITY MARGINAL: Small differences (max diff < 1e-4)")
    else:
        print(f"❌ PARITY FAIL: Significant mismatch (max diff = {max_diff:.6f})")
        print("   → Check preprocessing, feature order, or scaler")
    print(f"{'='*70}")

if __name__ == '__main__':
    asyncio.run(test_parity())
