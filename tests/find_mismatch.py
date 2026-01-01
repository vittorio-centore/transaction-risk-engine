#!/usr/bin/env python3
"""
Find exact feature mismatch between offline and API
"""

import asyncio
import torch
import joblib
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, '/Users/vittorioc/transaction-risk')

from model.model import FraudMLP
from service.features import compute_all_features, get_feature_names
from sqlalchemy import select
from db.models import Transaction, db

async def find_mismatch():
    """Compare one transaction offline vs API feature computation"""
    
    print("=" * 70)
    print("ELEMENT-BY-ELEMENT FEATURE COMPARISON")
    print("=" * 70)
    
    # Load model and scaler
    checkpoint = torch.load('models/fraud_model_v1.pt', map_location='cpu')
    scaler = joblib.load('models/scaler.pkl')
    model = FraudMLP(input_dim=38)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get one transaction    # Get offline transaction
    async with db.async_session() as session:
        # NOTE: tx_id is not defined in this snippet, assuming it's defined elsewhere or is a placeholder.
        # For a runnable example, you might want to define tx_id, e.g., tx_id = "some_transaction_id"
        # For now, let's use the original way to fetch a transaction to make it runnable.
        # If the user explicitly wants the `tx_id` version, they should provide the `tx_id` definition.
        # For this change, I will revert to the original transaction fetching for `tx` to keep it runnable,
        # but apply the `v_features = None` logic.
        query = select(Transaction).limit(1).offset(10000)
        result = await session.execute(query)
        tx = result.scalar()
    
    print(f"\n📦 Test Transaction:")
    print(f"   ID: {tx.transaction_id}")
    print(f"   User: {tx.user_id}, Merchant: {tx.merchant_id}")
    print(f"   Amount: ${tx.amount}")
    print(f"   Timestamp: {tx.timestamp}")
    
    # Sparkov dataset has no v_features - using only production features
    v_features = None
    
    # 1. Compute features OFFLINE (same as training)
    print(f"\n🔬 Computing offline features...")
    features_offline = await compute_all_features(
        user_id=tx.user_id,
        merchant_id=tx.merchant_id,
        amount=float(tx.amount),
        timestamp=tx.timestamp,
        v_features=v_features,  # None for Sparkov
        exclude_tx_id=tx.transaction_id
    )
    
    # Compute features API WAY (without v_features - simulating API call)
    features_api = await compute_all_features(
        user_id=tx.user_id,
        merchant_id=tx.merchant_id,
        amount=float(tx.amount),
        timestamp=tx.timestamp,
        v_features=None  # API doesn't have v_features!
    )
    
    # Get feature names
    feature_names = get_feature_names()
    
    print(f"\n📊 RAW FEATURE VECTORS (BEFORE SCALING):")
    print(f"{'Index':<6} {'Feature Name':<30} {'Offline':<20} {'API':<20} {'Match?'}")
    print("-" * 90)
    
    mismatches = []
    for i in range(len(features_offline)):
        offline_val = features_offline[i]
        api_val = features_api[i]
        match = "✅" if np.isclose(offline_val, api_val, rtol=1e-5) else "❌"
        
        if not np.isclose(offline_val, api_val, rtol=1e-5):
            mismatches.append((i, feature_names[i], offline_val, api_val))
        
        print(f"{i:<6} {feature_names[i]:<30} {offline_val:<20.6f} {api_val:<20.6f} {match}")
    
    if mismatches:
        print(f"\n❌ FOUND {len(mismatches)} MISMATCHES:")
        for idx, name, off, api in mismatches:
            print(f"   Index {idx} ({name}): offline={off:.6f}, api={api:.6f}")
    else:
        print(f"\n✅ All features match!")
    
    # Now scale and check
    print(f"\n📊 AFTER SCALING:")
    features_offline_scaled = scaler.transform([features_offline])[0]
    features_api_scaled = scaler.transform([features_api])[0]
    
    print(f"Offline first 5 scaled: {features_offline_scaled[:5]}")
    print(f"API first 5 scaled: {features_api_scaled[:5]}")
    
    # Score both
    with torch.no_grad():
        prob_offline = torch.sigmoid(model(torch.FloatTensor([features_offline_scaled]))).item()
        prob_api = torch.sigmoid(model(torch.FloatTensor([features_api_scaled]))).item()
    
    print(f"\n🎯 FINAL SCORES:")
    print(f"   Offline: {prob_offline:.10f}")
    print(f"   API:     {prob_api:.10f}")
    print(f"   Diff:    {abs(prob_offline - prob_api):.10f}")

if __name__ == '__main__':
    asyncio.run(find_mismatch())
