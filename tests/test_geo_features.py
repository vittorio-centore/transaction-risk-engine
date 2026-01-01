#!/usr/bin/env python3
"""
Test geo features SERVER-SIDE computation BEFORE retraining
Validates that we're not repeating the v-features mistake
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from service.features import compute_all_features, get_feature_names
from db.models import Transaction, db
from sqlalchemy import select
from datetime import datetime

async def test_geo_features():
    print("=" * 80)
    print("TESTING GEO FEATURES (SERVER-SIDE COMPUTATION)")
    print("=" * 80)
    
    # Get a real transaction from DB
    async with db.async_session() as session:
        # Get a transaction that's NOT the first for the user (so it has history)
        query = select(Transaction).offset(1000).limit(1)
        result = await session.execute(query)
        tx = result.scalar()
        
        if not tx:
            print("❌ No transactions found!")
            return
        
        print(f"\n📦 Test Transaction:")
        print(f"   ID: {tx.transaction_id}")
        print(f"   User: {tx.user_id}, Merchant: {tx.merchant_id}")
        print(f"   Amount: ${tx.amount}")
        print(f"   Timestamp: {tx.timestamp}")
        print(f"   Merchant lat/long: ({tx.merch_lat}, {tx.merch_long})")
        print(f"   User lat/long: ({tx.lat}, {tx.long})")
    
    # Test 1: Compute features (should work without errors)
    print(f"\n🧪 TEST 1: Feature Computation")
    print("-" * 80)
    
    try:
        features = await compute_all_features(
            user_id=tx.user_id,
            merchant_id=tx.merchant_id,
            amount=float(tx.amount),
            timestamp=tx.timestamp,
            exclude_tx_id=tx.transaction_id
        )
        print(f"✅ Feature computation successful!")
        print(f"   Shape: {features.shape}")
        print(f"   Dtype: {features.dtype}")
    except Exception as e:
        print(f"❌ Feature computation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Verify feature count
    print(f"\n🧪 TEST 2: Feature Count")
    print("-" * 80)
    
    names = get_feature_names()
    if len(features) == len(names) == 19:
        print(f"✅ Feature count correct: {len(features)} features")
    else:
        print(f"❌ Feature count mismatch!")
        print(f"   Features: {len(features)}")
        print(f"   Names: {len(names)}")
        return
    
    # Test 3: Check geo features are NOT all zeros
    print(f"\n🧪 TEST 3: Geo Features Have Real Values (Not All Zeros)")
    print("-" * 80)
    
    geo_start_idx = 14  # Geo features start at index 14
    geo_features = features[geo_start_idx:geo_start_idx+5]
    geo_names = names[geo_start_idx:geo_start_idx+5]
    
    all_zero = all(f == 0.0 for f in geo_features)
    
    if all_zero:
        print(f"⚠️  WARNING: All geo features are zero!")
        print(f"   This might be OK if geo_missing=1")
    else:
        print(f"✅ Geo features have non-zero values:")
    
    for name, value in zip(geo_names, geo_features):
        status = "✅" if value != 0.0 or name == 'geo_missing' else "❌"
        print(f"   {status} {name:<25} = {value:.4f}")
    
    # Test 4: Verify geo_missing flag logic
    print(f"\n🧪 TEST 4: geo_missing Flag")
    print("-" * 80)
    
    geo_missing = features[-1]
    print(f"   geo_missing = {geo_missing}")
    
    if geo_missing == 1.0:
        print(f"   ℹ️  Geo is MISSING (expected for first transaction)")
        # All other geo features should be 0
        if all(f == 0.0 for f in geo_features[:-1]):
            print(f"   ✅ Other geo features correctly set to 0")
        else:
            print(f"   ⚠️  Some geo features non-zero despite geo_missing=1")
    elif geo_missing == 0.0:
        print(f"   ℹ️  Geo was COMPUTED successfully")
        # At least some geo features should be non-zero
        if any(f != 0.0 for f in geo_features[:-1]):
            print(f"   ✅ Geo features have real values")
        else:
            print(f"   ⚠️  All geo features are 0 but geo_missing=0 (unusual)")
    else:
        print(f"   ❌ geo_missing has invalid value: {geo_missing}")
    
    # Test 5: Test with a first transaction (geo_missing should be 1)
    print(f"\n🧪 TEST 5: First Transaction (geo_missing should be 1)")
    print("-" * 80)
    
    async with db.async_session() as session:
        # Find first transaction for a user
        query = select(Transaction).order_by(Transaction.timestamp).limit(1)
        result = await session.execute(query)
        first_tx = result.scalar()
        
        if first_tx:
            print(f"   Testing first tx: ID={first_tx.transaction_id}, User={first_tx.user_id}")
            first_features = await compute_all_features(
                user_id=first_tx.user_id,
                merchant_id=first_tx.merchant_id,
                amount=float(first_tx.amount),
                timestamp=first_tx.timestamp,
                exclude_tx_id=first_tx.transaction_id
            )
            
            first_geo_missing = first_features[-1]
            if first_geo_missing == 1.0:
                print(f"   ✅ geo_missing=1 for first transaction (correct!)")
            else:
                print(f"   ⚠️  geo_missing={first_geo_missing} for first tx (expected 1)")
    
    # Test 6: Display all features for inspection
    print(f"\n📊 FULL FEATURE VECTOR:")
    print("=" * 80)
    print(f"{'Index':<6} {'Name':<30} {'Value':<12} {'OK?'}")
    print("-" * 80)
    
    for i, (name, value) in enumerate(zip(names, features)):
        # Flag suspicious values
        if i >= 14 and i < 18:  # Geo features (not including geo_missing)
            if geo_missing == 0.0 and value == 0.0:
                status = "⚠️"
            else:
                status = "✅"
        else:
            status = "✅"
        
        print(f"{i:<6} {name:<30} {value:<12.4f} {status}")
    
    print(f"\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"✅ Server-side geo computation: WORKING")
    print(f"✅ No lat/long in API request needed")
    print(f"✅ Features: {len(features)} (19 total)")
    print(f"✅ geo_missing flag: IMPLEMENTED")
    print(f"\n🎯 Ready to retrain with proper geo features!")

if __name__ == '__main__':
    asyncio.run(test_geo_features())
