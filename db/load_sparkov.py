#!/usr/bin/env python3
"""
Sparkov dataset loader - CLEAN VERSION with surrogate keys
"""

import pandas as pd
import asyncio
import hashlib
from datetime import datetime
from sqlalchemy import select, func
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.models import User, Merchant, Transaction, db

async def load_sparkov_data():
    """Load Sparkov dataset into database with surrogate keys"""
    
    print("=" * 70)
    print("SPARKOV DATASET MIGRATION")
    print("=" * 70)
    
    # Read CSV
    print("\n📂 Loading CSV...")
    df = pd.read_csv('data/sparkov/fraudTrain.csv')
    print(f"✅ Loaded {len(df):,} transactions")
    
    # Show fraud distribution
    fraud_count = df['is_fraud'].sum()
    fraud_pct = fraud_count / len(df) * 100
    print(f"   Fraud: {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"   Users: {df['cc_num'].nunique():,}")
    print(f"   Merchants: {df['merchant'].nunique():,}")
    print(f"   Categories: {df['category'].nunique()}")
    
   # Create database tables
    print("\n🗄️  Creating tables...")
    async with db.engine.begin() as conn:
        from db.models import Base
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created")
    
    # Load users with surrogate keys + hashed card numbers
    print("\n👥 Loading users...")
    users_df = df[['cc_num']].drop_duplicates('cc_num')
    
    # Create hash mapping for later lookup
    cc_to_hash = {str(row['cc_num']): hashlib.sha256(str(row['cc_num']).encode()).hexdigest() 
                  for _, row in users_df.iterrows()}
    
    async with db.async_session() as session:
        for idx, row in users_df.iterrows():
            cc_hash = cc_to_hash[str(row['cc_num'])]
            user = User(cc_num_hash=cc_hash)
            session.add(user)
            
            if idx % 500 == 0 and idx > 0:
                await session.commit()
                print(f"   Processed {idx}/{len(users_df)} users...")
        
        await session.commit()
    
    # Build user_id mapping after commit
    user_map = {}
    async with db.async_session() as session:
        result = await session.execute(select(User.user_id, User.cc_num_hash))
        hash_to_id = {cc_hash: uid for uid, cc_hash in result.all()}
        
        for cc_num, cc_hash in cc_to_hash.items():
            user_map[int(cc_num)] = hash_to_id[cc_hash]
    
    print(f"✅ Loaded {len(users_df):,} users")
    
    # Load merchants
    print("\n🏪 Loading merchants...")
    merchants_df = df[['merchant']].drop_duplicates('merchant')
    
    async with db.async_session() as session:
        for idx, row in merchants_df.iterrows():
            merchant = Merchant(name=row['merchant'])
            session.add(merchant)
            
            if idx % 500 == 0 and idx > 0:
                await session.commit()
                print(f"   Processed {idx}/{len(merchants_df)} merchants...")
        
        await session.commit()
    print(f"✅ Loaded {len(merchants_df):,} merchants")
    
    # Get merchant ID mapping
    print("\n🔗 Creating merchant ID mapping...")
    async with db.async_session() as session:
        result = await session.execute(select(Merchant.merchant_id, Merchant.name))
        merchant_map = {name: mid for mid, name in result.all()}
    
    # Load transactions (sample 50k for speed)
    print("\n💳 Loading transactions (sampling 50k)...")
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    
    async with db.async_session() as session:
        loaded = 0
        for idx, row in sample_df.iterrows():
            user_id = user_map.get(row['cc_num'])
            merchant_id = merchant_map.get(row['merchant'])
            
            if user_id is None or merchant_id is None:
                continue
            
            tx = Transaction(
                user_id=user_id,
                merchant_id=merchant_id,
                amount=float(row['amt']),
                timestamp=pd.to_datetime(row['trans_date_trans_time']),
                label=int(row['is_fraud']),
                category=row['category'],
                first_name=row['first'],
                last_name=row['last'],
                gender=row['gender'],
                street=row['street'],
                city=row['city'],
                state=row['state'],
                zip_code=str(row['zip']),
                lat=float(row['lat']),
                long=float(row['long']),
                city_pop=int(row['city_pop']) if pd.notna(row['city_pop']) else None,
                job=row['job'] if pd.notna(row['job']) else None,
                dob=pd.to_datetime(row['dob']) if pd.notna(row['dob']) else None,
                merch_lat=float(row['merch_lat']),
                merch_long=float(row['merch_long'])
            )
            session.add(tx)
            loaded += 1
            
            if loaded % 1000 == 0:
                await session.commit()
                print(f"   Processed {loaded:,} transactions...")
        
        await session.commit()
    print(f"✅ Loaded {loaded:,} transactions")
    
    # Verify
    print("\n✅ Migration complete!")
    async with db.async_session() as session:
        user_count = (await session.execute(select(func.count(User.user_id)))).scalar()
        merchant_count = (await session.execute(select(func.count(Merchant.merchant_id)))).scalar()
        tx_count = (await session.execute(select(func.count(Transaction.transaction_id)))).scalar()
        fraud_count = (await session.execute(
            select(func.count(Transaction.transaction_id)).where(Transaction.label == 1)
        )).scalar()
        
        print(f"   Users: {user_count:,}")
        print(f"   Merchants: {merchant_count:,}")
        print(f"   Transactions: {tx_count:,}")
        print(f"   Fraud: {fraud_count:,} ({fraud_count/tx_count*100:.2f}%)")

if __name__ == '__main__':
    asyncio.run(load_sparkov_data())
