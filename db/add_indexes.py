#!/usr/bin/env python3
"""Add database indexes for faster feature queries"""

import asyncio
from sqlalchemy import text
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.models import db

async def add_indexes():
    print("Adding database indexes for faster queries...")
    
    async with db.engine.begin() as conn:
        # Index for user velocity queries (user_id + timestamp)
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tx_user_time 
            ON transactions(user_id, timestamp DESC)
        """))
        print("✅ Added idx_tx_user_time")
        
        # Index for merchant queries (merchant_id + timestamp)
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tx_merchant_time 
            ON transactions(merchant_id, timestamp DESC)
        """))
        print("✅ Added idx_tx_merchant_time")
        
        # Index for label queries
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tx_label 
            ON transactions(label)
        """))
        print("✅ Added idx_tx_label")
    
    print("\n✅ All indexes added! Training should be much faster now.")

if __name__ == '__main__':
    asyncio.run(add_indexes())
