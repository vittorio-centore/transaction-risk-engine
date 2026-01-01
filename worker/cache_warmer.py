# background worker for cache warming and maintenance
# runs scheduled jobs to keep cache fresh for active users

import asyncio
from datetime import datetime, timedelta
from sqlalchemy import select, func, distinct
from typing import List

from db.models import Transaction, db
from service.features import compute_user_velocity_features, compute_merchant_baseline
from service.cache import user_cache, merchant_cache

class CacheWarmer:
    """
    background worker that pre-warms cache for active users/merchants
    runs periodically to ensure hot entities always have cached features
    """
    
    def __init__(self, interval_seconds: int = 300):
        """
        initialize cache warmer
        
        args:
            interval_seconds: how often to run (default 5 minutes)
        """
        self.interval = interval_seconds
        self.running = False
    
    async def start(self):
        """start the background worker"""
        self.running = True
        print(f"🔄 cache warmer started (interval: {self.interval}s)")
        
        while self.running:
            try:
                await self.warm_active_caches()
            except Exception as e:
                print(f"⚠️  cache warming error: {e}")
            
            await asyncio.sleep(self.interval)
    
    def stop(self):
        """stop the background worker"""
        self.running = False
        print("🛑 cache warmer stopped")
    
    async def warm_active_caches(self):
        """
        pre-warm caches for recently active users and merchants
        only processes entities that had activity in last 10 minutes
        """
        
        start_time = datetime.now()
        
        # get active users (transacted in last 10 minutes)
        active_users = await self._get_active_users(minutes=10)
        
        # get active merchants
        active_merchants = await self._get_active_merchants(minutes=10)
        
        if not active_users and not active_merchants:
            print(f"⏸️  no active entities to warm (idle period)")
            return
        
        print(f"🔥 warming cache for {len(active_users)} users, {len(active_merchants)} merchants")
        
        # warm user caches
        warmed_users = 0
        for user_id in active_users:
            try:
                features = await compute_user_velocity_features(user_id, datetime.now())
                user_cache.set(f"user_velocity_{user_id}_{datetime.now().minute // 5}", features)
                warmed_users += 1
            except Exception as e:
                print(f"⚠️  failed to warm user {user_id}: {e}")
        
        # warm merchant caches
        warmed_merchants = 0
        for merchant_id in active_merchants:
            try:
                features = await compute_merchant_baseline(merchant_id)
                merchant_cache.set(f"merchant_baseline_{merchant_id}", features)
                warmed_merchants += 1
            except Exception as e:
                print(f"⚠️  failed to warm merchant {merchant_id}: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ warmed {warmed_users} users, {warmed_merchants} merchants in {elapsed:.2f}s")
    
    async def _get_active_users(self, minutes: int = 10) -> List[int]:
        """get users who transacted in last N minutes"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        async with db.async_session() as session:
            query = select(distinct(Transaction.user_id)).where(
                Transaction.timestamp >= cutoff_time
            )
            result = await session.execute(query)
            user_ids = [row[0] for row in result.fetchall()]
        
        return user_ids
    
    async def _get_active_merchants(self, minutes: int = 10) -> List[int]:
        """get merchants who had transactions in last N minutes"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        async with db.async_session() as session:
            query = select(distinct(Transaction.merchant_id)).where(
                Transaction.timestamp >= cutoff_time
            )
            result = await session.execute(query)
            merchant_ids = [row[0] for row in result.fetchall()]
        
        return merchant_ids

# global worker instance
cache_warmer = None

def get_cache_warmer() -> CacheWarmer:
    """get or create global cache warmer instance"""
    global cache_warmer
    if cache_warmer is None:
        cache_warmer = CacheWarmer(interval_seconds=300)  # 5 minutes
    return cache_warmer
