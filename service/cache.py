# lazy ttl cache - stores expensive computations temporarily
# when you request data: if cached (and not expired) → instant return
#                        if not cached or expired → compute + cache + return

from datetime import datetime, timedelta
from typing import Optional, Any, Dict
import threading

class TTLCache:
    """
    time-to-live cache - items expire after a set time
    this speeds up repeated queries for the same user/merchant features
    """
    
    def __init__(self, ttl_seconds: int = 600):
        """
        initialize cache
        
        args:
            ttl_seconds: how long items stay valid (default 10 minutes)
        """
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = threading.Lock()  # thread-safe for concurrent access
    
    def get(self, key: str) -> Optional[Any]:
        """
        get value from cache if it exists and hasn't expired
        
        args:
            key: cache key (e.g., "user_velocity_123_2025-12-30-01")
            
        returns:
            cached value if found and fresh, None otherwise
        """
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                
                # check if still fresh
                if datetime.now() < expiry:
                    return value  # cache hit! 🎯
                else:
                    # expired - delete it
                    del self._cache[key]
        
        return None  # cache miss 😢
    
    def set(self, key: str, value: Any):
        """
        store value in cache with expiration time
        
        args:
            key: cache key
            value: data to cache
        """
        with self._lock:
            expiry = datetime.now() + self._ttl
            self._cache[key] = (value, expiry)
    
    def clear(self):
        """clear entire cache (useful for testing)"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """get number of items in cache"""
        with self._lock:
            return len(self._cache)
    
    def hit_rate(self, hits: int, misses: int) -> float:
        """
        calculate cache hit rate (for monitoring)
        
        args:
            hits: number of cache hits
            misses: number of cache misses
            
        returns:
            hit rate as percentage (0-100)
        """
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100

# global cache instances
# these are used across the app for different types of features
# ttl values come from config so they can be changed without code changes
from service.config import settings

user_cache = TTLCache(ttl_seconds=settings.user_cache_ttl)      # 5 min for user features
merchant_cache = TTLCache(ttl_seconds=settings.merchant_cache_ttl)  # 30 min for merchant features
