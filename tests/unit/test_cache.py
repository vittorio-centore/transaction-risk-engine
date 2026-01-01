# unit tests for cache implementation
# these verify ttl cache works correctly

import pytest
import time
from service.cache import TTLCache

def test_cache_basic_set_get():
    """test that we can set and get values"""
    cache = TTLCache(ttl_seconds=10)
    
    cache.set("key1", "value1")
    cache.set("key2", 42)
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == 42
    assert cache.get("nonexistent") is None

def test_cache_expiration():
    """test that items expire after ttl"""
    cache = TTLCache(ttl_seconds=1)  # 1 second ttl
    
    cache.set("key1", "value1")
    
    # immediately should work
    assert cache.get("key1") == "value1"
    
    # wait for expiration
    time.sleep(1.1)
    
    # should be expired now
    assert cache.get("key1") is None

def test_cache_overwrite():
    """test that setting same key overwrites"""
    cache = TTLCache(ttl_seconds=10)
    
    cache.set("key1", "old_value")
    cache.set("key1", "new_value")
    
    assert cache.get("key1") == "new_value"

def test_cache_clear():
    """test that clear removes all items"""
    cache = TTLCache(ttl_seconds=10)
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    
    assert cache.size() == 3
    
    cache.clear()
    
    assert cache.size() == 0
    assert cache.get("key1") is None
    assert cache.get("key2") is None

def test_cache_different_types():
    """test caching different data types"""
    cache = TTLCache(ttl_seconds=10)
    
    # string
    cache.set("str", "hello")
    assert cache.get("str") == "hello"
    
    # int
    cache.set("int", 123)
    assert cache.get("int") == 123
    
    # float
    cache.set("float", 45.67)
    assert cache.get("float") == 45.67
    
    # dict
    cache.set("dict", {"a": 1, "b": 2})
    assert cache.get("dict") == {"a": 1, "b": 2}
    
    # list
    cache.set("list", [1, 2, 3])
    assert cache.get("list") == [1, 2, 3]

def test_cache_hit_rate():
    """test hit rate calculation"""
    cache = TTLCache(ttl_seconds=10)
    
    # all misses
    assert cache.hit_rate(hits=0, misses=10) == 0.0
    
    # all hits
    assert cache.hit_rate(hits=10, misses=0) == 100.0
    
    # 50/50
    assert cache.hit_rate(hits=5, misses=5) == 50.0
    
    # 75% hit rate
    assert cache.hit_rate(hits=75, misses=25) == 75.0
