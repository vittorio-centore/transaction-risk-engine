# unit tests for feature computation
# these test that features are calculated correctly

import pytest
from datetime import datetime, timedelta
from service.features import (
    compute_user_velocity_features,
    compute_merchant_baseline,
    compute_all_features,
    get_feature_names
)
from service.cache import user_cache, merchant_cache
from db.models import db

# mark all tests in this file as async
pytestmark = pytest.mark.asyncio

@pytest.fixture(autouse=True)
async def clear_caches():
    """clear caches before each test"""
    user_cache.clear()
    merchant_cache.clear()
    yield
    user_cache.clear()
    merchant_cache.clear()

async def test_user_velocity_features():
    """test that user velocity features compute correctly with real data"""
    
    # use a real user from our database (user_id=1 definitely exists)
    user_id = 1
    current_time = datetime.now()
    
    # compute features
    features = await compute_user_velocity_features(user_id, current_time)
    
    # check all expected keys exist
    assert 'tx_count_10m' in features
    assert 'tx_count_1h' in features
    assert 'tx_count_24h' in features
    assert 'spend_1h' in features
    assert 'spend_24h' in features
    assert 'avg_amount_1h' in features
    assert 'unique_merchants_24h' in features
    assert 'time_since_last_tx' in features
    
    # check types
    assert isinstance(features['tx_count_10m'], int)
    assert isinstance(features['spend_1h'], float)
    
    # check logical constraints
    # 10min count should be <= 1h count should be <= 24h count
    assert features['tx_count_10m'] <= features['tx_count_1h']
    assert features['tx_count_1h'] <= features['tx_count_24h']
    
    # 1h spend should be <= 24h spend
    assert features['spend_1h'] <= features['spend_24h']
    
    print(f"✅ user velocity features: {features}")

async def test_user_velocity_caching():
    """test that repeated calls use cache"""
    
    user_id = 1
    current_time = datetime.now()
    
    # first call - cache miss
    features1 = await compute_user_velocity_features(user_id, current_time)
    
    # second call same time - should be cache hit
    features2 = await compute_user_velocity_features(user_id, current_time)
    
    # should be identical
    assert features1 == features2
    
    # verify cache is being used
    assert user_cache.size() > 0
    
    print(f"✅ cache working - size: {user_cache.size()}")

async def test_merchant_baseline():
    """test merchant fraud rate calculation"""
    
    # use a real merchant from database
    merchant_id = 1
    
    # compute baseline
    features = await compute_merchant_baseline(merchant_id)
    
    # check key exists
    assert 'merchant_fraud_rate_30d' in features
    
    # fraud rate should be between 0 and 1
    fraud_rate = features['merchant_fraud_rate_30d']
    assert 0.0 <= fraud_rate <= 1.0
    
    # check it's a float
    assert isinstance(fraud_rate, float)
    
    print(f"✅ merchant {merchant_id} fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")

async def test_merchant_caching():
    """test that merchant features are cached"""
    
    merchant_id = 1
    
    # clear cache first
    merchant_cache.clear()
    assert merchant_cache.size() == 0
    
    # first call
    features1 = await compute_merchant_baseline(merchant_id)
    
    # cache should have 1 item now
    assert merchant_cache.size() == 1
    
    # second call
    features2 = await compute_merchant_baseline(merchant_id)
    
    # should be identical
    assert features1 == features2
    
    print("✅ merchant cache working")

async def test_compute_all_features():
    """test complete feature vector assembly"""
    
    # test with a real user/merchant
    user_id = 1
    merchant_id = 1
    amount = 100.50
    timestamp = datetime.now()
    
    # compute all features
    feature_vector = await compute_all_features(
        user_id=user_id,
        merchant_id=merchant_id,
        amount=amount,
        timestamp=timestamp
    )
    
    # should be numpy array
    import numpy as np
    assert isinstance(feature_vector, np.ndarray)
    
    # should have correct length (10 base features + 28 v features)
    assert len(feature_vector) == 38
    
    # all should be floats
    assert feature_vector.dtype == np.float32
    
    # amount feature should match input
    assert feature_vector[8] == amount
    
    print(f"✅ feature vector shape: {feature_vector.shape}")
    print(f"   first 10 features: {feature_vector[:10]}")

def test_feature_names():
    """test that feature names are returned correctly"""
    
    names = get_feature_names()
    
    # should have 38 total (10 + 28)
    assert len(names) == 38
    
    # check some expected names
    assert 'tx_count_10m' in names
    assert 'merchant_fraud_rate_30d' in names
    assert 'amount' in names
    assert 'v1' in names
    assert 'v28' in names
    
    print(f"✅ feature names: {names[:10]}...")
