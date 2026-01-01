# feature engineering - compute fraud detection signals from raw transaction data
# these features are what the ml model actually uses to predict fraud

from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from db.models import Transaction, User, Merchant, db
from service.cache import user_cache, merchant_cache
import numpy as np
from typing import Dict, Optional

# feature names in order (must match model training)
FEATURE_NAMES = [
    'tx_count_10m',           # how many transactions in last 10 minutes
    'tx_count_1h',            # how many transactions in last 1 hour
    'tx_count_24h',           # how many transactions in last 24 hours
    'spend_1h',               # total spend in last hour
    'spend_24h',              # total spend in last 24 hours
    'avg_amount_1h',          # average transaction amount in last hour
    'unique_merchants_24h',   # how many different merchants in 24h
    'merchant_fraud_rate_30d', # what % of this merchant's transactions are fraud
    'amount',                 # current transaction amount
    'time_since_last_tx',     # seconds since user's last transaction
    # v1-v28 from kaggle dataset (pca features)
]

async def compute_user_velocity_features(
    user_id: int, 
    current_time: datetime,
    exclude_tx_id: Optional[int] = None
) -> Dict[str, float]:
    """
    compute how fast a user is transacting (velocity features)
    high velocity = many transactions in short time = fraud signal
    
    args:
        user_id: the user to check
        current_time: timestamp to compute features at
        exclude_tx_id: optional transaction id to exclude (when scoring current tx)
        
    returns:
        dict of velocity features
    """
    
    # create cache key with 5-minute bucketing
    # this means features computed at 1:02pm and 1:04pm use same cache
    bucket = current_time.minute // 5
    cache_key = f"user_velocity_{user_id}_{current_time.strftime('%Y%m%d%H')}_{bucket}"
    
    # try cache first
    cached = user_cache.get(cache_key)
    if cached:
        return cached  # cache hit! no db query needed
    
    # cache miss - compute from database
    async with db.async_session() as session:
        # base query filters
        base_filters = [
            Transaction.user_id == user_id,
        ]
        if exclude_tx_id:
            base_filters.append(Transaction.transaction_id != exclude_tx_id)
        
        # --- last 10 minutes ---
        query_10m = select(
            func.count(Transaction.transaction_id).label('count'),
            func.coalesce(func.sum(Transaction.amount), 0).label('spend')
        ).where(
            and_(
                *base_filters,
                Transaction.timestamp >= current_time - timedelta(minutes=10)
            )
        )
        result_10m = (await session.execute(query_10m)).one()
        tx_count_10m = result_10m.count
        
        # --- last 1 hour ---
        query_1h = select(
            func.count(Transaction.transaction_id).label('count'),
            func.coalesce(func.sum(Transaction.amount), 0).label('spend'),
            func.coalesce(func.avg(Transaction.amount), 0).label('avg')
        ).where(
            and_(
                *base_filters,
                Transaction.timestamp >= current_time - timedelta(hours=1)
            )
        )
        result_1h = (await session.execute(query_1h)).one()
        tx_count_1h = result_1h.count
        spend_1h = float(result_1h.spend)
        avg_amount_1h = float(result_1h.avg)
        
        # --- last 24 hours ---
        query_24h = select(
            func.count(Transaction.transaction_id).label('count'),
            func.coalesce(func.sum(Transaction.amount), 0).label('spend'),
            func.count(func.distinct(Transaction.merchant_id)).label('unique_merchants')
        ).where(
            and_(
                *base_filters,
                Transaction.timestamp >= current_time - timedelta(hours=24)
            )
        )
        result_24h = (await session.execute(query_24h)).one()
        tx_count_24h = result_24h.count
        spend_24h = float(result_24h.spend)
        unique_merchants_24h = result_24h.unique_merchants
        
        # --- time since last transaction ---
        query_last_tx = select(
            func.max(Transaction.timestamp)
        ).where(
            and_(
                *base_filters,
                Transaction.timestamp < current_time
            )
        )
        last_tx_time = (await session.execute(query_last_tx)).scalar()
        
        if last_tx_time:
            time_since_last_tx = (current_time - last_tx_time).total_seconds()
        else:
            # no previous transaction (new user)
            # cap at 7 days to prevent scaler issues
            time_since_last_tx = 7 * 24 * 3600  # 604800 seconds
    
    # build features dict
    features = {
        'tx_count_10m': int(tx_count_10m),
        'tx_count_1h': int(tx_count_1h),
        'tx_count_24h': int(tx_count_24h),
        'spend_1h': spend_1h,
        'spend_24h': spend_24h,
        'avg_amount_1h': avg_amount_1h,
        'unique_merchants_24h': int(unique_merchants_24h),
        'time_since_last_tx': time_since_last_tx,
    }
    
    # cache it for next time
    user_cache.set(cache_key, features)
    
    return features

async def compute_merchant_baseline(merchant_id: int, timestamp: datetime) -> Dict[str, float]:
    """
    compute merchant fraud baseline - what % of transactions at this merchant are fraud
    high fraud rate = risky merchant
    
    args:
        merchant_id: merchant to check
        
    returns:
        dict with merchant_fraud_rate_30d
    """
    
    # NO CACHE - must compute per-transaction for temporal correctness
    lookback_start = timestamp - timedelta(days=30)
    
    async with db.async_session() as session:
        query = select(
            func.count(Transaction.transaction_id).label('total'),
            func.coalesce(func.sum(Transaction.label), 0).label('fraud_count')
        ).where(
            and_(
                Transaction.merchant_id == merchant_id,
                Transaction.timestamp >= lookback_start,
                Transaction.timestamp < timestamp  # STRICT: only past data
            )
        )
        result = (await session.execute(query)).one()
        
        total = result.total
        fraud_count = result.fraud_count
        
        # calculate fraud rate (avoid division by zero)
        if total > 0:
            fraud_rate = float(fraud_count) / float(total)
        else:
            fraud_rate = 0.0  # no history = assume safe
    
    features = {
        'merchant_fraud_rate_30d': fraud_rate
    }
    
    return features

async def compute_all_features(
    user_id: int,
    merchant_id: int,
    amount: float,
    timestamp: datetime,
    v_features: Optional[Dict[str, float]] = None,
    exclude_tx_id: Optional[int] = None
) -> np.ndarray:
    """
    compute complete feature vector for a transaction
    this is what gets fed into the ml model
    
    args:
        user_id: user making transaction
        merchant_id: merchant receiving payment
        amount: transaction amount
        timestamp: when transaction happened
        v_features: optional dict with v1-v28 values (from kaggle dataset)
        exclude_tx_id: optional - exclude this transaction from velocity calcs
        
    returns:
        numpy array of features in correct order for model
    """
    
    # get user velocity features
    user_features = await compute_user_velocity_features(
        user_id, 
        timestamp,
        exclude_tx_id
    )
    
    # Get merchant risk baseline
    merchant_features = await compute_merchant_baseline(merchant_id, timestamp)
    
    # NEW: Category-based features (high ROI!)
    from service.category_features import compute_category_risk_features, compute_user_category_features
    
    category_features = await compute_category_risk_features(
        merchant_id=merchant_id,
        timestamp=timestamp
    )
    
    user_category_features = await compute_user_category_features(
        user_id=user_id,
        merchant_id=merchant_id,
        timestamp=timestamp
    )
    
    # NEW: Geographic features (computed SERVER-SIDE from DB)
    from service.geo_features import compute_geo_features
    
    geo_features = await compute_geo_features(
        user_id=user_id,
        merchant_id=merchant_id,
        current_time=timestamp,
        exclude_tx_id=exclude_tx_id
    )
    
    # Build feature vector (19 features total now with geo_missing)
    feature_vector = [
        # Velocity features (7)
        user_features['tx_count_10m'],
        user_features['tx_count_1h'],
        user_features['tx_count_24h'],
        user_features['spend_1h'],
        user_features['spend_24h'],
        user_features['avg_amount_1h'],
        user_features['unique_merchants_24h'],
        
        # Merchant risk (1)
        merchant_features['merchant_fraud_rate_30d'],
        
        # Amount & time (2)
        float(amount),
        user_features['time_since_last_tx'],
        
        # Category features (4)
        category_features['category_fraud_rate_30d'],
        category_features['category_avg_amount'],
        user_category_features['user_category_count_7d'],
        user_category_features['is_new_category_for_user'],
        
        # Geographic features (5 - includes geo_missing!)
        geo_features['distance_from_last_tx'],
        geo_features['implied_speed_mph'],
        geo_features['distance_from_home'],
        geo_features['is_foreign_country'],
        geo_features['geo_missing'],  # EXPLICIT missing flag
    ]
    
    # TOTAL: 19 features (all real, server-side computed, production-ready!)
    return np.array(feature_vector, dtype=np.float32)

def get_feature_names() -> list[str]:
    """
    Get ordered list of feature names (19 total)
    All real, server-side computed, production-ready features!
    """
    return [
        # Velocity (7)
        'tx_count_10m',
        'tx_count_1h',
        'tx_count_24h',
        'spend_1h',
        'spend_24h',
        'avg_amount_1h',
        'unique_merchants_24h',
        # Merchant (1)
        'merchant_fraud_rate_30d',
        # Amount & time (2)
        'amount',
        'time_since_last_tx',
        # Category (4)
        'category_fraud_rate_30d',
        'category_avg_amount',
        'user_category_count_7d',
        'is_new_category_for_user',
        # Geographic (5 - includes missing flag!)
        'distance_from_last_tx',
        'implied_speed_mph',
        'distance_from_home',
        'is_foreign_country',
        'geo_missing',  # Explicit flag for robustness
    ]
