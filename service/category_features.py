"""
Category and merchant risk baseline features
High-ROI features that use historical fraud rates
"""

from sqlalchemy import select, func
from datetime import datetime, timedelta
from db.models import Transaction, Merchant, db

async def compute_category_risk_features(
    merchant_id: int,
    timestamp: datetime,
    lookback_days: int = 30
) -> dict:
    """
    Compute category-level fraud statistics
    
    Returns:
        category_fraud_rate_30d: % of transactions in this category that were fraud
        category_avg_amount: average transaction amount for this category
    """
    
    # Get merchant's category
    async with db.async_session() as session:
        merchant_result = await session.execute(
            select(Merchant.category).where(Merchant.merchant_id == merchant_id)
        )
        category = merchant_result.scalar()
        
        if not category:
            return {
                'category_fraud_rate_30d': 0.0,
                'category_avg_amount': 0.0
            }
        
        # Get all merchants in this category
        merchants_in_category = await session.execute(
            select(Merchant.merchant_id).where(Merchant.category == category)
        )
        merchant_ids = [m[0] for m in merchants_in_category.all()]
        
        if not merchant_ids:
            return {
                'category_fraud_rate_30d': 0.0,
                'category_avg_amount': 0.0
            }
        
        # Calculate fraud rate for category
        lookback_start = timestamp - timedelta(days=lookback_days)
        
        # Total transactions in category
        total_query = select(func.count(Transaction.transaction_id)).where(
            Transaction.merchant_id.in_(merchant_ids),
            Transaction.timestamp >= lookback_start,
            Transaction.timestamp < timestamp
        )
        total_count = (await session.execute(total_query)).scalar() or 0
        
        # Fraud transactions in category
        fraud_query = select(func.count(Transaction.transaction_id)).where(
            Transaction.merchant_id.in_(merchant_ids),
            Transaction.timestamp >= lookback_start,
            Transaction.timestamp < timestamp,
            Transaction.label == 1
        )
        fraud_count = (await session.execute(fraud_query)).scalar() or 0
        
        # Average amount in category
        avg_query = select(func.avg(Transaction.amount)).where(
            Transaction.merchant_id.in_(merchant_ids),
            Transaction.timestamp >= lookback_start,
            Transaction.timestamp < timestamp
        )
        avg_amount = (await session.execute(avg_query)).scalar() or 0.0
        
        fraud_rate = fraud_count / total_count if total_count > 0 else 0.0
        
        return {
            'category_fraud_rate_30d': float(fraud_rate),
            'category_avg_amount': float(avg_amount)
        }

async def compute_user_category_features(
    user_id: int,
    merchant_id: int,
    timestamp: datetime,
    lookback_days: int = 7
) -> dict:
    """
    Compute user's interaction with this merchant category
    
    Returns:
        user_category_count_7d: # of times user transacted in this category in last 7 days
        is_new_category_for_user: 1 if user never used this category before, else 0
    """
    
    async with db.async_session() as session:
        # Get merchant's category
        merchant_result = await session.execute(
            select(Merchant.category).where(Merchant.merchant_id == merchant_id)
        )
        category = merchant_result.scalar()
        
        if not category:
            return {
                'user_category_count_7d': 0.0,
                'is_new_category_for_user': 1.0
            }
        
        # Get all merchants in this category
        merchants_in_category = await session.execute(
            select(Merchant.merchant_id).where(Merchant.category == category)
        )
        merchant_ids = [m[0] for m in merchants_in_category.all()]
        
        # Count user's transactions in this category (last 7 days)
        lookback_start = timestamp - timedelta(days=lookback_days)
        count_query = select(func.count(Transaction.transaction_id)).where(
            Transaction.user_id == user_id,
            Transaction.merchant_id.in_(merchant_ids),
            Transaction.timestamp >= lookback_start,
            Transaction.timestamp < timestamp
        )
        category_count = (await session.execute(count_query)).scalar() or 0
        
        # Check if this is first time user used this category
        ever_used_query = select(func.count(Transaction.transaction_id)).where(
            Transaction.user_id == user_id,
            Transaction.merchant_id.in_(merchant_ids),
            Transaction.timestamp < timestamp
        )
        ever_count = (await session.execute(ever_used_query)).scalar() or 0
        
        return {
            'user_category_count_7d': float(category_count),
            'is_new_category_for_user': 1.0 if ever_count == 0 else 0.0
        }
