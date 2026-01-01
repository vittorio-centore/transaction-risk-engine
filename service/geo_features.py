"""
Geographic distance features for fraud detection
Uses transaction lat/long to detect impossible travel and location anomalies
"""

from sqlalchemy import select, func
from datetime import datetime, timedelta
from db.models import Transaction, Merchant, db
import math

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points on Earth in miles
    Uses Haversine formula
    """
    # Convert to float (database returns Decimal)
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    
    R = 3959  # Earth radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

async def compute_geo_features(
    user_id: int,
    merchant_id: int,
    current_time: datetime,
    exclude_tx_id: int = None
) -> dict:
    """
    Compute geographic features SERVER-SIDE (production best practice)
    
    Fetches locations from DB instead of requiring client to send them.
    This prevents training/serving mismatch and matches how real risk engines work.
    
    Returns:
        distance_from_last_tx: miles from previous transaction
        implied_speed_mph: speed needed to travel from last tx
        distance_from_home: miles from user's typical location
        is_foreign_country: 1 if very far from typical locations
        geo_missing: 1 if we can't compute geo (first transaction)
    """
    
    async with db.async_session() as session:
        # Fetch merchant location from merchants table
        merchant_query = select(Merchant).where(Merchant.merchant_id == merchant_id)
        merchant_result = await session.execute(merchant_query)
        merchant = merchant_result.scalar()
        
        if not merchant or not merchant.name:
            # Merchant not found - return missing geo
            return {
                'distance_from_last_tx': 0.0,
                'implied_speed_mph': 0.0,
                'distance_from_home': 0.0,
                'is_foreign_country': 0.0,
                'geo_missing': 1.0
            }
        
        # Get merchant lat/long from merchant's transaction history (Sparkov stores it per-tx)
        # This is a workaround since merchants table doesn't have lat/long directly
        merchant_tx_query = select(Transaction).where(
            Transaction.merchant_id == merchant_id
        ).limit(1)
        merchant_tx_result = await session.execute(merchant_tx_query)
        merchant_tx = merchant_tx_result.scalar()
        
        if not merchant_tx:
            return {
                'distance_from_last_tx': 0.0,
                'implied_speed_mph': 0.0,
                'distance_from_home': 0.0,
                'is_foreign_country': 0.0,
                'geo_missing': 1.0
            }
        
        current_lat = float(merchant_tx.merch_lat)
        current_long = float(merchant_tx.merch_long)
        
        # Get user's most recent transaction (for last location)
        last_tx_query = select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.timestamp < current_time
        )
        
        if exclude_tx_id:
            last_tx_query = last_tx_query.where(Transaction.transaction_id != exclude_tx_id)
        
        last_tx_query = last_tx_query.order_by(Transaction.timestamp.desc()).limit(1)
        last_tx_result = await session.execute(last_tx_query)
        last_tx = last_tx_result.scalar()
        
        if not last_tx:
            # First transaction for user - can't compute geo
            return {
                'distance_from_last_tx': 0.0,
                'implied_speed_mph': 0.0,
                'distance_from_home': 0.0,
                'is_foreign_country': 0.0,
                'geo_missing': 1.0  # EXPLICIT missing flag
            }
        
        # Calculate distance from last transaction
        distance_from_last = haversine_distance(
            last_tx.lat, last_tx.long,
            current_lat, current_long
        )
        
        # Calculate implied travel speed
        time_diff = (current_time - last_tx.timestamp).total_seconds() / 3600  # hours
        implied_speed = distance_from_last / time_diff if time_diff > 0 else 0.0
        
        # Find user's "home" location (centroid of last 30 days)
        home_query = select(
            func.avg(Transaction.lat).label('avg_lat'),
            func.avg(Transaction.long).label('avg_long')
        ).where(
            Transaction.user_id == user_id,
            Transaction.timestamp >= current_time - timedelta(days=30),
            Transaction.timestamp < current_time
        )
        
        if exclude_tx_id:
            home_query = home_query.where(Transaction.transaction_id != exclude_tx_id)
        
        home_result = await session.execute(home_query)
        home_location = home_result.first()
        
        if home_location and home_location.avg_lat:
            distance_from_home = haversine_distance(
                home_location.avg_lat, home_location.avg_long,
                current_lat, current_long
            )
        else:
            distance_from_home = 0.0
        
        # Flag if very far from typical locations
        is_foreign = 1.0 if distance_from_home > 500 else 0.0
        
        return {
            'distance_from_last_tx': float(distance_from_last),
            'implied_speed_mph': float(implied_speed),
            'distance_from_home': float(distance_from_home),
            'is_foreign_country': float(is_foreign),
            'geo_missing': 0.0  # We successfully computed geo
        }
