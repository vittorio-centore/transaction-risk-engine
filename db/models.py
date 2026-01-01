# async sqlalchemy models for postgres
# these map our database tables to python objects

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, DECIMAL, TIMESTAMP, ForeignKey, func, Date
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

load_dotenv()

Base = declarative_base()

# user model - represents a user account
class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)  # surrogate key
    cc_num_hash = Column(String(64), unique=True, nullable=False)  # hashed card number
    created_at = Column(TIMESTAMP, default=datetime.now)
    home_country = Column(String(3))
    account_age_days = Column(Integer)
    
    # relationship to transactions (one user -> many transactions)
    transactions = relationship('Transaction', back_populates='user')

# merchant model - represents a merchant/store
class Merchant(Base):
    __tablename__ = 'merchants'
    
    merchant_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False)  # merchant name
    category = Column(String(50))
    country = Column(String(3))
    created_at = Column(TIMESTAMP, default=datetime.now)
    
    # relationship to transactions (one merchant -> many transactions)
    transactions = relationship('Transaction', back_populates='merchant')

# transaction model - main table with all transaction data
class Transaction(Base):
    __tablename__ = 'transactions'
    
    transaction_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    merchant_id = Column(Integer, ForeignKey('merchants.merchant_id'))
    amount = Column(DECIMAL(12, 2), nullable=False)
    currency = Column(String(3), default='USD')
    country = Column(String(3))
    timestamp = Column(TIMESTAMP, default=datetime.now, nullable=False)
    label = Column(Integer, nullable=False)  # 0 or 1 (is_fraud)
    
    # Sparkov dataset features (real, interpretable features)
    category = Column(String(50))  # merchant category (e.g., gas_transport, grocery_pos)
    first_name = Column(String(50))  # cardholder first name
    last_name = Column(String(50))  # cardholder last name
    gender = Column(String(1))  # M/F
    street = Column(String(200))  # billing street address
    city = Column(String(100))  # billing city
    state = Column(String(2))  # billing state
    zip_code = Column(String(10))  # billing zip
    lat = Column(DECIMAL(10, 7))  # billing latitude
    long = Column(DECIMAL(10, 7))  # billing longitude
    city_pop = Column(Integer)  # city population
    job = Column(String(100))  # cardholder job
    dob = Column(Date)  # date of birth
    merch_lat = Column(DECIMAL(10, 7))  # merchant latitude
    merch_long = Column(DECIMAL(10, 7))  # merchant longitude
    
    # relationships
    user = relationship('User', back_populates='transactions')
    merchant = relationship('Merchant', back_populates='transactions')
    risk_score = relationship('RiskScore', back_populates='transaction', uselist=False)

# risk score model - audit log of predictions
class RiskScore(Base):
    __tablename__ = 'risk_scores'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey('transactions.transaction_id'))
    model_version = Column(String(50))
    risk_score = Column(DECIMAL(5, 4))  # probability 0-1
    decision = Column(String(10))  # approve/review/decline
    top_features = Column(JSONB)  # json array of feature contributions
    created_at = Column(TIMESTAMP, default=datetime.now)
    
    # relationship back to transaction
    transaction = relationship('Transaction', back_populates='risk_score')

# database connection helpers
class Database:
    """manages async database connections"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://frauduser:fraudpass@localhost:5432/frauddb')
        # create async engine with connection pooling
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # set to True to see all sql queries (useful for debugging)
            pool_size=10,  # max 10 concurrent connections
            max_overflow=20  # can create up to 20 extra connections if needed
        )
        # session factory for creating async sessions
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """get a new async database session"""
        async with self.async_session() as session:
            yield session
    
    async def create_tables(self):
        """create all tables (usually done via migration, but useful for testing)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """close database connections"""
        await self.engine.dispose()

# global database instance
db = Database()
