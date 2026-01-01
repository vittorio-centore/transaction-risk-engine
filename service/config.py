# configuration management using pydantic settings
# reads from .env file and provides type-safe config

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """app configuration loaded from environment variables"""
    
    # database urls
    database_url: str = "postgresql+asyncpg://frauduser:fraudpass@localhost:5432/frauddb"
    database_url_sync: str = "postgresql://frauduser:fraudpass@localhost:5432/frauddb"
    database_url_local: str = "postgresql+asyncpg://frauduser:fraudpass@localhost:5432/frauddb"  # for local scripts
    
    # model settings
    model_version: str = "v1"
    model_path: str = "models/fraud_model_v1.pt"
    
    # decision thresholds (what score = decline/review/approve)
    decline_threshold: float = 0.8  # score >= 0.8 → decline
    review_threshold: float = 0.5   # 0.5 <= score < 0.8 → review
    # score < 0.5 → approve
    
    # cache ttl (time-to-live in seconds)
    user_cache_ttl: int = 300     # 5 minutes for user features (aligns with bucketing)
    merchant_cache_ttl: int = 1800  # 30 minutes for merchant features (slower to change)
    
    # api settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    class Config:
        env_file = ".env"  # load from .env file
        case_sensitive = False  # DATABASE_URL and database_url both work
        extra = 'ignore'  # ignore extra fields in .env

# global settings instance
settings = Settings()
