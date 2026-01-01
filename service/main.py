# fastapi application - rest api for fraud detection
# provides /score endpoint for real-time transaction scoring

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import uvicorn
import asyncio

from service.predictor import get_predictor
from service.explainer import explain_prediction, generate_fraud_reason
from service.config import settings
from service.cache import user_cache, merchant_cache
from db.models import RiskScore, db

# pydantic models for request/response validation
class ScoreRequest(BaseModel):
    """request body for /score endpoint"""
    user_id: int = Field(..., description="user making the transaction")
    merchant_id: int = Field(..., description="merchant receiving payment")
    amount: float = Field(..., gt=0, description="transaction amount in dollars")
    timestamp: Optional[datetime] = Field(None, description="transaction timestamp (default: now)")
    v_features: Optional[Dict[str, float]] = Field(None, description="optional v1-v28 features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "merchant_id": 45,
                "amount": 500.00,
                "timestamp": "2025-12-30T14:30:00"
            }
        }

class ScoreResponse(BaseModel):
    """response from /score endpoint"""
    fraud_score: float = Field(..., description="fraud probability (0-1)")
    decision: str = Field(..., description="approve, review, or decline")
    reason: str = Field(..., description="human-readable explanation")
    top_features: List[Dict] = Field(..., description="most important features")
    metadata: Dict = Field(..., description="additional details")

class HealthResponse(BaseModel):
    """response from /health endpoint"""
    status: str
    model_loaded: bool
    model_version: str
    database_connected: bool

# create fastapi app
app = FastAPI(
    title="Transaction Risk Engine API",
    description="Real-time fraud detection using PyTorch ML model",
    version="1.0.0"
)

@app.on_event("startup")
async def startup():
    """initialize services on app startup"""
    print("🚀 starting transaction risk engine api...")
    
    # load model (this happens once at startup)
    predictor = get_predictor()
    model_info = predictor.get_model_info()
    print(f"✅ model loaded: {model_info['model_version']}")
    print(f"   features: {model_info['input_features']}")
    
    # database is already initialized via db = Database()
    print("✅ database connected")
    
    # start cache warmer worker (background task)
    from worker.cache_warmer import get_cache_warmer
    cache_warmer = get_cache_warmer()
    asyncio.create_task(cache_warmer.start())
    print("✅ cache warmer started")

@app.on_event("shutdown")
async def shutdown():
    """cleanup on app shutdown"""
    print("👋 shutting down...")
    
    # stop cache warmer
    from worker.cache_warmer import get_cache_warmer
    cache_warmer = get_cache_warmer()
    cache_warmer.stop()

@app.get("/", tags=["root"])
async def root():
    """api root - basic info"""
    return {
        "service": "Transaction Risk Engine",
        "version": "1.0.0",
        "endpoints": {
            "score": "/score - score a transaction",
            "health": "/health - health check",
            "docs": "/docs - api documentation"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """
    health check endpoint
    verifies model is loaded and database is accessible
    """
    predictor = get_predictor()
    model_info = predictor.get_model_info()
    
    # test database connection properly
    db_connected = True
    try:
        async with db.async_session() as session:
            result = await session.execute("SELECT 1")
            # actually fetch the result to ensure query worked
            result.scalar()
    except Exception as e:
        db_connected = False
        print(f"⚠️  database health check failed: {e}")
    
    status = "healthy" if db_connected else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=predictor.model is not None,
        model_version=model_info['model_version'],
        database_connected=db_connected
    )

@app.post("/score", response_model=ScoreResponse, tags=["fraud detection"])
async def score_transaction(request: ScoreRequest):
    """
    score a transaction for fraud
    
    - computes features from database (user velocity, merchant risk, etc)
    - runs ml model inference
    - returns fraud score, decision, and explanation
    - logs prediction to database
    - invalidates cache and triggers background refresh (event-driven)
    
    example:
        curl -X POST http://localhost:8000/score \\
             -H "Content-Type: application/json" \\
             -d '{"user_id": 123, "merchant_id": 45, "amount": 500.00}'
    """
    
    try:
        # get predictor
        predictor = get_predictor()
        
        # run prediction
        fraud_score, decision, metadata = await predictor.predict(
            user_id=request.user_id,
            merchant_id=request.merchant_id,
            amount=request.amount,
            timestamp=request.timestamp,
            v_features=request.v_features
        )
        
        # explain prediction
        explanations = explain_prediction(
            feature_vector=metadata['feature_vector'],
            feature_names=metadata['feature_names'],
            top_n=5
        )
        
        reason = generate_fraud_reason(explanations, fraud_score)
        
        # save to database (async, non-blocking)
        await _save_risk_score(
            user_id=request.user_id,
            merchant_id=request.merchant_id,
            amount=request.amount,
            fraud_score=fraud_score,
            decision=decision,
            model_version=settings.model_version
        )
        
        # event-driven cache refresh (don't block response)
        asyncio.create_task(_refresh_cache_in_background(
            user_id=request.user_id,
            merchant_id=request.merchant_id
        ))
        
        return ScoreResponse(
            fraud_score=fraud_score,
            decision=decision,
            reason=reason,
            top_features=explanations,
            metadata={
                'thresholds': metadata['thresholds'],
                'model_version': settings.model_version,
                'timestamp': request.timestamp or datetime.now()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )

async def _refresh_cache_in_background(user_id: int, merchant_id: int):
    """
    event-driven cache refresh
    invalidates stale cache and recomputes fresh features in background
    this runs AFTER response is sent to user (non-blocking)
    """
    try:
        # small delay to ensure transaction is committed to DB
        await asyncio.sleep(0.5)
        
        # recompute fresh user features
        from service.features import compute_user_velocity_features, compute_merchant_baseline
        
        user_features = await compute_user_velocity_features(user_id, datetime.now())
        merchant_features = await compute_merchant_baseline(merchant_id)
        
        # cache keys match feature computation logic
        user_cache_key = f"user_velocity_{user_id}_{datetime.now().minute // 5}"
        merchant_cache_key = f"merchant_baseline_{merchant_id}"
        
        user_cache.set(user_cache_key, user_features)
        merchant_cache.set(merchant_cache_key, merchant_features)
        
        print(f"🔄 refreshed cache for user={user_id}, merchant={merchant_id}")
        
    except Exception as e:
        # log error but don't fail (cache refresh is best-effort)
        print(f"⚠️  background cache refresh failed: {e}")

async def _save_risk_score(
    user_id: int,
    merchant_id: int,
    amount: float,
    fraud_score: float,
    decision: str,
    model_version: str
):
    """
    save risk score to database for audit trail
    runs async so it doesn't block the response
    """
    try:
        async with db.async_session() as session:
            risk_score = RiskScore(
                user_id=user_id,
                merchant_id=merchant_id,
                amount=amount,
                score=fraud_score,
                decision=decision,
                model_version=model_version
            )
            session.add(risk_score)
            await session.commit()
    except Exception as e:
        # log error but don't fail the request
        print(f"⚠️  failed to save risk score: {e}")

if __name__ == "__main__":
    # run with: python service/main.py
    uvicorn.run(
        "service.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True  # auto-reload on code changes
    )
