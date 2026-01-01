# fraud prediction service  
# loads trained pytorch model and makes predictions

import torch
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from model.model import FraudMLP
from service.features import compute_all_features, get_feature_names
from service.config import settings

class FraudPredictor:
    """
    handles model loading and fraud prediction
    uses trained pytorch model to score transactions
    """
    
    def __init__(self, model_path: str = None):
        """
        initialize predictor and load model
        
        args:
            model_path: path to saved model checkpoint
        """
        if model_path is None:
            model_path = settings.model_path
        
        self.model_path = Path(model_path)
        self.scaler_path = self.model_path.parent / 'scaler.pkl'
        self.model = None
        self.scaler = None
        self.feature_names = get_feature_names()
        
        # load model at initialization
        self._load_model()
    
    def _load_model(self):
        """load trained model and scaler from checkpoint"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model not found at {self.model_path}. "
                f"train model first with: python model/train.py"
            )
        
        # load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # initialize model with correct architecture
        # Load model with correct 19-feature input (includes geo_missing flag)
        self.model = FraudMLP(input_dim=19)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # set to evaluation mode (no dropout)
        
        # load scaler
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            print(f"✅ loaded scaler from {self.scaler_path}")
        else:
            print(f"⚠️  scaler not found - predictions will be unreliable!")
            self.scaler = None
        
        print(f"✅ loaded model from {self.model_path}")
        print(f"   trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"   val_loss: {checkpoint.get('val_loss', 'unknown')}")
    
    async def predict(
        self,
        user_id: int,
        merchant_id: int,
        amount: float,
        timestamp: datetime = None,
        v_features: Dict[str, float] = None
    ) -> Tuple[float, str, Dict]:
        """
        predict fraud score for a transaction
        
        args:
            user_id: user making transaction
            merchant_id: merchant receiving payment
            amount: transaction amount
            timestamp: when transaction happened (default: now)
            v_features: optional v1-v28 features from kaggle dataset
            
        returns:
            (fraud_score, decision, metadata)
            - fraud_score: 0.0-1.0 probability of fraud
            - decision: "approve", "review", or "decline"
            - metadata: dict with features and explanations
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # compute feature vector
        feature_vector = await compute_all_features(
            user_id=user_id,
            merchant_id=merchant_id,
            amount=amount,
            timestamp=timestamp,
            v_features=v_features
        )
        
        # CRITICAL: scale features using saved scaler
        if self.scaler is not None:
            feature_vector_scaled = self.scaler.transform([feature_vector])[0]
        else:
            # fallback: no scaling (will give poor predictions)
            feature_vector_scaled = feature_vector
            print("⚠️  predicting without scaling - results may be unreliable")
        
        # run model inference
        with torch.no_grad():
            features_tensor = torch.FloatTensor(feature_vector_scaled).unsqueeze(0)
            fraud_score = self.model.predict_proba(features_tensor).item()
        
        # make decision based on thresholds
        if fraud_score >= settings.decline_threshold:
            decision = "decline"
        elif fraud_score >= settings.review_threshold:
            decision = "review"
        else:
            decision = "approve"
        
        # package metadata
        metadata = {
            'fraud_score': float(fraud_score),
            'decision': decision,
            'thresholds': {
                'decline': settings.decline_threshold,
                'review': settings.review_threshold
            },
            'feature_vector': feature_vector.tolist(),
            'feature_names': self.feature_names
        }
        
        return fraud_score, decision, metadata
    
    def get_model_info(self) -> Dict:
        """get information about loaded model"""
        return {
            'model_path': str(self.model_path),
            'model_version': settings.model_version,
            'input_features': len(self.feature_names),
            'feature_names': self.feature_names
        }

# global predictor instance
# loaded once at app startup for efficiency
predictor = None

def get_predictor() -> FraudPredictor:
    """
    get or create global predictor instance
    ensures model is loaded only once
    """
    global predictor
    if predictor is None:
        predictor = FraudPredictor()
    return predictor
