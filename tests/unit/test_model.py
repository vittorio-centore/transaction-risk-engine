# unit tests for model inference and prediction
# tests pytorch model loading and scoring

import pytest
import torch
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.model import FraudMLP

class TestModelInference:
    """unit tests for fraud detection model"""
    
    def test_model_initialization(self):
        """test model can be initialized with correct architecture"""
        model = FraudMLP(input_dim=38)
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward_pass(self):
        """test model can perform forward pass"""
        model = FraudMLP(input_dim=38)
        model.eval()
        
        # create dummy input
        x = torch.randn(1, 38)
        
        # forward pass
        with torch.no_grad():
            output = model(x)
        
        assert output is not None
        assert output.shape == torch.Size([])  # single scalar output
    
    def test_model_output_range(self):
        """test model output is in valid range"""
        model = FraudMLP(input_dim=38)
        model.eval()
        
        # create dummy input
        x = torch.randn(10, 38)
        
        # get predictions
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
        
        # probabilities should be between 0 and 1
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_model_predict_proba(self):
        """test model.predict_proba method"""
        model = FraudMLP(input_dim=38)
        
        # create dummy input
        x = torch.randn(5, 38)
        
        # get probabilities
        probs = model.predict_proba(x)
        
        assert probs is not None
        assert len(probs) == 5
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_model_batch_inference(self):
        """test model can handle batch inference"""
        model = FraudMLP(input_dim=38)
        
        batch_sizes = [1, 10, 100]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 38)
            probs = model.predict_proba(x)
            assert len(probs) == batch_size
    
    def test_model_parameter_count(self):
        """test model has reasonable number of parameters"""
        model = FraudMLP(input_dim=38)
        
        param_count = model.count_parameters()
        
        # should have parameters
        assert param_count > 0
        
        # should be reasonable size (not too small or huge)
        assert param_count < 1_000_000  # less than 1M params
    
    def test_model_consistency(self):
        """test model gives consistent results for same input"""
        model = FraudMLP(input_dim=38)
        model.eval()
        
        x = torch.randn(1, 38)
        
        # run twice with no_grad
        with torch.no_grad():
            output1 = model.predict_proba(x)
            output2 = model.predict_proba(x)
        
        # should be identical
        assert torch.allclose(output1, output2)
