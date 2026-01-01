# integration tests for fraud detection api
# tests end-to-end workflow: api → features → model → response

import pytest
import asyncio
from httpx import AsyncClient
from datetime import datetime

# we'll need to import the app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from service.main import app

pytestmark = pytest.mark.asyncio

class TestFraudAPI:
    """integration tests for fraud detection api"""
    
    @pytest.fixture
    async def client(self):
        """create test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    async def test_health_endpoint(self, client):
        """test health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
        assert 'model_loaded' in data
        assert 'database_connected' in data
    
    async def test_root_endpoint(self, client):
        """test root endpoint returns api info"""
        response = await client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert 'service' in data
        assert 'version' in data
        assert 'endpoints' in data
    
    async def test_score_endpoint_valid_request(self, client):
        """test scoring a valid transaction"""
        payload = {
            "user_id": 1,
            "merchant_id": 1,
            "amount": 100.00
        }
        
        response = await client.post("/score", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'fraud_score' in data
        assert 'decision' in data
        assert 'reason' in data
        assert 'top_features' in data
        
        # validate score is between 0 and 1
        assert 0.0 <= data['fraud_score'] <= 1.0
        
        # validate decision is one of the expected values
        assert data['decision'] in ['approve', 'review', 'decline']
        
        # validate top features is a list
        assert isinstance(data['top_features'], list)
        assert len(data['top_features']) > 0
    
    async def test_score_endpoint_high_velocity(self, client):
        """test multiple transactions from same user (velocity detection)"""
        payload = {
            "user_id": 999,
            "merchant_id": 1,
            "amount": 100.00
        }
        
        # make several requests quickly
        responses = []
        for i in range(3):
            response = await client.post("/score", json=payload)
            assert response.status_code == 200
            responses.append(response.json())
            await asyncio.sleep(0.1)
        
        # verify all requests succeeded
        assert len(responses) == 3
        
        # scores should exist for all
        for data in responses:
            assert 'fraud_score' in data
    
    async def test_score_endpoint_missing_fields(self, client):
        """test request with missing required fields"""
        payload = {
            "user_id": 1
            # missing merchant_id and amount
        }
        
        response = await client.post("/score", json=payload)
        assert response.status_code == 422  # validation error
    
    async def test_score_endpoint_invalid_amount(self, client):
        """test request with invalid amount"""
        payload = {
            "user_id": 1,
            "merchant_id": 1,
            "amount": -100.00  # negative amount
        }
        
        response = await client.post("/score", json=payload)
        assert response.status_code == 422  # validation error
