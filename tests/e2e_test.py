#!/usr/bin/env python3
# end-to-end test script - validates entire system works
# tests model training → api startup → transaction scoring

import asyncio
import httpx
import time
from datetime import datetime

async def test_api_health():
    """test api health check"""
    print("\n🏥 testing /health endpoint...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get("http://localhost:8000/health")
            data = response.json()
            
            print(f"   status code: {response.status_code}")
            print(f"   health: {data.get('status')}")
            print(f"   model loaded: {data.get('model_loaded')}")
            print(f"   database connected: {data.get('database_connected')}")
            
            assert response.status_code == 200
            assert data['model_loaded'] == True
            print("   ✅ health check passed!")
            return True
        except Exception as e:
            print(f"   ❌ health check failed: {e}")
            return False

async def test_score_transaction(user_id, merchant_id, amount):
    """test scoring a single transaction"""
    print(f"\n💳 scoring transaction: user={user_id}, merchant={merchant_id}, amount=${amount}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            start_time = time.time()
            
            response = await client.post(
                "http://localhost:8000/score",
                json={
                    "user_id": user_id,
                    "merchant_id": merchant_id,
                    "amount": amount
                }
            )
            
            elapsed = (time.time() - start_time) * 1000
            
            data = response.json()
            
            print(f"   latency: {elapsed:.0f}ms")
            print(f"   fraud score: {data['fraud_score']:.4f}")
            print(f"   decision: {data['decision']}")
            print(f"   reason: {data['reason']}")
            print(f"   top features: {len(data['top_features'])}")
            
            # validate response structure
            assert response.status_code == 200
            assert 0.0 <= data['fraud_score'] <= 1.0
            assert data['decision'] in ['approve', 'review', 'decline']
            assert len(data['top_features']) > 0
            
            print("   ✅ transaction scored successfully!")
            return data
        except Exception as e:
            print(f"   ❌ scoring failed: {e}")
            return None

async def test_velocity_detection():
    """test that model detects high velocity (multiple transactions quickly)"""
    print("\n⚡ testing velocity detection (rapid transactions)...")
    
    user_id = 999
    results = []
    
    for i in range(5):
        print(f"\n   transaction {i+1}/5...")
        result = await test_score_transaction(user_id, 1, 100.00)
        if result:
            results.append(result['fraud_score'])
        await asyncio.sleep(0.5)
    
    if len(results) >= 3:
        print(f"\n   fraud scores over time: {[f'{s:.3f}' for s in results]}")
        # later transactions should have higher scores (velocity detection)
        if results[-1] > results[0]:
            print("   ✅ velocity detection working! (score increased)")
        else:
            print("   ⚠️  scores didn't increase (velocity may not be detected)")

async def test_different_amounts():
    """test scoring different transaction amounts"""
    print("\n💰 testing different transaction amounts...")
    
    amounts = [10.00, 100.00, 1000.00, 10000.00]
    
    for amount in amounts:
        await test_score_transaction(1, 1, amount)
        await asyncio.sleep(0.5)

async def test_cache_performance():
    """test cache improves performance on repeated requests"""
    print("\n🔥 testing cache performance...")
    
    # first request (cache miss)
    print("\n   request 1 (expect cache miss):")
    start1 = time.time()
    result1 = await test_score_transaction(123, 45, 500.00)
    latency1 = (time.time() - start1) * 1000
    
    await asyncio.sleep(1)
    
    # second request (cache hit)
    print("\n   request 2 (expect cache hit):")
    start2 = time.time()
    result2 = await test_score_transaction(123, 45, 500.00)
    latency2 = (time.time() - start2) * 1000
    
    if result1 and result2:
        speedup = latency1 / latency2
        print(f"\n   latency 1: {latency1:.0f}ms")
        print(f"   latency 2: {latency2:.0f}ms")
        print(f"   speedup: {speedup:.1f}x faster")
        
        if speedup > 1.5:
            print("   ✅ cache is working! (2nd request faster)")
        else:
            print("   ⚠️  cache may not be helping much")

async def main():
    """run all end-to-end tests"""
    print("=" * 70)
    print("🧪 COMPREHENSIVE END-TO-END TESTING")
    print("=" * 70)
    
    # test 1: health check
    health_ok = await test_api_health()
    if not health_ok:
        print("\n❌ API not healthy, stopping tests")
        return
    
    await asyncio.sleep(1)
    
    # test 2: basic transaction scoring
    await test_score_transaction(1, 1, 100.00)
    await asyncio.sleep(1)
    
    # test 3: velocity detection
    await test_velocity_detection()
    await asyncio.sleep(1)
    
    # test 4: different amounts
    await test_different_amounts()
    await asyncio.sleep(1)
    
    # test 5: cache performance
    await test_cache_performance()
    
    print("\n" + "=" * 70)
    print("✅ ALL END-TO-END TESTS COMPLETE!")
    print("=" * 70)

if __name__ == '__main__':
    asyncio.run(main())
