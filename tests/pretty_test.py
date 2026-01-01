#!/usr/bin/env python3
"""
Pretty API tester - Makes fraud detection results easy to read
"""

import httpx
import sys
from datetime import datetime

def format_score(score: float) -> str:
    """Format score with color"""
    if score < 0.5:
        return f"🟢 {score:.4f} (LOW RISK)"
    elif score < 0.8:
        return f"🟡 {score:.4f} (MEDIUM RISK)"
    else:
        return f"🔴 {score:.4f} (HIGH RISK)"

def format_decision(decision: str) -> str:
    """Format decision with emoji"""
    emoji_map = {
        "approve": "✅ APPROVE",
        "review": "⚠️  REVIEW",
        "decline": "❌ DECLINE"
    }
    return emoji_map.get(decision.lower(), decision.upper())

def pretty_print_response(response_data: dict):
    """Pretty print the API response"""
    
    print("\n" + "="*80)
    print("🔍 FRAUD DETECTION RESULT".center(80))
    print("="*80 + "\n")
    
    # Main decision
    score = response_data.get('fraud_score', 0)
    decision = response_data.get('decision', 'unknown')
    reason = response_data.get('reason', 'N/A')
    
    print(f"  Decision:     {format_decision(decision)}")
    print(f"  Fraud Score:  {format_score(score)}")
    print(f"  Reason:       {reason}")
    
    # Metadata
    if 'metadata' in response_data:
        meta = response_data['metadata']
        print(f"\n  Model:        {meta.get('model_version', 'N/A')}")
        print(f"  Timestamp:    {meta.get('timestamp', 'N/A')}")
        
        if 'thresholds' in meta:
            thresh = meta['thresholds']
            print(f"  Thresholds:   Review: {thresh.get('review', 0):.2f} | Decline: {thresh.get('decline', 0):.2f}")
    
    # Top features
    print("\n" + "-"*80)
    print("📊 TOP FEATURES (What influenced this decision)".center(80))
    print("-"*80 + "\n")
    
    features = response_data.get('top_features', [])
    if features:
        print(f"  {'Feature':<30} {'Value':<20} {'Impact'}")
        print(f"  {'-'*30} {'-'*20} {'-'*20}")
        
        for i, feat in enumerate(features[:5], 1):
            name = feat.get('feature', 'unknown')
            value = feat.get('value', 0)
            interp = feat.get('interpretation', str(value))
            
            # Format value
            if isinstance(value, float):
                if value > 1000:
                    value_str = f"{value:,.0f}"
                else:
                    value_str = f"{value:.3f}"
            else:
                value_str = str(value)
            
            print(f"  {i}. {name:<27} {value_str:<20} {interp[:40]}")
    else:
        print("  No features available")
    
    print("\n" + "="*80 + "\n")

def test_transaction(user_id: int, merchant_id: int, amount: float):
    """Test a transaction"""
    
    api_url = "http://localhost:8000/score"
    
    payload = {
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount
    }
    
    print(f"\n🔄 Testing transaction...")
    print(f"   User ID:     {user_id}")
    print(f"   Merchant ID: {merchant_id}")
    print(f"   Amount:      ${amount:,.2f}")
    
    try:
        response = httpx.post(api_url, json=payload, timeout=30.0)
        response.raise_for_status()
        
        data = response.json()
        pretty_print_response(data)
        
    except httpx.HTTPError as e:
        print(f"\n❌ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command line args or use defaults
    if len(sys.argv) == 4:
        user_id = int(sys.argv[1])
        merchant_id = int(sys.argv[2])
        amount = float(sys.argv[3])
    else:
        # Default test case
        user_id = 1
        merchant_id = 1
        amount = 100.00
        
        print("\n💡 Usage: python tests/pretty_test.py <user_id> <merchant_id> <amount>")
        print(f"   Using defaults: user={user_id}, merchant={merchant_id}, amount=${amount}")
    
    test_transaction(user_id, merchant_id, amount)
