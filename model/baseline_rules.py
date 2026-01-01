# simple rule-based fraud detection for baseline comparison
# these are heuristics that a human might write without ml

def rule_based_fraud_detection(features: dict) -> tuple[float, str]:
    """
    simple rule-based fraud scoring
    
    args:
        features: dict with computed features
            - tx_count_10m, tx_count_1h, tx_count_24h
            - spend_1h, spend_24h, avg_amount_1h
            - unique_merchants_24h
            - merchant_fraud_rate_30d
            - amount
            - time_since_last_tx
            
    returns:
        (fraud_score, decision) where score is 0-1 and decision is approve/review/decline
    """
    
    fraud_score = 0.0
    
    # rule 1: high velocity in short time (bot-like behavior)
    if features.get('tx_count_10m', 0) >= 5:
        fraud_score += 0.35  # 5+ transactions in 10 min is very suspicious
    elif features.get('tx_count_10m', 0) >= 3:
        fraud_score += 0.20
    
    # rule 2: high transaction count in 1 hour
    if features.get('tx_count_1h', 0) >= 10:
        fraud_score += 0.25
    elif features.get('tx_count_1h', 0) >= 6:
        fraud_score += 0.15
    
    # rule 3: spending spike compared to average
    avg_amount = features.get('avg_amount_1h', 1)
    current_amount = features.get('amount', 0)
    
    if avg_amount > 0 and current_amount > avg_amount * 5:
        fraud_score += 0.25  # 5x larger than normal
    elif avg_amount > 0 and current_amount > avg_amount * 3:
        fraud_score += 0.15
    
    # rule 4: high merchant fraud rate (risky merchant)
    merchant_fraud_rate = features.get('merchant_fraud_rate_30d', 0)
    if merchant_fraud_rate > 0.10:  # 10% fraud rate
        fraud_score += 0.20
    elif merchant_fraud_rate > 0.05:  # 5% fraud rate
        fraud_score += 0.10
    
    # rule 5: testing card at many merchants (card testing)
    unique_merchants = features.get('unique_merchants_24h', 0)
    if unique_merchants > 15:
        fraud_score += 0.20
    elif unique_merchants > 10:
        fraud_score += 0.10
    
    # rule 6: very fast transactions (bot behavior)
    time_since_last = features.get('time_since_last_tx', 9999)
    if time_since_last < 10:  # less than 10 seconds
        fraud_score += 0.15
    elif time_since_last < 30:
        fraud_score += 0.08
    
    # rule 7: high daily spend
    spend_24h = features.get('spend_24h', 0)
    if spend_24h > 10000:
        fraud_score += 0.15
    elif spend_24h > 5000:
        fraud_score += 0.08
    
    # cap score at 1.0
    fraud_score = min(fraud_score, 1.0)
    
    # make decision based on score
    if fraud_score >= 0.7:
        decision = "decline"
    elif fraud_score >= 0.4:
        decision = "review"
    else:
        decision = "approve"
    
    return fraud_score, decision

def get_rule_names():
    """
    return list of rules used in baseline
    useful for explaining what the rules are checking
    """
    return [
        "high_velocity_10m (5+ tx in 10 min)",
        "high_velocity_1h (10+ tx in 1 hour)",
        "spending_spike (5x normal amount)",
        "risky_merchant (>10% fraud rate)",
        "card_testing (15+ merchants in 24h)",
        "bot_speed (<10s between tx)",
        "high_daily_spend (>$10k in 24h)"
    ]
