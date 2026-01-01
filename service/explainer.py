# model explainability - shows why model made a decision
# computes feature importance for individual predictions

import numpy as np
from typing import List, Dict, Tuple

def explain_prediction(
    feature_vector: np.ndarray,
    feature_names: List[str],
    top_n: int = 5
) -> List[Dict]:
    """
    explain prediction by showing most important features
    uses feature magnitude as proxy for importance
    
    args:
        feature_vector: computed features for transaction
        feature_names: ordered list of feature names
        top_n: how many top features to return
        
    returns:
        list of dicts with feature name, value, and contribution
    """
    
    # normalize features to see relative importance
    # features with larger absolute values contribute more
    feature_abs = np.abs(feature_vector)
    
    # get indices of top features by magnitude
    top_indices = np.argsort(feature_abs)[::-1][:top_n]
    
    explanations = []
    for idx in top_indices:
        explanations.append({
            'feature': feature_names[idx],
            'value': float(feature_vector[idx]),
            'importance': float(feature_abs[idx]),
            'interpretation': _interpret_feature(
                feature_names[idx],
                feature_vector[idx]
            )
        })
    
    return explanations

def _interpret_feature(feature_name: str, value: float) -> str:
    """
    human-readable interpretation of feature value
    
    args:
        feature_name: name of feature
        value: feature value
        
    returns:
        human-readable string
    """
    
    interpretations = {
        'tx_count_10m': f"{int(value)} transactions in last 10 minutes",
        'tx_count_1h': f"{int(value)} transactions in last hour",
        'tx_count_24h': f"{int(value)} transactions in last 24 hours",
        'spend_1h': f"${value:.2f} spent in last hour",
        'spend_24h': f"${value:.2f} spent in last 24 hours",
        'avg_amount_1h': f"${value:.2f} average transaction amount",
        'unique_merchants_24h': f"{int(value)} different merchants visited",
        'merchant_fraud_rate_30d': f"{value*100:.1f}% merchant fraud rate",
        'amount': f"${value:.2f} current transaction amount",
        'time_since_last_tx': f"{value:.0f} seconds since last transaction"
    }
    
    # check if it's a known feature
    if feature_name in interpretations:
        return interpretations[feature_name]
    
    # handle v1-v28 features
    if feature_name.startswith('v'):
        return f"PCA feature {feature_name} = {value:.3f}"
    
    return f"{feature_name} = {value:.3f}"

def generate_fraud_reason(
    explanations: List[Dict],
    fraud_score: float
) -> str:
    """
    generate human-readable reason for fraud decision
    
    args:
        explanations: top feature explanations
        fraud_score: predicted fraud score
        
    returns:
        explanation string
    """
    
    if fraud_score < 0.3:
        return "Transaction appears normal with no significant risk factors"
    
    # get top 3 contributing factors
    top_factors = explanations[:3]
    
    reasons = []
    for factor in top_factors:
        feature = factor['feature']
        interp = factor['interpretation']
        
        # flag suspicious patterns
        if feature == 'tx_count_10m' and factor['value'] >= 3:
            reasons.append(f"High velocity: {interp}")
        elif feature == 'tx_count_1h' and factor['value'] >= 6:
            reasons.append(f"Unusual activity: {interp}")
        elif feature == 'merchant_fraud_rate_30d' and factor['value'] >= 0.05:
            reasons.append(f"Risky merchant: {interp}")
        elif feature == 'unique_merchants_24h' and factor['value'] >= 10:
            reasons.append(f"Card testing pattern: {interp}")
        elif feature == 'time_since_last_tx' and factor['value'] < 30:
            reasons.append(f"Bot-like speed: {interp}")
        elif feature == 'spend_24h' and factor['value'] >= 5000:
            reasons.append(f"High spending: {interp}")
    
    if reasons:
        return " + ".join(reasons)
    else:
        return f"Model detected suspicious patterns (score: {fraud_score:.2f})"

def explain_prediction_gradient(
    model,
    feature_vector: np.ndarray,
    feature_names: List[str],
    top_n: int = 5
) -> List[Dict]:
    """
    enhanced explainability using gradient-based attribution
    shows which features the MODEL thinks are most important
    
    args:
        model: trained pytorch model
        feature_vector: computed features for transaction
        feature_names: ordered list of feature names
        top_n: how many top features to return
        
    returns:
        list of dicts with feature name, value, gradient importance
    """
    
    import torch
    
    # convert to tensor and enable gradient tracking
    x = torch.FloatTensor(feature_vector).unsqueeze(0)
    x.requires_grad = True
    
    # forward pass
    model.eval()
    output = model(x)
    
    # backward pass to get gradients
    output.backward()
    
    # gradient magnitude = how much changing this feature affects output
    gradients = x.grad.abs().squeeze().numpy()
    
    # multiply gradients by feature values for attribution
    # (gradient * value = contribution to final score)
    attribution = gradients * np.abs(feature_vector)
    
    # get indices of top contributing features
    top_indices = np.argsort(attribution)[::-1][:top_n]
    
    explanations = []
    for idx in top_indices:
        explanations.append({
            'feature': feature_names[idx],
            'value': float(feature_vector[idx]),
            'gradient': float(gradients[idx]),
            'attribution': float(attribution[idx]),
            'interpretation': _interpret_feature(
                feature_names[idx],
                feature_vector[idx]
            ),
            'impact': _describe_impact(attribution[idx])
        })
    
    return explanations

def _describe_impact(attribution_score: float) -> str:
    """describe the impact level of a feature"""
    if attribution_score > 0.5:
        return "very high impact"
    elif attribution_score > 0.2:
        return "high impact"
    elif attribution_score > 0.1:
        return "moderate impact"
    else:
        return "low impact"

