# model evaluation metrics and visualization
# computes precision, recall, pr-auc and generates comparison plots

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server use

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    evaluate model performance on test set
    
    args:
        model: trained pytorch model
        X_test: test features
        y_test: test labels
        threshold: classification threshold (default 0.5)
        
    returns:
        dict of metrics
    """
    
    model.eval()
    
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        y_pred_proba = model.predict_proba(X_test_t).numpy()
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    # compute metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # pr curve and auc
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recalls, precisions)
    
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'threshold': threshold,
        'y_pred_proba': y_pred_proba,
        'precisions': precisions,
        'recalls': recalls,
    }
    
    return metrics

def plot_metrics(model, X_test, y_test, save_path='evaluation_plots.png'):
    """
    generate evaluation plots: pr curve, confusion matrix, score distribution
    
    args:
        model: trained model
        X_test, y_test: test data
        save_path: where to save plot
    """
    
    metrics = evaluate_model(model, X_test, y_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # plot 1: precision-recall curve
    axes[0].plot(metrics['recalls'], metrics['precisions'], linewidth=2)
    axes[0].set_xlabel('recall')
    axes[0].set_ylabel('precision')
    axes[0].set_title(f'precision-recall curve\npr-auc = {metrics["pr_auc"]:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # plot 2: confusion matrix
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    im = axes[1].imshow(cm, cmap='Blues')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['predicted\nlegit', 'predicted\nfraud'])
    axes[1].set_yticklabels(['actual\nlegit', 'actual\nfraud'])
    axes[1].set_title('confusion matrix')
    
    # annotate cells
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=14)
    
    # plot 3: score distribution
    fraud_scores = metrics['y_pred_proba'][y_test == 1]
    legit_scores = metrics['y_pred_proba'][y_test == 0]
    
    axes[2].hist(legit_scores, bins=50, alpha=0.6, label='legitimate', color='green')
    axes[2].hist(fraud_scores, bins=50, alpha=0.6, label='fraud', color='red')
    axes[2].axvline(x=0.5, color='black', linestyle='--', label='threshold=0.5')
    axes[2].set_xlabel('fraud score')
    axes[2].set_ylabel('count')
    axes[2].set_title('score distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   saved plots to {save_path}")
    plt.close()

def compare_models(model, X_test, y_test, baseline_predictions):
    """
    compare ml model against rule-based baseline
    
    args:
        model: trained pytorch model
        X_test, y_test: test data
        baseline_predictions: predictions from rule-based system
        
    returns:
        comparison dict
    """
    
    # evaluate ml model
    ml_metrics = evaluate_model(model, X_test, y_test)
    
    # evaluate baseline
    baseline_pred = (baseline_predictions >= 0.5).astype(int)
    baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)
    baseline_precisions, baseline_recalls, _ = precision_recall_curve(y_test, baseline_predictions)
    baseline_pr_auc = auc(baseline_recalls, baseline_precisions)
    
    comparison = {
        'ml': {
            'precision': ml_metrics['precision'],
            'recall': ml_metrics['recall'],
            'pr_auc': ml_metrics['pr_auc'],
            'f1': ml_metrics['f1']
        },
        'baseline': {
            'precision': baseline_precision,
            'recall': baseline_recall,
            'pr_auc': baseline_pr_auc,
            'f1': f1_score(y_test, baseline_pred, zero_division=0)
        }
    }
    
    # calculate improvement
    for metric in ['precision', 'recall', 'pr_auc', 'f1']:
        ml_value = comparison['ml'][metric]
        baseline_value = comparison['baseline'][metric]
        
        if baseline_value > 0:
            improvement = ((ml_value - baseline_value) / baseline_value) * 100
            comparison[f'{metric}_improvement_pct'] = improvement
    
    return comparison
