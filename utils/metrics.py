# utils/metrics.py
import numpy as np
from scipy.stats import pearsonr, spearmanr

def calc_directional_accuracy(actual, predicted):
    # sign(A_t) == sign(P_t)
    actual_dir = np.sign(actual)
    pred_dir = np.sign(predicted)
    correct = np.sum(actual_dir == pred_dir)
    return correct / len(actual)

def calc_directional_f1(actual, predicted):
    actual_dir = np.sign(actual) > 0
    pred_dir = np.sign(predicted) > 0
    
    tp = np.sum(actual_dir & pred_dir)
    fp = np.sum(~actual_dir & pred_dir)
    fn = np.sum(actual_dir & ~pred_dir)
    
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_all_metrics(y_true, y_pred, y_naive, weights):
    """Calculates the full suite of quant metrics for a single feature array."""
    metrics = {}
    
    # Standard Error Metrics
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    metrics['MAE'] = np.mean(abs_errors)
    metrics['RMSE'] = np.sqrt(np.mean(errors**2))
    
    # Scaled/Relative Errors
    epsilon = 1e-8
    metrics['sMAPE'] = np.mean(2.0 * abs_errors / (np.abs(y_true) + np.abs(y_pred) + epsilon))
    
    naive_mae = np.mean(np.abs(y_true - y_naive))
    metrics['MASE'] = metrics['MAE'] / (naive_mae + epsilon)
    
    # Weighted Errors (Using your Huber Confidence Weights)
    weighted_abs_errors = abs_errors * weights
    weighted_sq_errors = (errors**2) * weights
    
    metrics['Weighted_MAE'] = np.sum(weighted_abs_errors) / (np.sum(weights) + epsilon)
    metrics['Weighted_RMSE'] = np.sqrt(np.sum(weighted_sq_errors) / (np.sum(weights) + epsilon))
    
    # Directional & Correlation
    metrics['DirAcc'] = calc_directional_accuracy(y_true, y_pred)
    metrics['DirF1'] = calc_directional_f1(y_true, y_pred)
    
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        metrics['RetCorr'], _ = pearsonr(y_true, y_pred)
        metrics['IC'], _ = spearmanr(y_true, y_pred) # Rank correlation for Info Coefficient
    else:
        metrics['RetCorr'] = 0.0
        metrics['IC'] = 0.0
        
    return metrics