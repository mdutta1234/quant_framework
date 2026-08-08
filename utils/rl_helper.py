# utils/rl_helper.py
import torch
import pandas as pd
import numpy as np

def generate_predictor_outputs(models: dict, feature_datasets: dict, scaler, config: dict) -> dict:
    """
    Generates out-of-sample prediction matrices across train, val, and test splits 
    using frozen PyTorch predictors.
    """
    print("\n[RL HELPER] Pre-computing frozen predictor outputs for high-speed RL training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seq_len = config['predictor']['seq_len']
    exclude_cols = ['ticker', 'date', 'cluster_id', 'local_id', 'weight']
    feature_cols = [c for c in feature_datasets['train'].columns if c not in exclude_cols]
    
    close_idx = feature_cols.index('close') if 'close' in feature_cols else 0
    
    predictions_by_split = {}
    
    for split_name, df in feature_datasets.items():
        results = []
        
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('date').reset_index(drop=True)
            features = group[feature_cols].values.astype(np.float32)
            cluster_id = group['cluster_id'].values[0]
            local_id = group['local_id'].values[0]
            
            model = models[cluster_id]
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                for i in range(len(group)):
                    if i < seq_len:
                        results.append({'ticker': ticker, 'date': group.loc[i, 'date'], 'pred_return': 0.0})
                    else:
                        x_win = torch.tensor(features[i - seq_len : i]).unsqueeze(0).to(device)
                        id_tens = torch.tensor([local_id]).to(device)
                        
                        pred_scaled = model(x_win, id_tens).cpu().numpy()[0]
                        pred_real = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
                        
                        curr_real_close = scaler.inverse_transform(features[i - 1].reshape(1, -1))[0][close_idx]
                        pred_real_close = pred_real[close_idx]
                        
                        pred_return = (pred_real_close - curr_real_close) / (curr_real_close + 1e-8)
                        results.append({'ticker': ticker, 'date': group.loc[i, 'date'], 'pred_return': pred_return})
                        
        predictions_by_split[split_name] = pd.DataFrame(results)
        
    print("  -> Predictor outputs successfully synthesized.")
    return predictions_by_split