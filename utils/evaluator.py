# utils/evaluator.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import calculate_all_metrics

class QuantEvaluator:
    def __init__(self, models, scaler, config, feature_cols):
        self.models = models
        self.scaler = scaler
        self.config = config
        self.feature_cols = feature_cols
        self.seq_len = config['predictor']['seq_len']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for m in self.models.values():
            m.eval()

    def evaluate_and_plot(self, feature_datasets, alpha_val=None):
        print(f"\n[EVALUATOR] Running inference and generating plots...")
        all_metrics = []

        for split_name, df in feature_datasets.items():
            for ticker, group in df.groupby('ticker'):
                # Set up directory: plots/AAPL_train_data/
                save_dir = f"plots/{ticker}_{split_name}_data"
                os.makedirs(save_dir, exist_ok=True)
                
                # Extract data
                features = group[self.feature_cols].values.astype(np.float32)
                weights = group['weight'].values.astype(np.float32)
                local_id = group['local_id'].values[0]
                cluster_id = group['cluster_id'].values[0]
                
                model = self.models[cluster_id]
                
                # We will manually slide the window to keep chronological order for plotting
                y_true_list, y_pred_list, w_list = [], [], []
                
                with torch.no_grad():
                    for i in range(len(features) - self.seq_len):
                        x_window = torch.tensor(features[i : i + self.seq_len]).unsqueeze(0).to(self.device)
                        id_tensor = torch.tensor([local_id]).to(self.device)
                        
                        pred = model(x_window, id_tensor).cpu().numpy()[0]
                        
                        y_pred_list.append(pred)
                        y_true_list.append(features[i + self.seq_len])
                        w_list.append(weights[i + self.seq_len])
                        
                y_pred_scaled = np.array(y_pred_list)
                y_true_scaled = np.array(y_true_list)
                w_array = np.array(w_list)
                
                # Inverse transform to get real numbers (like price and volume) back
                y_pred_real = self.scaler.inverse_transform(y_pred_scaled)
                y_true_real = self.scaler.inverse_transform(y_true_scaled)
                
                # The naive forecast is just the value from t-1 (which is index i + seq_len - 1)
                y_naive_real = self.scaler.inverse_transform(features[self.seq_len - 1 : -1])
                
                # Calculate metrics for the primary target (e.g., closing price or log return)
                # Assuming index 3 is 'close', or we can average across all features. Let's do 'close'
                target_idx = self.feature_cols.index('close') if 'close' in self.feature_cols else 0
                
                stock_metrics = calculate_all_metrics(
                    y_true_real[:, target_idx], 
                    y_pred_real[:, target_idx], 
                    y_naive_real[:, target_idx], 
                    w_array
                )
                
                stock_metrics['ticker'] = ticker
                stock_metrics['split'] = split_name
                if alpha_val is not None:
                    stock_metrics['alpha'] = alpha_val
                    
                all_metrics.append(stock_metrics)
                
                # Only plot the feature graphs if we aren't in a massive alpha loop
                # or plot them just for the final run.
                if alpha_val is None or alpha_val == 0.5:
                    self._plot_features(y_true_real, y_pred_real, save_dir, ticker, split_name)
                    
        return pd.DataFrame(all_metrics)

    def _plot_features(self, y_true, y_pred, save_dir, ticker, split_name):
        for i, f_name in enumerate(self.feature_cols):
            plt.figure(figsize=(12, 6))
            plt.plot(y_true[:, i], label='Actual', color='black', alpha=0.7)
            plt.plot(y_pred[:, i], label='Predicted', color='blue', alpha=0.7, linestyle='--')
            plt.title(f"{ticker} - {split_name.upper()} - {f_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{f_name}.png"))
            plt.close()