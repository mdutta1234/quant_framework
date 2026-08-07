# modules/weights/huber_weights.py
import pandas as pd
import numpy as np
from typing import Dict
from core.base_weight import BaseWeightGenerator

class HuberWeightGenerator(BaseWeightGenerator):
    
    def generate_weights(self, actual_datasets: Dict[str, pd.DataFrame], expected_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        weight_datasets = {}
        
        alpha = self.config.get('alpha', 0.5)
        delta = self.config.get('delta', 1.0)
        min_weight = self.config.get('min_weight', 0.1)
        
        print(f"[WEIGHTS] Generating Huber Weights (alpha={alpha}, delta={delta})")
        
        # If alpha is 0, we can skip the heavy math entirely
        if alpha == 0.0:
            print("  -> Alpha is 0. Returning uniform weights (1.0).")
        
        exclude_cols = ['ticker', 'date', 'cluster_id']

        for split_name in actual_datasets.keys():
            actual_df = actual_datasets[split_name]
            expected_df = expected_datasets[split_name]
            
            feature_cols = [c for c in actual_df.columns if c not in exclude_cols]
            
            # 1. Calculate absolute errors across all features
            errors = np.abs(actual_df[feature_cols].values - expected_df[feature_cols].values)
            
            # 2. Aggregate the error into a single scalar per row (Mean Absolute Error across features)
            row_errors = np.mean(errors, axis=1)
            
            # 3. Robust Scaling of Errors
            # We scale the errors by the median and IQR of the training set so alpha is universally applicable
            if split_name == 'train':
                self.median_error = np.median(row_errors)
                self.iqr_error = np.percentile(row_errors, 75) - np.percentile(row_errors, 25) + 1e-8
                
            scaled_errors = (row_errors - self.median_error) / self.iqr_error
            
            # Ensure errors are positive
            scaled_errors = np.maximum(scaled_errors, 0.0)
            
            # 4. Huber Transformation
            # H(e) = 0.5 * e^2 if e <= delta, else delta * (e - 0.5 * delta)
            is_small_error = scaled_errors <= delta
            huber_loss = np.where(
                is_small_error,
                0.5 * (scaled_errors ** 2),
                delta * (scaled_errors - 0.5 * delta)
            )
            
            # 5. Weight Calculation: (1 - alpha * HuberLoss)
            if alpha == 0.0:
                weights = np.ones_like(huber_loss)
            else:
                weights = 1.0 - (alpha * huber_loss)
                # Clip to prevent negative weights and enforce the min_weight floor
                weights = np.clip(weights, a_min=min_weight, a_max=1.0)
            
            # 6. Construct output DataFrame
            weight_df = actual_df[['ticker', 'date']].copy()
            weight_df['weight'] = weights
            
            # --- ZERO NaN INVARIANT ---
            nan_count = weight_df['weight'].isna().sum()
            assert nan_count == 0, f"Critical Error: Weight vector contains {nan_count} NaNs!"
            
            weight_datasets[split_name] = weight_df
            print(f"  -> Generated {split_name} weights | Mean: {weights.mean():.3f} | Min: {weights.min():.3f}")

        return weight_datasets