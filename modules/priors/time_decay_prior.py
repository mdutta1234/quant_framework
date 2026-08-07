# modules/priors/time_decay_prior.py
import pandas as pd
import numpy as np
from typing import Dict
from core.base_prior import BaseMathPrior

class TimeDecayEnsemblePrior(BaseMathPrior):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.L = config.get('window_length', 50)
        self.E = config.get('ensemble_size', 20)
        self.decay_type = config.get('decay_type', 'polynomial')
        self.EPSILON = 1e-8
        
        # Pre-calculate the time-decay weights (w_k)
        # k represents "how many days ago the prediction was made" (1 to E)
        k = np.arange(1, self.E + 1)
        
        if self.decay_type == 'polynomial':
            # e.g., (E - k)^2 -> Recent days get high weight, distant days approach 0
            w = (self.E - k) ** 2
        elif self.decay_type == 'exponential':
            w = np.exp(-0.1 * k)
        else:
            w = np.ones_like(k) # Equal weighting
            
        # Normalize weights so they sum to 1
        self.w = w / (w.sum() + self.EPSILON)
        
        # We also need weights for the drift multiplier (w_k * k)
        self.wk = self.w * k

    def fit(self, train_features: pd.DataFrame):
        print(f"[PRIOR] Initialized Time-Decay Ensemble Prior.")
        print(f"        -> Drift Window: {self.L} days | Ensemble Size: {self.E} vantage points.")
        pass

    def generate_prior(self, feature_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        expected_datasets = {}
        exclude_cols = ['ticker', 'date', 'cluster_id']

        for split_name, df in feature_datasets.items():
            print(f"[PRIOR] Generating Time-Decay Consensus Prior for {split_name} split...")
            
            expected_df = df.copy()
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            # Enforce dynamic float typing to prevent LossySetitemErrors
            expected_df[feature_cols] = expected_df[feature_cols].astype(float)
            
            for ticker, ticker_group in df.groupby('ticker'):
                ticker_feats = ticker_group[feature_cols].astype(float)
                
                # 1. Calculate drift (mu) for every point in time using the past L days
                mu = ticker_feats.diff().rolling(window=self.L, min_periods=1).mean().fillna(0.0)
                
                # 2. Vectorized Ensemble Forecasting using Rolling Dot Products
                # The prediction for time T from time T-k is: P = X_{T-k} + k * mu_{T-k}
                # The weighted sum is: Sum(w_k * X_{T-k}) + Sum(w_k * k * mu_{T-k})
                
                # We use [::-1] because rolling window places the oldest value at index 0 
                # and newest at index E-1. We want w_1 (most recent) to multiply index E-1.
                w_rev = self.w[::-1]
                wk_rev = self.wk[::-1]
                
                def apply_weights(x, weights):
                    # Failsafe for the beginning of the series where window < E
                    if len(x) < self.E:
                        # Pad weights to match length and re-normalize
                        w_sub = weights[-len(x):]
                        return np.dot(x, w_sub) / (w_sub.sum() + self.EPSILON)
                    return np.dot(x, weights)

                # Convolution 1: The Base State (X)
                base_state = ticker_feats.shift(1).fillna(0.0).rolling(
                    window=self.E, min_periods=1
                ).apply(lambda x: apply_weights(x, w_rev), raw=True)
                
                # Convolution 2: The Drift component (k * mu)
                drift_state = mu.shift(1).fillna(0.0).rolling(
                    window=self.E, min_periods=1
                ).apply(lambda m: apply_weights(m, wk_rev), raw=True)
                
                # 3. Consensus Prior = Weighted Base + Weighted Drift
                consensus_prior = base_state + drift_state
                
                # Inject back into dataframe safely
                expected_df.loc[ticker_group.index, feature_cols] = consensus_prior.values

            # --- ZERO NaN/Inf INVARIANT ---
            expected_df[feature_cols] = expected_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            nan_count = expected_df.isna().sum().sum()
            assert nan_count == 0, f"Critical Error: Prior matrix ({split_name}) contains {nan_count} NaNs!"
            
            expected_datasets[split_name] = expected_df
            print(f"  -> Generated {split_name} Prior matrix shape: {expected_df.shape}")

        return expected_datasets