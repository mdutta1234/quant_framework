# modules/priors/abm_prior.py
import pandas as pd
import numpy as np
from typing import Dict
from core.base_prior import BaseMathPrior

class EnsembleABMPrior(BaseMathPrior):
    
    def fit(self, train_features: pd.DataFrame):
        # Brownian motion relies on rolling calculations, so no static fitting is required.
        print(f"[PRIOR] Initialized Ensemble Arithmetic Brownian Motion Prior (Windows: {self.windows})")
        pass

    def generate_prior(self, feature_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        expected_datasets = {}
        
        # We don't model metadata columns
        exclude_cols = ['ticker', 'date', 'cluster_id']

        for split_name, df in feature_datasets.items():
            print(f"[PRIOR] Generating Consensus Prior for {split_name} split...")
            
            # Create a dataframe to hold expected values
            expected_df = df.copy()
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            # ---> THE FIX: Cast all target columns to float before assigning decimals <---
            expected_df[feature_cols] = expected_df[feature_cols].astype(float)
            
            for ticker, ticker_group in df.groupby('ticker'):
                # 1. Isolate the feature values
                ticker_feats = ticker_group[feature_cols]
                
                # 2. X_{t-1} : Yesterday's exact values
                yesterday = ticker_feats.shift(1).fillna(0.0)
                
                # 3. Calculate historical daily differences (dX)
                daily_diff = ticker_feats.diff()
                
                # 4. PREVENT LOOK-AHEAD BIAS: 
                # Use differences up to t-1 to calculate the drift for predicting t.
                past_diffs = daily_diff.shift(1).fillna(0.0)
                
                # 5. Ensemble Window Drift Calculation
                ensemble_expected = np.zeros_like(yesterday.values, dtype=float)
                
                for w in self.windows:
                    # Calculate rolling mu (drift) over the specific window
                    rolling_mu = past_diffs.rolling(window=w, min_periods=1).mean().fillna(0.0)
                    
                    # E[X_t] = X_{t-1} + mu_{t-1}
                    expected_w = yesterday + rolling_mu
                    ensemble_expected += expected_w.values
                    
                # 6. Average the ensemble to get the Consensus Prior
                ensemble_expected /= len(self.windows)
                
                # Inject back into the dataframe safely
                expected_df.loc[ticker_group.index, feature_cols] = ensemble_expected

            # --- ZERO NaN/Inf INVARIANT ---
            expected_df[feature_cols] = expected_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            nan_count = expected_df.isna().sum().sum()
            assert nan_count == 0, f"Critical Error: Prior matrix ({split_name}) contains {nan_count} NaNs!"
            
            expected_datasets[split_name] = expected_df
            print(f"  -> Generated {split_name} Prior matrix shape: {expected_df.shape}")

        return expected_datasets