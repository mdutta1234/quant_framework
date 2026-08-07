import pandas as pd
import numpy as np
from typing import Dict
from core.base_cleaner import BaseDataCleaner

class StandardCleaner(BaseDataCleaner):
    def clean(self, raw_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        cleaned_datasets = {}
        fill_method = self.config.get('fill_method', 'ffill')
        cols_to_fill = ['open', 'high', 'low', 'close', 'volume']

        for split_name, df in raw_datasets.items():
            print(f"Cleaning {split_name} data...")
            
            # 1. Remove duplicates
            df = df.drop_duplicates(subset=['ticker', 'date'])

            # 2. Drop tickers that are mostly missing (>30% NaNs)
            nan_ratio_per_ticker = df.groupby('ticker')['close'].apply(lambda x: x.isna().mean())
            bad_tickers = nan_ratio_per_ticker[nan_ratio_per_ticker > 0.30].index.tolist()
            if bad_tickers:
                print(f"  [CLEANER] Dropping severely incomplete tickers: {bad_tickers}")
                df = df[~df['ticker'].isin(bad_tickers)].copy()

            # 3. Alignment & Holidays
            all_dates = df['date'].sort_values().unique()
            tickers = df['ticker'].unique()

            multi_index = pd.MultiIndex.from_product([tickers, all_dates], names=['ticker', 'date'])
            df = df.set_index(['ticker', 'date']).reindex(multi_index).reset_index()

            # 4. Missing Values
            if fill_method == 'ffill':
                # Forward fill: Carry the last known price forward
                df[cols_to_fill] = df.groupby('ticker')[cols_to_fill].ffill()
                
                # Backward fill: Catch any NaNs at the very beginning
                df[cols_to_fill] = df.groupby('ticker')[cols_to_fill].bfill()

            # 5. Failsafe: Replace any remaining NaNs or Infs with 0.0
            df[cols_to_fill] = df[cols_to_fill].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            cleaned_datasets[split_name] = df
            
            nan_count = df.isna().sum().sum()
            print(f"  -> Cleaned {split_name} shape: {df.shape} | Total NaNs remaining: {nan_count}")
            assert nan_count == 0, f"Critical Error: {split_name} still contains {nan_count} NaNs!"

        return cleaned_datasets