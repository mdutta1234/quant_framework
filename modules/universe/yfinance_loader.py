import yfinance as yf
import pandas as pd
from typing import Dict
from core.base_data import BaseUniverseSelector

class YFinanceUniverseSelector(BaseUniverseSelector):
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        tickers = self.config.get('tickers', [])
        splits = self.config.get('splits', {})
        
        result_dfs = {}
        
        for split_name, dates in splits.items():
            start_date = dates['start']
            end_date = dates['end']
            print(f"Downloading {split_name} data ({start_date} to {end_date})...")
            
            # Download all tickers for this split
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            processed_data = []
            valid_tickers = []
            
            for ticker in tickers:
                try:
                    # Handle YFinance's output format (MultiIndex if >1 ticker)
                    if len(tickers) > 1:
                        # Ensure the ticker actually exists in the downloaded columns to avoid KeyErrors
                        if ticker not in raw_data.columns.levels[1]:
                            print(f"  [WARNING] {ticker} missing from download. Dropping.")
                            continue
                        ticker_df = raw_data.xs(ticker, level=1, axis=1).copy()
                    else:
                        ticker_df = raw_data.copy()
                        
                    # ---> THE FIX: Check for completely broken or empty downloads <---
                    if ticker_df.empty or ('Close' in ticker_df.columns and ticker_df['Close'].isna().all()):
                        print(f"  [WARNING] Dropping {ticker} from {split_name} split (No valid price data).")
                        continue
                        
                    ticker_df.reset_index(inplace=True)
                    ticker_df['ticker'] = ticker
                    processed_data.append(ticker_df)
                    valid_tickers.append(ticker)
                    
                except Exception as e:
                    print(f"  [WARNING] Could not process {ticker}: {e}")
                    continue
                    
            if not processed_data:
                raise ValueError(f"No valid data could be downloaded for the {split_name} split.")
                
            # Combine, clean, and standardize columns
            split_df = pd.concat(processed_data, ignore_index=True)
            split_df.columns = [col.lower() for col in split_df.columns]
            
            # Enforce strict column order and types
            required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            split_df = split_df[required_cols].copy()
            split_df['date'] = pd.to_datetime(split_df['date'])
            split_df = split_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
            
            result_dfs[split_name] = split_df
            print(f"  -> {split_name} shape: {split_df.shape} | Active Tickers: {len(valid_tickers)}")
            
        return result_dfs