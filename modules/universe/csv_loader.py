import pandas as pd
from core.base_data import BaseUniverseSelector

class CSVUniverseSelector(BaseUniverseSelector):
    def fetch_data(self) -> pd.DataFrame:
        file_path = self.config.get('file_path', './data/raw_market_data.csv')
        print(f"[CSVLoader] Loading universe data from {file_path}...")
        
        df = pd.read_csv(file_path)
        
        # Enforce standard column naming
        df.columns = [col.lower() for col in df.columns]
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns. Expected: {required_cols}")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        
        return df