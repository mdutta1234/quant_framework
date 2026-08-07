# modules/weights/equal_weights.py
import pandas as pd
import numpy as np
from typing import Dict
from core.base_weight import BaseWeightGenerator

class EqualWeightGenerator(BaseWeightGenerator):
    
    def generate_weights(self, actual_datasets: Dict[str, pd.DataFrame], expected_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        print("[WEIGHTS] Generating Equal Weights (Baseline Model). All weights = 1.0")
        weight_datasets = {}
        
        for split_name, actual_df in actual_datasets.items():
            weight_df = actual_df[['ticker', 'date']].copy()
            weight_df['weight'] = 1.0
            weight_datasets[split_name] = weight_df
            
        return weight_datasets