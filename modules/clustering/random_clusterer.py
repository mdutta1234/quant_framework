# modules/clustering/random_clusterer.py
import numpy as np
import pandas as pd
from typing import Dict
from core.base_clusterer import BaseClusterer

class RandomClusterer(BaseClusterer):
    
    def fit_predict(self, train_features: pd.DataFrame) -> Dict[str, int]:
        print("\n[CLUSTERING] Executing Random Clustering strategy...")
        tickers = train_features['ticker'].unique()
        
        # 1. Extract constraints
        constraints = self.config.get('constraints', {})
        target_clusters = constraints.get('target_clusters', 5)
        max_size = constraints.get('max_size', 1)
        
        total_needed = target_clusters * max_size
        
        # Failsafe check
        if total_needed > len(tickers):
            raise ValueError(
                f"Need {total_needed} stocks for {target_clusters} clusters of size {max_size}, "
                f"but only {len(tickers)} valid stocks remain in the universe."
            )
            
        # 2. Set seed for reproducibility (Optional but good for quant pipelines)
        seed = self.config.get('random_seed', 42)
        np.random.seed(seed)
        
        # 3. Randomly select the exact number of tickers needed
        selected_tickers = np.random.choice(tickers, size=total_needed, replace=False)
        
        # 4. Assign them to clusters sequentially
        cluster_mapping = {}
        for i, ticker in enumerate(selected_tickers):
            # Integer division determines the cluster ID (e.g., 0,0,0, 1,1,1, 2,2,2)
            cluster_id = i // max_size
            cluster_mapping[ticker] = cluster_id
            
        print(f"[CLUSTERING] Randomly selected {total_needed} stocks and distributed into {target_clusters} clusters.")
        
        # Print assignments for visibility
        for c_id in range(target_clusters):
            c_tickers = [t for t, cid in cluster_mapping.items() if cid == c_id]
            print(f"  -> Selected Cluster {c_id} | Size: {len(c_tickers)} | Tickers: {c_tickers}")
            
        return cluster_mapping