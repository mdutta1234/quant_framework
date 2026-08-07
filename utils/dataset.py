# utils/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class QuantTimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, feature_cols: list):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        
        x_list, y_list, w_list, id_list, cid_list = [], [], [], [], []
        
        for ticker, group in df.groupby('ticker'):
            features = group[self.feature_cols].values.astype(np.float32)
            weights = group['weight'].values.astype(np.float32)
            local_ids = group['local_id'].values.astype(np.int64)     # For One-Hot
            cluster_ids = group['cluster_id'].values.astype(np.int64) # For routing
            
            for i in range(len(group) - self.seq_len):
                x_list.append(features[i : i + self.seq_len])
                y_list.append(features[i + self.seq_len])
                w_list.append(weights[i + self.seq_len])
                id_list.append(local_ids[i + self.seq_len])
                cid_list.append(cluster_ids[i + self.seq_len])
                
        self.x_data = torch.tensor(np.array(x_list))
        self.y_data = torch.tensor(np.array(y_list))
        self.w_data = torch.tensor(np.array(w_list))
        self.id_data = torch.tensor(np.array(id_list))
        self.cid_data = torch.tensor(np.array(cid_list))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.id_data[idx], self.cid_data[idx], self.y_data[idx], self.w_data[idx]