# core/base_clusterer.py
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

class BaseClusterer(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.descriptors_list = config.get('descriptors', [])
        
    @abstractmethod
    def fit_predict(self, train_features: pd.DataFrame) -> Dict[str, int]:
        """
        Takes the TRAIN feature dataframe.
        Calculates descriptors, performs clustering, scores clusters.
        Returns a mapping of {ticker: cluster_id} ONLY for selected stocks.
        """
        pass