# core/base_features.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd

class BaseFeatureEngineer(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.features_list = config.get('selected_features', [])

    @abstractmethod
    def process(self, cleaned_datasets: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], int]:
        """
        Receives cleaned DataFrames and returns:
        1. Processed DataFrames with calculated features.
        2. The integer dimension of the feature space (for Neural Nets).
        """
        pass