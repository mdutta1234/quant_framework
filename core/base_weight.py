# core/base_weight.py
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class BaseWeightGenerator(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate_weights(self, actual_datasets: Dict[str, pd.DataFrame], expected_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Receives the actual features and expected prior features.
        Returns DataFrames with ['ticker', 'date', 'weight']
        """
        pass