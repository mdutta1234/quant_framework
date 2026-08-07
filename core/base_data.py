from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class BaseUniverseSelector(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Must return a dictionary with 'train', 'val', and 'test' keys.
        Each value is a standardized Pandas DataFrame with:
        Columns: ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        """
        pass