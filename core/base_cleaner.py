# core/base_cleaner.py
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class BaseDataCleaner(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def clean(self, raw_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Receives raw DataFrames and returns cleaned DataFrames.
        Must handle alignment, duplicates, and missing values.
        """
        pass