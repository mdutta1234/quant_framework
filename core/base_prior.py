# core/base_prior.py
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class BaseMathPrior(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.windows = config.get('windows', [20, 50, 100])

    @abstractmethod
    def fit(self, train_features: pd.DataFrame):
        """
        Fit any static parameters (e.g., Maximum Likelihood Estimation for GARCH).
        For rolling priors like Brownian Motion, this may be a pass.
        """
        pass

    @abstractmethod
    def generate_prior(self, feature_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Receives the actual feature datasets.
        Returns identical DataFrames where every value is the *expected* value for time t,
        calculated using ONLY data from time t-1 and earlier.
        """
        pass