# core/base_loss.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseLossFunction(nn.Module, ABC):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        predictions: (batch_size, feature_dim)
        targets: (batch_size, feature_dim)
        weights: (batch_size, 1) or (batch_size, feature_dim)
        
        Returns a scalar tensor representing the weighted loss.
        """
        pass