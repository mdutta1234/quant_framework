# core/base_predictor.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePredictor(nn.Module, ABC):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.id_dim = config['id_dim'] # Automatically injected based on cluster size

    @abstractmethod
    def forward(self, x: torch.Tensor, stock_id: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, input_dim)
        stock_id shape: (batch_size,) 
        Returns: (batch_size, output_dim) -> The predicted next feature vector
        """
        pass