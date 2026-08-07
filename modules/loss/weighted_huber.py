# modules/loss/weighted_huber.py
import torch
import torch.nn.functional as F
from core.base_loss import BaseLossFunction

class WeightedHuberLoss(BaseLossFunction):
    def __init__(self, config: dict):
        super().__init__(config)
        self.delta = config.get('delta', 1.0)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # 1. Calculate raw Huber Loss for every single feature independently
        # Shape: (batch_size, feature_dim)
        raw_loss = F.huber_loss(predictions, targets, reduction='none', delta=self.delta)
        
        # 2. Ensure weights broadcast correctly 
        # If weights are (batch_size,), reshape to (batch_size, 1) so it multiplies across all features
        if weights.dim() < raw_loss.dim():
            weights = weights.unsqueeze(-1)
            
        # 3. Apply the Mathematical Prior Confidence Weights
        weighted_loss = raw_loss * weights
        
        # 4. Return the mean scalar loss for backpropagation
        return weighted_loss.mean()