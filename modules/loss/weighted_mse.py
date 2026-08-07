# modules/loss/weighted_mse.py
import torch
from core.base_loss import BaseLossFunction

class WeightedMSELoss(BaseLossFunction):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # 1. Squared error: (y_hat - y)^2
        raw_loss = (predictions - targets) ** 2
        
        if weights.dim() < raw_loss.dim():
            weights = weights.unsqueeze(-1)
            
        # 2. Apply weights and mean
        return (raw_loss * weights).mean()