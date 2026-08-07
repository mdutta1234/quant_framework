# modules/predictors/lstm_cnn_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_predictor import BasePredictor

class LSTM_CNN_MLP(BasePredictor):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.use_embedding = config.get('use_embedding', False)
        
        # 1. Handle Stock Identifier Dimension
        if self.use_embedding:
            self.emb_dim = config.get('embedding_dim', 8)
            self.embedding = nn.Embedding(num_embeddings=self.id_dim, embedding_dim=self.emb_dim)
            lstm_input_size = self.input_dim + self.emb_dim
        else:
            # For one-hot, the added dimension is simply the number of stocks in this cluster
            lstm_input_size = self.input_dim + self.id_dim
            
        # 2. LSTM Block
        lstm_hidden = config.get('lstm_hidden', 64)
        lstm_layers = config.get('lstm_layers', 1)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        
        # 3. CNN Block (1D Convolution over the time axis of LSTM outputs)
        cnn_channels = config.get('cnn_channels', 32)
        cnn_kernel = config.get('cnn_kernel', 3)
        self.conv1d = nn.Conv1d(
            in_channels=lstm_hidden, 
            out_channels=cnn_channels, 
            kernel_size=cnn_kernel
        )
        self.relu = nn.ReLU()
        # Adaptive pooling forces the output to be 1 value per channel, regardless of sequence length
        self.pool = nn.AdaptiveAvgPool1d(1) 
        
        # 4. MLP Block
        mlp_hidden_layers = config.get('mlp_hidden', [64, 32])
        mlp_layers = []
        
        in_features = cnn_channels
        for hidden_size in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(in_features, hidden_size))
            mlp_layers.append(nn.ReLU())
            in_features = hidden_size
            
        # Final output layer predicting the future feature vector
        mlp_layers.append(nn.Linear(in_features, self.output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, stock_id: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Process Stock Identifier
        if self.use_embedding:
            id_features = self.embedding(stock_id) # (batch, emb_dim)
        else:
            id_features = F.one_hot(stock_id, num_classes=self.id_dim).float() # (batch, id_dim)
            
        # Expand ID to match sequence length: (batch, seq_len, id_dim)
        id_features = id_features.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate raw features with stock ID
        combined_x = torch.cat([x, id_features], dim=-1)
        
        # LSTM Pass
        lstm_out, _ = self.lstm(combined_x) # (batch, seq_len, lstm_hidden)
        
        # CNN Pass (PyTorch Conv1d expects [batch, channels, seq_len])
        cnn_in = lstm_out.transpose(1, 2)
        cnn_out = self.conv1d(cnn_in)
        cnn_out = self.relu(cnn_out)
        
        # Pool across remaining time dimension & flatten
        pooled = self.pool(cnn_out).squeeze(-1) # (batch, cnn_channels)
        
        # MLP Pass
        predictions = self.mlp(pooled) # (batch, output_dim)
        
        return predictions