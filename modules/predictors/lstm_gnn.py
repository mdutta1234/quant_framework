# modules/predictors/lstm_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_predictor import BasePredictor

class GraphConvolution(nn.Module):
    """Simple implementation of a Graph Convolutional Layer (GCN)."""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (batch_size, num_nodes, features)
        # adj: (num_nodes, num_nodes) - degree-normalized adjacency matrix
        support = torch.matmul(x, self.weight)
        # Message passing across nodes
        output = torch.matmul(adj, support)
        return output + self.bias

class LSTM_GNN(BasePredictor):
    def __init__(self, config: dict):
        super().__init__(config)
        
        # --- Hyperparameters extracted from the paper ---
        lstm_hidden = config.get('lstm_hidden', 64)
        lstm_layers = config.get('lstm_layers', 2)
        gcn_hidden_1 = config.get('gcn_hidden_1', 64)
        gcn_hidden_2 = config.get('gcn_hidden_2', 32)
        dropout_rate = config.get('dropout', 0.5) # Paper specifies 0.5 dropout
        
        self.num_nodes = config.get('num_stocks', 5) # Number of stocks in the cluster
        
        # 1. LSTM Component (Temporal Embeddings)
        self.lstm = nn.LSTM(
            input_size=self.input_dim, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # 2. GNN Component (Relational Embeddings)
        # The paper uses 2 graph convolutional layers
        self.gc1 = GraphConvolution(self.input_dim, gcn_hidden_1)
        self.gc2 = GraphConvolution(gcn_hidden_1, gcn_hidden_2)
        self.gnn_dropout = nn.Dropout(dropout_rate)
        
        # 3. Hybrid Integration (Concatenation & Dense Layers)
        # Paper specifies additional dense layers after concatenation
        combined_dim = lstm_hidden + gcn_hidden_2
        
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Final dense layer with a linear activation function for regression
        self.output_layer = nn.Linear(32, self.output_dim)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, num_nodes, seq_len, features)
        adj_matrix shape: (num_nodes, num_nodes)
        """
        batch_size, num_nodes, seq_len, features = x.size()
        
        # --- A. LSTM Forward Pass ---
        # Reshape to process all stocks through the LSTM
        x_lstm_in = x.view(batch_size * num_nodes, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x_lstm_in)
        
        # Take the embedding from the final time step
        temporal_embeddings = lstm_out[:, -1, :] 
        temporal_embeddings = self.lstm_dropout(temporal_embeddings)
        temporal_embeddings = temporal_embeddings.view(batch_size, num_nodes, -1)
        
        # --- B. GNN Forward Pass ---
        # Use the most recent day's features as initial node attributes for the GNN
        x_gnn_in = x[:, :, -1, :] 
        
        relational_embeddings = F.relu(self.gc1(x_gnn_in, adj_matrix))
        relational_embeddings = self.gnn_dropout(relational_embeddings)
        relational_embeddings = F.relu(self.gc2(relational_embeddings, adj_matrix))
        
        # --- C. Hybrid Integration ---
        # Concatenate temporal and relational embeddings
        unified_vector = torch.cat((temporal_embeddings, relational_embeddings), dim=-1)
        
        # Pass through dense layers
        out = F.relu(self.fc1(unified_vector))
        out = F.relu(self.fc2(out))
        
        # Linear activation output
        predictions = self.output_layer(out)
        
        return predictions