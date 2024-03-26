import torch
import torch.nn as nn

class StackingMLP(nn.Module):
    """
    Multi-layer Percetions
    """
    def __init__(self, input_size, hidden_size, num_clusters, num_layers):
        super(StackingMLP, self).__init__()
        self.num_layers = num_layers
        self.mlps = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_clusters),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Input:
        input (batch, input_size): input embedding vectors
        ---
        Output:
        output (batch, num_clusters): the final layer output vectors
        """
        
        return self.mlps(input)