import torch
import torch.nn as nn

class StackingMLP(nn.Module):
    """
    Multi-layer Percetions
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(StackingMLP, self).__init__()
        self.num_layers = num_layers
        self.mlps = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input):
        """
        Input:
        input (batch, input_size): input embedding vectors
        ---
        Output:
        output (batch, 1): the final layer output prediction of length
        """
        
        return self.mlps(input)