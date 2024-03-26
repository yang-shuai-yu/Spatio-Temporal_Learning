import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class sPoints_net(nn.Module):
    """
        sePoints_net: a network for starting points attack.
    """
    def __init__(self, emb_dim, hid_dim):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim//2),
            nn.Dropout(0.3),
            nn.Linear(hid_dim//2, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, emb):
        return self.net(emb)
    
class ePoints_net(nn.Module):
    """
        ePoints_net: a network for ending points attack.
    """
    def __init__(self, emb_dim, hid_dim):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim//2),
            nn.Dropout(0.3),
            nn.Linear(hid_dim//2, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, emb):
        return self.net(emb)

class MLP_net(nn.Module):
    """
        MLP_net: a network for MLP.
    """
    def __init__(self, hid_dim):
        super().__init__()

        self.hid_dim = hid_dim
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 4)
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, emb):
        return self.net(emb)
    

class se_net(nn.Module):
    """
        se_net: a fusion of all the three networks.
    """
    def __init__(self, emb_dim, hid_dim):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.s_net = sPoints_net(emb_dim, hid_dim)
        self.e_net = ePoints_net(emb_dim, hid_dim)
        self.se_net = MLP_net(hid_dim)

    def forward(self, emb):
        s_output = self.s_net(emb)
        e_output = self.e_net(emb)
        se_output = self.se_net(torch.cat((s_output, e_output), dim = -1))
        return se_output