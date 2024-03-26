import torch 
import os, sys, pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    """
        Dataset for starting, ending points attack.
        Return: embedding, starting cell, ending cell.
    """
    def __init__(self, data_path, data_name, emb_dim, hid_dim, mode = 'train'):
        self.data_path = data_path
        self.data_name = data_name
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.mode = mode
        self.load_data()

    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        path = os.path.join(self.data_path, self.data_name, str(self.emb_dim))
        self.data = None
        if self.mode == 'train':
            emb_path = os.path.join(path, 'train', 'train_data')
            grid_path = os.path.join(path, 'train', 'train_grid')
        elif self.mode == 'test':
            emb_path = os.path.join(path, 'test', 'test_data')
            grid_path = os.path.join(path, 'test', 'test_grid')
        elif self.mode == 'val':
            emb_path = os.path.join(path, 'val', 'val_data')
            grid_path = os.path.join(path, 'val', 'val_grid')
        else:
            raise ValueError("mode must be train, test or val!")
        
        with open(emb_path, 'rb') as f:
            data = pickle.load(f)
            self.data = torch.from_numpy(np.array(data['embs'])).float()
        with open(grid_path, 'rb') as f:
            self.grid = pickle.load(f)

        return

    def __getitem__(self, idx):
        emb = self.data[idx]
        grid_len = len(self.grid[idx])
        start_cell = [self.grid[idx][0][0], self.grid[idx][0][1]]    # because every grid is [x, y, cell]
        end_cell = [self.grid[idx][grid_len-1][0], self.grid[idx][grid_len-1][1]]
        se_cells = torch.Tensor([start_cell, end_cell]).float()    # float for loss bachward
        return emb, se_cells