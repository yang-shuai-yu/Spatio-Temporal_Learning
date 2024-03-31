import torch
import torch.nn as nn
import numpy as np
import os, math, time, argparse
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class TrajDataset(Dataset):
    '''
        Dataset: embeddings
        Target: grid sequence with max length 20
    '''
    def __init__(self, emb_path, grid_path, emb_model, emb_dim, mode = 'train', max_len = 20):
        self.emb_path = emb_path
        self.grid_path = grid_path
        self.emb_model = emb_model
        self.emb_dim = emb_dim
        self.mode = mode
        self.grid_max_len = max_len
        self.emb_data = None
        self.grid_data = None
        self.emb_patch = None

        self.load_data()
        self.get_emb_patch()

        print("data shape: ", self.emb_data.shape, self.grid_data.shape)

    def __len__(self):
        return len(self.grid_data)
    
    def load_data(self):
        ### load embeddings
        emb_name = '%s_%d_%s' % (self.emb_model, self.emb_dim, self.mode)
        emb_path = os.path.join(self.emb_path, emb_name)
        with open(emb_path, 'rb') as f:
            data = pickle.load(f)
            self.emb_data = np.array(data)

        ### load grid data
        grid_path  = '%s%s%s' % (self.grid_path, self.mode,'.trg')
        grid_stream = open(grid_path, 'r')
        grid_data = []
        for grid in grid_stream:
            grid = np.asarray([int(x) for x in grid.split()])
            grid_data.append(grid)
        grid_data = np.array(grid_data)
        self.grid_data = grid_data
        grid_stream.close()

        return
    
    def get_emb_patch(self):
        # size: (num, emb_dim) -> (num, max_len, emb_dim)
        if self.emb_data is None:
            raise ValueError('No data loaded!')
        emb_patch = np.expand_dims(self.emb_data, axis = 1).repeat(self.grid_max_len, axis = 1)
        self.emb_patch = emb_patch
        return
    
    def __getitem__(self, idx):
        if self.grid_data is None:
            raise ValueError('No data loaded!')
        emb = self.emb_patch[idx]
        grid = self.grid_data[idx]
        return emb, grid    # emb: (max_len, emb_dim), grid: (max_len)
        
