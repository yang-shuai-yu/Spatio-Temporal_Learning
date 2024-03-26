import numpy as np
import os, pickle
import torch
from torch.utils.data import Dataset, DataLoader

class DistDataset(Dataset):
    def __init__(self, dist_path, emb_path):
        self.dist_path = dist_path
        self.emb_path = emb_path
        self.dist_data = None
        self.norm_data = None
        self.emb_data = None
        self.mean = None
        self.std = None
        self.load_data()
        self.normalize()    # normalize the distance data
    def __len__(self):
        return len(self.emb_data)
    def load_data(self):
        self.dist_data = pickle.load(open(self.dist_path, 'rb'))
        self.emb_data = pickle.load(open(self.emb_path, 'rb'))

        samples = 100_000
        if len(self.emb_data) > samples:    # only get 100k samples if train
            # get 100k samples if train 
            sample_idx = np.random.choice(len(self.emb_data), samples, replace=False)
            # print(sample_idx)
            self.dist_data = np.array(self.dist_data)[sample_idx]
            self.emb_data = np.array(self.emb_data)[sample_idx]

        self.dist_data = torch.tensor(self.dist_data).unsqueeze(1)
        self.emb_data = torch.tensor(self.emb_data)
        self.mean = self.dist_data.mean()
        self.std = self.dist_data.std()
        print(self.dist_data.shape, self.emb_data.shape, self.mean, self.std)
    def normalize(self):
        if self.mean is None or self.std is None:
            raise ValueError("mean or std is None")
        self.norm_data = (self.dist_data - self.mean) / self.std
    def __getitem__(self, idx):
        if self.norm_data is None or self.emb_data is None:
            raise ValueError("data is None")
        return self.emb_data[idx], self.norm_data[idx]
