import numpy as np
import os, pickle
import torch
from torch.utils.data import Dataset, DataLoader

class DistDataset(Dataset):
    def __init__(self, circle_path, emb_path):
        self.circle_path = circle_path
        self.emb_path = emb_path
        self.circle_data = None
        self.center_data = None
        self.emb_data = None

        self.norm_centerX = None
        self.norm_centerY = None
        self.norm_radius = None
        self.mean_centerX = None
        self.mean_centerY = None
        self.mean_radius = None
        self.std_radius = None
        self.std_centerX = None
        self.std_centerY = None
        self.center_data_x = None
        self.center_data_y = None
        self.load_data()
        self.normalize()    # normalize the distance data

    def __len__(self):
        return len(self.emb_data)
    def load_data(self):
        self.circle_data = np.array(pickle.load(open(self.circle_path, 'rb')))
        self.emb_data = pickle.load(open(self.emb_path, 'rb'))
        self.center_data = self.circle_data[:, 0]
        self.radius_data = self.circle_data[:, 1]
        # get 100k samples if train 
        # sample_idx = np.random.choice(len(self.emb_data), 100000, replace=False)
        # print(sample_idx)
        # self.center_data = np.array(self.center_data)[sample_idx]
        # self.emb_data = np.array(self.emb_data)[sample_idx]

        self.center_data = np.array([np.array(x) for x in self.center_data])
        self.center_data_x = torch.tensor(np.array(self.center_data[:,0]).astype(np.float64)).unsqueeze(1)
        self.center_data_y = torch.tensor(np.array(self.center_data[:,1]).astype(np.float64)).unsqueeze(1)
        self.radius_data = torch.tensor(self.radius_data.astype(np.float64)).unsqueeze(1)
        self.emb_data = torch.tensor(self.emb_data)
        self.mean_centerX = self.center_data_x.mean()
        self.mean_centerY = self.center_data_y.mean()
        self.std_centerX = self.center_data_x.std()
        self.std_centerY = self.center_data_y.std()
        self.mean_radius = self.radius_data.mean()
        self.std_radius = self.radius_data.std()
        print("mean_centerX: ", self.mean_centerX, "mean_centerY: ", self.mean_centerY, "mean_radius: ",
               self.mean_radius, "std_radius: ", self.std_radius)
        
    def normalize(self):
        if self.mean_radius is None or self.std_radius is None:
            raise ValueError("mean or std is None")
        self.norm_centerX = (self.center_data_x - self.mean_centerX) / self.std_centerX
        self.norm_centerY = (self.center_data_y - self.mean_centerY) / self.std_centerY
        self.norm_radius = (self.radius_data - self.mean_radius) / self.std_radius

    def __getitem__(self, idx):
        if self.norm_radius is None or self.emb_data is None:
            raise ValueError("data is None")
        circle_info = torch.cat([self.norm_centerX[idx], self.norm_centerY[idx], self.norm_radius[idx]], dim=0)
        return self.emb_data[idx], circle_info    # data, label
