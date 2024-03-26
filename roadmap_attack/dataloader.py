import torch
import numpy as np
import pickle, os
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    '''
        Dataaset for roadmap attack.
        Return: embedding, one hot vector of road segment.
    '''
    def __init__(self, aligned_path, emb_path, data_name, emb_name, mode = 'train'):
        self.aligned_path = aligned_path    # align_data/aligned_gps
        self.emb_path = emb_path    # data/embedding name
        self.data_name = data_name
        self.emb_name = emb_name
        self.mode = mode
        self.aligned_data = None
        self.roadseg_num = 11095
        self.load_data()

    def __len__(self):
        return len(self.aligned_data)
    
    def load_data(self):
        # Load aligned gps data
        with open(os.path.join(self.aligned_path, self.data_name), 'rb') as f:
            self.aligned_data = pickle.load(f)
        # Load embedding data
        with open(os.path.join(self.emb_path, self.emb_name), 'rb') as f:
            self.embedding = pickle.load(f)
        # get embedding as the lenth of the road segment
        self.embedding = self.embedding[:len(self.aligned_data)]
        return
    
    def __getitem__(self, idx):
        # Get the idx-th road segment list
        traj = self.aligned_data[idx]
        # Get the idx-th embedding
        emb = self.embedding[idx]
        emb = torch.tensor(emb)
        # deduce the road segment
        road_seg = torch.zeros(self.roadseg_num)
        road_seg[traj] = 1    # one-hot like vector
    
        return emb, road_seg