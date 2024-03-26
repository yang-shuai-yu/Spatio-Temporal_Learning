import _pickle as cPickle
import os, sys, time
import csv, random, math
import numpy as np
import torch

porto_lat_range = [40.953673,41.307945]
porto_lon_range = [-8.735152,-8.156309]

def prepare_data(data_path, data_name, emb_dim, train_ratio, valid_ratio, test_ratio):
    '''
      return: train_data, attack_data, test_data
        train_data: a set, {train_embedding, train_trajectory}
        attack_data: a set, {attack_embedding, attack_trajectory}
        test_data: a set, {test_embedding, test_trajectory}
    '''
    # data_path = '/home/yangshuaiyu6791/RepresentAttack/myAttack2/r_data/'

    # if the data is exist, load it
    print(" Loading data...")
    if os.path.exists('./data/{}/{}/train/train_data'.format(data_name, emb_dim)):
        with open('./data/{}/{}/train/train_data'.format(data_name, emb_dim), 'rb') as f:
            train_data = cPickle.load(f)
        with open('./data/{}/{}/test/test_data'.format(data_name, emb_dim), 'rb') as f:
            test_data = cPickle.load(f)
        with open('./data/{}/{}/val/val_data'.format(data_name, emb_dim), 'rb') as f:
            val_data = cPickle.load(f)
        return train_data, val_data, test_data

    # load trajectory data
    print("Generating data...")
    with open(data_path + 'gps_seqs/traingps', 'rb') as f:
        rtrain_trajs = cPickle.load(f)
    with open(data_path + 'gps_seqs/testgps', 'rb') as f:
        rtest_trajs = cPickle.load(f)
    with open(data_path + 'gps_seqs/valgps', 'rb') as f:
        rval_trajs = cPickle.load(f)

    # load embedding data
    with open(data_path + 'embeddings/{}/{}/{}_train'.format(data_name, emb_dim, data_name), 'rb') as f:
        rtrain_embs = cPickle.load(f)
    with open(data_path + 'embeddings/{}/{}/{}_test'.format(data_name, emb_dim, data_name), 'rb') as f:
        rtest_embs = cPickle.load(f)
    with open(data_path + 'embeddings/{}/{}/{}_val'.format(data_name, emb_dim, data_name), 'rb') as f:
        rval_embs = cPickle.load(f)

    # for train, test, attack, only use the first 1000 trajectories
    train_trajs = rtrain_trajs[:1000]; train_embs = rtrain_embs[:1000]
    test_trajs = rtest_trajs[:1000]; test_embs = rtest_embs[:1000]
    val_trajs = rval_trajs[:1000]; val_embs = rval_embs[:1000]

    train_data = {'embs': train_embs, 'trajs': train_trajs}
    test_data = {'embs': test_embs, 'trajs': test_trajs}
    val_data = {'embs': val_embs, 'trajs': val_trajs}

    # store data if the data is not exist
    if not os.path.exists('./data/{}/'.format(data_name)):
        os.makedirs('./data/{}/'.format(data_name));os.makedirs('./data/{}/{}/'.format(data_name, emb_dim))
        os.makedirs('./data/{}/{}/train/'.format(data_name, emb_dim));os.makedirs('./data/{}/{}/test/'.format(data_name, emb_dim))
        os.makedirs('./data/{}/{}/val/'.format(data_name, emb_dim))
        with open('./data/{}/{}/train/train_data'.format(data_name, emb_dim), 'wb') as f:
            cPickle.dump(train_data, f)
        with open('./data/{}/{}/test/test_data'.format(data_name, emb_dim), 'wb') as f:
            cPickle.dump(test_data, f)
        with open('./data/{}/{}/val/val_data'.format(data_name, emb_dim), 'wb') as f:
            cPickle.dump(val_data, f)

    return train_data, val_data, test_data

class Traj2Cell(object):
    def __init__(self, delta = 0.005, lat_range = porto_lat_range, lon_range = porto_lon_range):
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.__init_grid_hash_function()

    def __init_grid_hash_function(self):
        dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
        x  = self._frange(dXMin,dXMax, self.delta)
        y  = self._frange(dYMin,dYMax, self.delta)
        self.x = x
        self.y = y
    
    def _frange(self, start, end=None, inc=None):
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
        L = []
        while 1:
            next = start + len(L) * inc
            if inc > 0 and next >= end:
                break
            elif inc < 0 and next <= end:
                break
            L.append(next)
        return L

    def point2XYandCell(self, point):
        '''
          point: [lon, lat]
          return: [x, y, cell]
        '''
        x = int((point[0] - self.lon_range[0])/self.delta)
        y = int((point[1] - self.lat_range[0])/self.delta)
        cell = x + y * int((self.lon_range[1] - self.lon_range[0])/self.delta)
        return [x, y, cell]

    def traj2grid_seq(self, traj, isCoordinate = False):
        '''
          traj: a list of [len, lat, lon]
          return: a list of [x, y, cell]
        '''
        # I have no isCoordinate
        # if isCoordinate:
        #     traj = self.delDup(traj, isCoordinate = isCoordinate)
        traj_grids = []
        for point in traj:
            traj_grids.append(self.point2XYandCell(point))
        return traj_grids

    def preprocess(self, traj_index, traj, isCoordinate = False):
        # first, I have no isCoordinate
        # traj_index: {i: [0, lat, lon]...}
        # For the first time, I only get cells for trajecotries in data, not in r_data
        if traj_index%100 == 0:
            print("Preprocessing the {}th trajectory...".format(traj_index))
        traj_grids = self.traj2grid_seq(traj, isCoordinate = isCoordinate)
        return traj_grids
    
def get_CellMap(data_path):
    traj2cell = Traj2Cell()
    sub_path = ['train', 'test', 'val']
    for path in sub_path:
        print("Preprocessing {} data...".format(path))
        grid_seqs = []
        with open(data_path + '{}/{}_data'.format(path, path), 'rb') as f:
            data_set = cPickle.load(f)
        trajs = data_set['trajs']
        for i, traj in enumerate(trajs):
            traj_grids = traj2cell.preprocess(i, traj)    # a list of [x, y, cell]
            grid_seqs.append(traj_grids)
        print("Saving {} data...".format(path))
        with open(data_path + '{}/{}_grid'.format(path, path), 'wb') as f:
            cPickle.dump(grid_seqs, f)


if __name__ == '__main__':
    data_path = '/home/yangshuaiyu6791/RepresentAttack/myAttack2/r_data/'
    data_name = 'neutraj'
    emb_dim = 128
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    train_data, val_data, test_data = prepare_data(data_path, data_name, emb_dim, train_ratio, valid_ratio, test_ratio)
    print('train_data: ', type(train_data), train_data.keys())
    print('test_data: ', type(test_data), test_data.keys())
    print('val_data: ', type(val_data), val_data.keys())
    path = './data/{}/{}/'.format(data_name, emb_dim)
    get_CellMap(data_path = path)