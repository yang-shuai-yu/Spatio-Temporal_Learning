import _pickle as cPickle
import os, sys, time
import csv, random, math, argparse
import numpy as np
import torch

porto_lat_range = [40.953673,41.307945]
porto_lon_range = [-8.735152,-8.156309]

def prepare_data(data_path, data_name, emb_dim):
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

    # Only use the first 60000 trajectories for training, 10000 for testing and 10000 for validation
    train_trajs = rtrain_trajs[:60000]; train_embs = rtrain_embs[:60000]
    test_trajs = rtest_trajs[:10000]; test_embs = rtest_embs[:10000]
    val_trajs = rval_trajs[:10000]; val_embs = rval_embs[:10000]

    # store data
    # data structure: {'embs': embs, 'trajs': trajs}
    train_data = {'embs': train_embs, 'trajs': train_trajs}
    test_data = {'embs': test_embs, 'trajs': test_trajs}
    val_data = {'embs': val_embs, 'trajs': val_trajs}

    # store data if the data is not exist
    if not os.path.exists('./data/{}/{}/'.format(data_name, emb_dim)):
        os.makedirs('./data/{}/'.format(data_name), exist_ok= True);os.makedirs('./data/{}/{}/'.format(data_name, emb_dim))
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
    def __init__(self, delta = 0.001, lat_range = porto_lat_range, lon_range = porto_lon_range):
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
          traj: a list of [len, lon, lat]
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
    
    def get_cell_num(self):
        x_num = int((self.lon_range[1] - self.lon_range[0])/self.delta)
        y_num = int((self.lat_range[1] - self.lat_range[0])/self.delta)
        num_list = [x_num, y_num, x_num*y_num]
        return num_list
    
def get_CellMap(data_path):
    ''' get cell map for train, test and val data set '''
    traj2cell = Traj2Cell()
    sub_path = ['train', 'test', 'val']
    log_list = []
    for path in sub_path:
        print("Preprocessing {} data...".format(path))
        grid_seqs = []
        with open(data_path + '{}/{}_data'.format(path, path), 'rb') as f:
            data_set = cPickle.load(f)
        trajs = data_set['trajs']
        lenth = len(trajs)
        for i, traj in enumerate(trajs):
            traj_grids = traj2cell.preprocess(i, traj)    # a list of [x, y, cell]
            grid_seqs.append(traj_grids)
        print("Saving {} data...".format(path))
        with open(data_path + '{}/{}_grid'.format(path, path), 'wb') as f:
            cPickle.dump(grid_seqs, f)
        print("Saving {} data finished!".format(path))
        log_list.append([path, lenth])

    cells_num = traj2cell.get_cell_num()

    with open(data_path + 'readme.txt', 'w') as f:
        f.write("Data: {}\n".format(data_path))
        f.write("Data size: {}\n".format(log_list))
        f.write("Trajectory to cell map: {}\n".format(data_path + 'train/train_grid'))
        f.write("Trajectory to cell map: {}\n".format(data_path + 'test/test_grid'))
        f.write("Trajectory to cell map: {}\n".format(data_path + 'val/val_grid'))
        f.write("Cell num: {}\n".format(cells_num))
        f.close()
    print("Preprocessing finished!")
        



if __name__ == '__main__':
    os.chdir("/home/yangshuaiyu6791/RepresentAttack/inversion_sePoint/")    # change the current directory to inversion_sePoint
    print("Current working directory: ", os.getcwd())
    data_path = './rdata/'
    data_name = 't2vec'
    emb_dim = 256
    train_data, val_data, test_data = prepare_data(data_path, data_name, emb_dim)
    print('train_data: ', type(train_data), train_data.keys())
    print('test_data: ', type(test_data), test_data.keys())
    print('val_data: ', type(val_data), val_data.keys())

    # get cell map for train, test and val data set
    path = './data/{}/{}/'.format(data_name, emb_dim)
    get_CellMap(data_path = path)