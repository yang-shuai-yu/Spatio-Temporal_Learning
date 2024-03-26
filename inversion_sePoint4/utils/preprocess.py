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

    # Only use the first 10000 trajectories for training, 10000 for testing and 10000 for validation
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
        
def get_Normalization(data_path):
    ''' get normalization for train, test and val data set '''
    sub_path = ['train', 'test', 'val']
    log_list = []
    for path in sub_path:
        norm_seqs = []
        print("Preprocessing {} data...".format(path))
        with open(data_path + '{}/{}_data'.format(path, path), 'rb') as f:
            data = cPickle.load(f)
        grid_seqs = data['trajs']    # len, lon, lat
        lenth = len(grid_seqs)
        for i, traj in enumerate(grid_seqs):    # len, lon, lat
            traj = np.array(traj)
            porto_lon_mid = (porto_lon_range[0] + porto_lon_range[1])/2
            porto_lat_mid = (porto_lat_range[0] + porto_lat_range[1])/2
            traj[:, 0] = (traj[:, 0] - porto_lon_mid)/((porto_lon_range[1] - porto_lon_range[0])/2)
            traj[:, 1] = (traj[:, 1] - porto_lat_mid)/((porto_lat_range[1] - porto_lat_range[0])/2)
            norm_seqs.append(traj)
        # save data
        print("Saving {} data...".format(path))
        with open(data_path + '{}/{}_norm'.format(path, path), 'wb') as f:
            cPickle.dump(norm_seqs, f)
        print("Saving {} data finished!".format(path))

    with open(data_path + 'readme.txt', 'a') as f:
        f.write("Normalization: {}\n".format(data_path + 'train/train_norm'))
        f.write("Normalization: {}\n".format(data_path + 'test/test_norm'))
        f.write("Normalization: {}\n".format(data_path + 'val/val_norm'))
        f.close()
    print("Preprocessing finished!")

# compute the g



if __name__ == '__main__':
    # 做归一化，不要搞grid了
    os.chdir("/home/yangshuaiyu6791/RepresentAttack/inversion_sePoint4/")    # change the current directory to inversion_sePoint
    print("Current working directory: ", os.getcwd())
    data_path = './rdata/'
    data_name = 'start'
    emb_dim = 256
    train_data, val_data, test_data = prepare_data(data_path, data_name, emb_dim)
    print('train_data: ', type(train_data), train_data.keys())
    print('test_data: ', type(test_data), test_data.keys())
    print('val_data: ', type(val_data), val_data.keys())

    # # get cell map for train, test and val data set
    # 对于轨迹点进行归一化即可
    path = './data/{}/{}/'.format(data_name, emb_dim)
    get_Normalization(data_path = path)