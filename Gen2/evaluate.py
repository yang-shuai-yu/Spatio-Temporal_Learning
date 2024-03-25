import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import ast
import random, os, pickle, time, argparse
import scipy.stats as stats

from utils import * 

max_loc = [-8.156309, 41.307945]   # Porto, longitude, latitude
min_loc = [-8.735152, 40.953673]
max_len = 100

def JS_divergence(p, q):
    # p, q are two distribution
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)

def load_data(syn_path, real_path):
    # load the generated and real trajectories
    syn_trajs = pickle.load(open(syn_path, 'rb'))
    real_trajs = pickle.load(open(real_path, 'rb'))
    return syn_trajs, real_trajs

def clip_traj(traj):
    # clip the trajectory into the range
    for i in range(len(traj)):
        traj[i] = np.clip(traj[i], min_loc, max_loc)
    return traj

def get_gridid(point, grid_size=16, epsilon=1e-6):
    grid_size = grid_size - epsilon
    # get the grid id of each point
    lon = int((point[0] - min_loc[0]) / (max_loc[0] - min_loc[0]) * grid_size)
    lat = int((point[1] - min_loc[1]) / (max_loc[1] - min_loc[1]) * grid_size)
    return lon, lat

def density_error(gen_trajs, real_trajs):
    # calculate the density error
    # divide the lon-lat into 16*16 grids
    # calculate the density of each grid for generated and real trajectories
    # calculate the JS divergence between the two distributions

    # first, alreadly clipped

    # second, calculate the density of each grid
    gen_density = np.zeros((16, 16))
    real_density = np.zeros((16, 16))
    for traj in gen_trajs:
        for point in traj:
            lon, lat = get_gridid(point)
            gen_density[lon, lat] += 1
    for traj in real_trajs:
        for point in traj:
            lon, lat = get_gridid(point)
            real_density[lon, lat] += 1
    # second, calculate the JS divergence
    gen_density = gen_density / np.sum(gen_density)
    real_density = real_density / np.sum(real_density)
    js_div = JS_divergence(gen_density.flatten(), real_density.flatten())
    return js_div

def trip_error(gen_trajs, real_trajs):
    # calculate the trip error
    # get the start point and end point of each trajectory
    gen_start, gen_end = [], []
    real_start, real_end = [], []
    for traj in gen_trajs:
        if len(traj) < 2:
            continue
        gen_start.append(traj[0])
        gen_end.append(traj[-1])
    for traj in real_trajs:
        if len(traj) < 2:
            continue
        real_start.append(traj[0])
        real_end.append(traj[-1])

    # calculate the distribution of start and end points
    gen_start_density = np.zeros((16, 16))
    gen_end_density = np.zeros((16, 16))
    real_start_density = np.zeros((16, 16))
    real_end_density = np.zeros((16, 16))

    for point in gen_start:
        lon, lat = get_gridid(point)
        gen_start_density[lon, lat] += 1
    for point in gen_end:
        lon, lat = get_gridid(point)
        gen_end_density[lon, lat] += 1
    for point in real_start:
        lon, lat = get_gridid(point)
        real_start_density[lon, lat] += 1
    for point in real_end:
        lon, lat = get_gridid(point)
        real_end_density[lon, lat] += 1

    # calculate the JS divergence
    gen_start_density = gen_start_density / np.sum(gen_start_density)
    gen_end_density = gen_end_density / np.sum(gen_end_density)
    real_start_density = real_start_density / np.sum(real_start_density)
    real_end_density = real_end_density / np.sum(real_end_density)
    js_div_start = JS_divergence(gen_start_density.flatten(), real_start_density.flatten())
    js_div_end = JS_divergence(gen_end_density.flatten(), real_end_density.flatten())
    return np.mean([js_div_start, js_div_end])

def lenth_error(gen_trajs, real_trajs, num = 50):
    # calculate the length error
    # calculate the length of each trajectory
    gen_len = np.zeros(num+1)
    real_len = np.zeros(num+1)
    for i, traj in enumerate(gen_trajs):
        index  = int(len(traj)/max_len * num)
        gen_len[index] += 1
    for i, traj in enumerate(real_trajs):
        index  = int(len(traj)/max_len * num)
        real_len[index] += 1
    # calculate the JS divergence
    gen_len = gen_len / np.sum(gen_len)
    real_len = real_len / np.sum(real_len)
    js_div = JS_divergence(gen_len, real_len)
    return js_div

def pattern_score(gen_trajs, real_trajs, num = 16):
    # calculate the pattern score
    # calulate the density of each grid 
    gen_density = np.zeros((num, num))
    real_density = np.zeros((num, num))
    for traj in gen_trajs:
        for point in traj:
            lon, lat = get_gridid(point, grid_size=num)
            gen_density[lon, lat] += 1
    for traj in real_trajs:
        for point in traj:
            lon, lat = get_gridid(point, grid_size=num)
            real_density[lon, lat] += 1
    # second, get the top-k grid
    k = 10
    gen_density = gen_density.flatten()
    real_density = real_density.flatten()
    gen_density = np.argsort(gen_density)[-k:]
    real_density = np.argsort(real_density)[-k:]
    # calculate the pattern score
    precision = len(set(gen_density) & set(real_density)) / k
    recall = len(set(gen_density) & set(real_density)) / k
    score = 2 * precision * recall / (precision + recall)
    return score

if __name__ == '__main__':
    syn_path = './result/gen_valgps'
    real_path = './data/gps/valgps'
    syn_trajs, real_trajs = load_data(syn_path, real_path)
    syn_trajs = clip_traj(syn_trajs)
    real_trajs = clip_traj(real_trajs)

    # calculate the density error
    density_error = density_error(syn_trajs, real_trajs)
    print('Density Error:', density_error)

    # calculate the trip error
    trip_error = trip_error(syn_trajs, real_trajs)
    print('Trip Error:', trip_error)

    # calculate the length error
    # get the max length of the real and generated trajectories
    max_len = max([len(traj) for traj in syn_trajs] + [len(traj) for traj in real_trajs])
    print('Max Length:', max_len)
    length_error = lenth_error(syn_trajs, real_trajs, num = max_len)
    print('Length Error:', length_error)

    # calculate the pattern score
    pattern_score = pattern_score(syn_trajs, real_trajs)
    print('Pattern Score:', pattern_score)