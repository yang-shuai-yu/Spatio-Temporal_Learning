''' evaluation script '''
import os, sys, random, time
import numpy as np
import scipy.stats as stats

max_loc = [-8.555, 41.186]   # Porto, longitude, latitude
min_loc = [-8.690, 41.139]
max_len = 1026

def JS_divergence(p, q):
    # p, q are two distribution
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)

def load_data(gen_path, real_path):
    # load the generated and real trajectories
    gen_trajs = np.load(gen_path, allow_pickle=True)
    real_trajs = np.load(real_path, allow_pickle=True)
    return gen_trajs, real_trajs

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
        gen_start.append(traj[0])
        gen_end.append(traj[-1])
    for traj in real_trajs:
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
    base_path = os.path.dirname(os.path.abspath(__file__))    # base path, /home/.../DiffTraj
    gen_path = os.path.join(base_path, 
                            'DiffTraj/Porto_steps=500_len=400_0.05_bs=32/test_trajs.npy')
    real_path = os.path.join(base_path,
                             'mydata/gen_data/detoured_test_gps.npy')
    gen_trajs, real_trajs = load_data(gen_path, real_path)    # both (num, length, 2), length=400
    gen_trajs = clip_traj(gen_trajs); real_trajs = clip_traj(real_trajs)    # clip the trajectories

    # calculate the density error
    start_time = time.time()
    density_error = density_error(gen_trajs, real_trajs)
    end_time = time.time()
    print('density error: ', density_error, 'time: ', end_time - start_time)

    # calculate the trip error
    start_time = time.time()
    trip_error = trip_error(gen_trajs, real_trajs)
    end_time = time.time()
    print('trip error: ', trip_error, 'time: ', end_time - start_time)

    # calculate the length error
    start_time = time.time()
    length_error = lenth_error(gen_trajs, real_trajs)
    end_time = time.time()
    print('length error: ', length_error, 'time: ', end_time - start_time)

    # calculate the pattern score
    start_time = time.time()
    pattern_score = pattern_score(gen_trajs, real_trajs)
    end_time = time.time()
    print('pattern score: ', pattern_score, 'time: ', end_time - start_time)

    print("done?")
