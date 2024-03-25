import numpy as np
import random, os, pickle
import argparse

from dataloader import GridGenerator
from ldp import GenServer, GenClient

def get_args():
    parser = argparse.ArgumentParser(description='Generate grid data')
    parser.add_argument('--dataset', type=str, default='porto', help='dataset name')
    parser.add_argument('--delta', type=float, default=0.006, help='grid size')
    parser.add_argument('--epsilon', type=float, default=2.0, help='privacy budget')
    parser.add_argument('--lamda', type=float, default=20.0, help='lamda')
    parser.add_argument('--sampling_ratio', type=float, default=1.0, help='sampling ratio')
    parser.add_argument('--num', type=int, default=2000, help='number of synthetic trajectories')
    return parser.parse_args()

if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # Generate a grid map & print info
    print("Generating grid map...")
    gridGenerator = GridGenerator(args.dataset, delta=args.delta, epsilon=args.epsilon,
                                   lamda=args.lamda, sampling_ratio=args.sampling_ratio)
    # gridGenerator.generate(delta = args.delta, autogrid = False)
    # gridGenerator.print_info()
    _, test_grid, _ = gridGenerator.get_grid_data(is_load=True)
    info = gridGenerator.get_grid_info(is_load=True)

    # generate and save markov model
    print("Generating markov model...")
    genServer = GenServer(args.dataset, args.epsilon, args.delta, info)
    genServer.load_data(test_grid)
    genServer.generate()
    lenth_dist, transition_dist, start_dist, end_dist = genServer.get_dist()
    genServer.save()

    # generate and save synthetic data
    print("Generating synthetic data...")
    genClient = GenClient(args.dataset, args.epsilon, args.delta, info)
    # genClient.load_dist(lenth_dist, transition_dist, start_dist, end_dist)
    genClient.load_dist(genServer)
    genClient.trajs_synthesis(args.num)
    syn_trajs = genClient.get_syn_trajs()
    genClient.save()

    # done?
    print("Done?")

    
