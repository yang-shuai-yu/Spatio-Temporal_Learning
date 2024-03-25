import numpy as np
import matplotlib.pyplot as plt
import pickle, os, argparse
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import ast

from utils import * 

class Calculator():
    def __init__(self, plot, segment_num):
        self.gps_data = None
        self.aligned_data = None
        self.decup_aligned_data = None
        self.roadmap_file = None
        self.plot = plot
        self.segment_num = segment_num
        self.adj_matrix = None
        self.segment_freq = None
        self.lenth_array = None

    def load_data(self, gps_path, aligned_path, roadmap_path):
        geo_file_path = 'porto_roadmap_edge.geo'
        self.gps_data = pickle.load(open(gps_path, 'rb'))
        self.aligned_data = pickle.load(open(aligned_path, 'rb'))
        self.roadmap_file = load_geo_file(os.path.join(roadmap_path, geo_file_path))
        return 
    
    def calculate_adj_matrix(self):
        # calculate the adjacent matrix using the roadmap
        # use nest set to store the adjacent matrix
        adj_matrix = [dict() for _ in range(self.segment_num)]     # set list

        # decup the aligned data
        decup_data = []
        for i in range(len(self.aligned_data)):
            decup_data.append(list(set(self.aligned_data[i])))
        self.decup_aligned_data = decup_data    # decup 

        # calculate the adjacent matrix
        tmp_data = self.decup_aligned_data    # for simplicity
        print(len(tmp_data))
        for i in range(len(tmp_data)):
            for j in range(len(tmp_data[i]) - 1):
                if tmp_data[i][j] not in adj_matrix[tmp_data[i][j + 1]]:
                    adj_matrix[tmp_data[i][j + 1]][tmp_data[i][j]] = 1
                else:
                    adj_matrix[tmp_data[i][j + 1]][tmp_data[i][j]] += 1

                if tmp_data[i][j + 1] not in adj_matrix[tmp_data[i][j]]:
                    adj_matrix[tmp_data[i][j]][tmp_data[i][j + 1]] = 1
                else:
                    adj_matrix[tmp_data[i][j]][tmp_data[i][j + 1]] += 1
        self.adj_matrix = adj_matrix

    def calculate_segment_freq(self):
        # calculate the frequency of each segment
        segment_freq = [0 for _ in range(self.segment_num)]
        for i in range(len(self.decup_aligned_data)):
            for j in range(len(self.decup_aligned_data[i])):
                segment_freq[self.decup_aligned_data[i][j]] += 1
        segment_freq = np.array(segment_freq) / np.sum(segment_freq)
        self.segment_freq = segment_freq

    def calculate_length(self):
        # calculate the length of each segment
        max_length = 0
        for i in range(len(self.decup_aligned_data)):
            length = len(self.decup_aligned_data[i])
            # print(length)
            if length > max_length:
                max_length = length
        
        self.lenth_array = np.zeros(max_length+1, dtype=np.float32)
        for i in range(len(self.decup_aligned_data)):
            self.lenth_array[len(self.decup_aligned_data[i])] += 1
        self.lenth_array = self.lenth_array / np.sum(self.lenth_array)
        return self.lenth_array

    def calculate_all(self):
        self.calculate_adj_matrix()
        self.calculate_segment_freq()
        self.calculate_length()
        return self.adj_matrix, self.segment_freq, self.lenth_array

    def get_adj_matrix(self):
        if self.adj_matrix is None:
            print("Please calculate the adjacent matrix first!")
            return None
        return self.adj_matrix
    
    def get_segment_freq(self):
        if self.segment_freq is None:
            print("Please calculate the segment frequency first!")
            return None
        return self.segment_freq
    
    def get_segment_length(self):
        if self.lenth_array is None:
            print("Please calculate the segment length first!")
            return None
        return self.lenth_array
                    
class Server():
    def __init__(self, roadmap_path, output_path, segment_num, plot):
        self.roadmap_path = roadmap_path
        self.segment_num = segment_num
        self.output_path = output_path
        self.plot = plot
        self.adj_matrix = None
        self.segment_freq = None
        self.lenth_array = None
        self.gen_trajs = None

    def load(self, adj_matrix, segment_freq, lenth_array):
        geo_file_path = 'porto_roadmap_edge.geo'
        self.roadmap_file = load_geo_file(os.path.join(self.roadmap_path, geo_file_path))
        self.adj_matrix = adj_matrix
        self.segment_freq = segment_freq
        self.lenth_array = lenth_array
        return
    
    def generate_id(self):
        # generate one trajectory
        traj_id = []
        # select the length of the trajectory
        length = np.random.choice(len(self.lenth_array), p=self.lenth_array)
        # select the start point
        start = np.random.choice(self.segment_num, p=self.segment_freq)
        traj_id.append(start)
        # iterate to generate the trajectory
        for i in range(length - 1):
            nexts = list(self.adj_matrix[start].keys())
            probs = list(self.adj_matrix[start].values())
            start = np.random.choice(nexts, p=np.array(probs) / np.sum(probs))
            traj_id.append(start)
        return traj_id
    
    def generate_one_traj(self, segment_ids):
        # from segment ids to trajectory
        traj = []
        # print(segment_ids)
        for i in range(len(segment_ids)):
            traj_str = self.roadmap_file.loc[segment_ids[i]]['coordinates']
            # print(traj_str)
            traj_seg = np.array(ast.literal_eval(traj_str))
            # print(traj_seg)
            traj.append(traj_seg)
        traj = np.concatenate(traj, axis=0)
        return traj
        
    def trajs_generate(self, num):
        # generate the trajectories
        trajs = []
        for i in range(num):
            segment_ids = self.generate_id()
            traj = self.generate_one_traj(segment_ids)
            trajs.append(traj)
        self.gen_trajs = trajs
        return trajs

    def save_trajs(self):
        # save the generated trajectories
        if self.gen_trajs is None:
            print("Please generate the trajectories first!")
            return
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.gen_trajs, f)
            f.close()
        return


