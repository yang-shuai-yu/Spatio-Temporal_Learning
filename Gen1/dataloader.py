import pickle, json, random, os
import numpy as np
import math

''' aattempt only use test data to generate markov model '''
class GridGenerator():
    def __init__(self, dataset, delta = 0.005, epsilon = 1.0, lamda = 10.0, 
                 sampling_ratio = 1.0):
        """
        Generate grid data for the given dataset
        """
        self.dataset = dataset
        self.train_data = None; self.test_data = None; self.val_data = None
        self.train_grid = None; self.test_grid = None; self.val_grid = None
        self.info = None    # info = {"maxlon", "minlon", "maxlat", "minlat","dataset"}
        self.delta = delta
        self.epsilon = epsilon
        self.sampling_ratio = sampling_ratio
        self.lamda = lamda
        self.x_num = None; self.y_num = None
        self.grid_num = None

    def load_data(self):
        """
        Load dataset from trajectory file
        """
        if not os.path.exists("./data/trajs/{}/traingps".format(self.dataset)):
            print("Error: No such dataset")
            exit()
        
        with open ("./data/trajs/{}/traingps".format(self.dataset), "rb") as fp:
            self.train_data = pickle.load(fp)
        with open ("./data/trajs/{}/testgps".format(self.dataset), "rb") as fp:
            self.test_data = pickle.load(fp)
        with open ("./data/trajs/{}/valgps".format(self.dataset), "rb") as fp:
            self.val_data = pickle.load(fp)
        
        with open ("./data/trajs/{}/info.txt".format(self.dataset), "r") as fp:
            self.info = json.load(fp)    # info = {"ataset", "maxlat", "minlat", "maxlon", "minlon"}

    def gridnum_calculate(self):
        """
        Calculate the grid number and delta
        """
        if self.train_data is None or self.test_data is None or self.val_data is None:
            raise Exception("Error: No dataset loaded")
        # N = lamba * math.pow(...,1/4), delta = 1/N
        # calculate the max length of the trajectories, only use test data
        len_list = []
        for traj in self.test_data:
            len_list.append(len(traj))
        max_len = max(len_list); min_len = min(len_list); mean_len = np.mean(len_list)
        print("max_len: ", max_len, "min_len: ", min_len, "mean_len: ", mean_len)
        para1 = math.exp(self.epsilon * self.sampling_ratio / mean_len)
        para2 = (len(self.test_data) * mean_len * math.pow(para1 - 1, 2)) / para1
        N = self.lamda * math.pow(para2, 1/4)
        self.delta = round((self.info["maxlon"] - self.info["minlon"])/N, 3)
        print("delta: ", self.delta, "N: ", N)
        

    def grid_compute(self, point):
        """
        Compute the grid data for the given point (lon, lat)
        """
        x = int((point[0] - self.info["minlon"]) // self.delta)
        y = int((point[1] - self.info["minlat"]) // self.delta)
        cell = y * self.x_num + x    # cell_index = y * x_num + x
        return [x, y, cell]
    
    def interpolation(self):
        if self.train_data is None or self.test_data is None or self.val_data is None:
            raise Exception("Error: No dataset loaded")
        # interpolation train_data, test_data, val_data
        traj_list = [self.train_data, self.test_data, self.val_data]
        clk = 0
        new_traj_list = []
        for trajs in traj_list:
            tmp_trajs = []
            for traj in trajs:
                clk += 1
                tmp_traj = []
                for i in range(len(traj)-1):
                    if traj[i][0] == traj[i+1][0] and traj[i][1] == traj[i+1][1]:
                        tmp_traj.append(traj[i])
                    else:    # insert intermediate points with lon_step and lat_step = 0.001
                        lon_step = 0.001; lat_step = 0.001
                        max_num = max(abs(int((traj[i+1][0] - traj[i][0]) // lon_step)), abs(int((traj[i+1][1] - traj[i][1]) // lat_step)))    # get the max step
                        lon_step = (traj[i+1][0] - traj[i][0]) / max_num
                        lat_step = (traj[i+1][1] - traj[i][1]) / max_num
                        lon_flag = 1 if traj[i+1][0] > traj[i][0] else -1
                        lat_flag = 1 if traj[i+1][1] > traj[i][1] else -1
                        tmp_traj.append(traj[i])
                        for j in range(max_num):
                            tmp_traj.append([traj[i][0] + lon_step * (j+1) * lon_flag, traj[i][1] + lat_step * (j+1) * lat_flag])
                tmp_traj.append(traj[-1])
                tmp_trajs.append(tmp_traj)
                if clk % 1000 == 0:
                    print("Interpolation: ", clk)
            new_traj_list.append(tmp_trajs)
        self.train_data = new_traj_list[0]
        self.test_data = new_traj_list[1]
        self.val_data = new_traj_list[2]
                            
    def no_adjacent_duplicate(self):
        """
        Get rid of the adjacent duplicate elements
        """
        grid_list = [self.train_grid, self.test_grid, self.val_grid]
        tmp_grid_list = []
        for grids in grid_list:
            tmp_grids = []
            for grid_traj in grids:
                tmp_grid_traj = []
                for i in range(len(grid_traj)-1):
                    if grid_traj[i] != grid_traj[i+1]:
                        tmp_grid_traj.append(grid_traj[i])
                tmp_grid_traj.append(grid_traj[-1])
                tmp_grids.append(tmp_grid_traj)
            tmp_grid_list.append(tmp_grids)
        self.train_grid, self.test_grid, self.val_grid = tmp_grid_list[0], tmp_grid_list[1], tmp_grid_list[2]
                
    def generate(self, delta = 0.001, autogrid = False):
        """
        Generate grid data
        """
        print("Generating grid data...")
        if self.train_data is None or self.test_data is None or self.val_data is None:
            self.load_data()
        self.interpolation()    # interpolation train_data, test_data, val_data

        if autogrid:
            self.gridnum_calculate()
        else:
            self.delta = delta

        x_num = int((self.info["maxlon"] - self.info["minlon"]) // self.delta) + 1
        y_num = int((self.info["maxlat"] - self.info["minlat"]) // self.delta) + 1
        self.x_num = x_num; self.y_num = y_num
        self.grid_num = x_num * y_num

        self.train_grid = []; self.test_grid = []; self.val_grid = []
        if self.is_exist():
            with open ("./data/grids/{}/{}/traingrid".format(self.dataset, self.delta), "rb") as fp:
                self.train_grid = pickle.load(fp)
            with open ("./data/grids/{}/{}/testgrid".format(self.dataset, self.delta), "rb") as fp:
                self.test_grid = pickle.load(fp)
            with open ("./data/grids/{}/{}/valgrid".format(self.dataset, self.delta), "rb") as fp:
                self.val_grid = pickle.load(fp)
        else:
            for traj in self.train_data:
                grid_traj = []
                for point in traj:    # traj = [len, lon, lat]
                    grid_traj.append(self.grid_compute(point))    # [lon, lat] -> [x, y, cell]
                self.train_grid.append(grid_traj)
            for traj in self.test_data:
                grid_traj = []
                for point in traj:
                    grid_traj.append(self.grid_compute(point))
                self.test_grid.append(grid_traj)                    
            for traj in self.val_data:
                grid_traj = []
                for point in traj:
                    grid_traj.append(self.grid_compute(point))
                self.val_grid.append(grid_traj)
            # get rid of the adjacent duplicate elements
            self.no_adjacent_duplicate()
            self.save()

    def save(self):
        """
        Save grid data
        """
        print("Saving grid data...")
        if self.train_grid is None or self.test_grid is None or self.val_grid is None:
            print("Error: No grid data generated")
            exit()

        if not os.path.exists("./data/grids/{}/{}".format(self.dataset, self.delta)):
            os.makedirs("./data/grids/{}/{}".format(self.dataset, self.delta))
        with open ("./data/grids/{}/{}/traingrid".format(self.dataset, self.delta), "wb") as fp:
            pickle.dump(self.train_grid, fp)
        with open ("./data/grids/{}/{}/testgrid".format(self.dataset, self.delta), "wb") as fp:
            pickle.dump(self.test_grid, fp)
        with open ("./data/grids/{}/{}/valgrid".format(self.dataset, self.delta), "wb") as fp:
            pickle.dump(self.val_grid, fp)
        
        with open ("./data/grids/{}/{}/info.txt".format(self.dataset, self.delta), "w") as fp:
            json.dump(self.info, fp)
            fp.write("\n")
            fp.write("delta: {}\n".format(self.delta))
            fp.write("x_num: {}\n".format(self.x_num))
            fp.write("y_num: {}\n".format(self.y_num))
            fp.write("grid_num: {}\n".format(self.grid_num))
            fp.write("\n")
            fp.close()

    def is_exist(self):
        """
        Check if grid data exists
        """
        if os.path.exists("./data/grids/{}/{}".format(self.dataset, self.delta)):
            return True
        else:
            return False
        
    def print_info(self):
        """
        Print grid information
        """
        if self.info is None:
            print("Error: No grid information")
            exit()
        print("Grid information:")
        print("maxlon: {}, minlon: {}, maxlat: {}, minlat: {}".format(self.info["maxlon"], self.info["minlon"], self.info["maxlat"], self.info["minlat"]))
        print("delta: {}, x_num: {}, y_num: {}, grid_num: {}".format(self.delta, self.x_num, self.y_num, self.grid_num))
        print("dataset: {}".format(self.dataset))

    def load_grid_data(self):
        """
        Load grid data
        """
        if self.train_grid is None or self.test_grid is None or self.val_grid is None:
            if not os.path.exists("./data/grids/{}/{}/traingrid".format(self.dataset, self.delta)):
                print("Error: No grid data generated")
                exit()
            with open ("./data/grids/{}/{}/traingrid".format(self.dataset, self.delta), "rb") as fp:
                self.train_grid = pickle.load(fp)
            with open ("./data/grids/{}/{}/testgrid".format(self.dataset, self.delta), "rb") as fp:
                self.test_grid = pickle.load(fp)
            with open ("./data/grids/{}/{}/valgrid".format(self.dataset, self.delta), "rb") as fp:
                self.val_grid = pickle.load(fp)

    def get_grid_data(self, is_load = False):
        """
        Return grid data
        """
        if is_load:
            self.load_grid_data()
        if self.train_grid is None or self.test_grid is None or self.val_grid is None:
            print("Error: No grid data generated")
            exit()
        return self.train_grid, self.test_grid, self.val_grid
    
    def get_grid_info(self, is_load = False):
        """
        Return grid information
        """
        if is_load:
            self.grid_info()
        new_info = {"x_num": self.x_num, "y_num": self.y_num, "delta": self.delta,
                     "dataset": self.dataset, "grid_num": self.grid_num}
        new_info.update(self.info)
        return new_info
    
    def grid_info(self):
        """
        grid information, just read on manual
        """
        self.x_num = 97
        self.y_num = 60
        self.delta = 0.006
        self.grid_num = self.x_num * self.y_num
        self.info = {"maxlon": -8.156309, "minlon": -8.735152,
                     "maxlat": 41.307945, "minlat": 40.953673}
        