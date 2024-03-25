import numpy as np
import random, os, pickle, math
    
class LDPServer():
    def __init__(self, dataset, epsilon, delta, info = None):
        """
        Get the data distribution of the given dataset
        Including length distribution, transition distribution, start/end distribution
        """
        self.dataset = dataset
        self.epsilon = epsilon
        self.epsilon_1 = epsilon / 10.0
        self.epsilon_2 = (epsilon - self.epsilon_1) / 2.0
        self.epsilon_3 = self.epsilon_2
        self.delta = delta
        self.info = info    # info = {"maxlon", "minlon", "maxlat", "minlat","dataset"} etc...
        self.n = None
        self.grid_data = None
        self.max_grid_lenth = -1

    def lenth_calculate(self):
        """
        Calculate the length distribution
        """
        raise NotImplementedError
    
    def transition_calculate(self):
        """
        Calculate the transition distribution
        """
        raise NotImplementedError
    
    def startend_calculate(self):
        """
        Calculate the start/end distribution
        """
        raise NotImplementedError
    
class GenServer(LDPServer):
    def __init__(self, dataset, epsilon, delta, info = None):
        """
        Get the data distribution of the given dataset
        Including length distribution, transition distribution, start/end distribution
        """
        super().__init__(dataset, epsilon, delta, info)
        self.grid_data = None
        self.n = -1   # number of trajectories
        self.length_dist = None
        self.transition_dist = None
        self.start_dist = None
        self.end_dist = None

        if self.info is None:
            raise ValueError("Error: No info")
        self.grid_num = self.info["grid_num"]
        self.x_num = self.info["x_num"]
        self.y_num = self.info["y_num"]

    def load_data(self, grid_data):
        """
        Load grid data from grid_data
        get the number of trajectories -- n
        """
        self.grid_data = grid_data
        self.n = len(grid_data)
        self.max_grid_lenth = max([len(traj) for traj in self.grid_data])

    def lenth_preturb(self,vector):
        """
        Preturb the length distribution
        """
        # print("Preturbing length distribution...")
        p = 1/2; q = 1.0/(math.exp(self.epsilon_1) + 1.0)
        for i in range(len(vector)):
            if vector[i] == 0:
                vector[i] = random.choices([0,1], weights = [1-q, q])[0]
            else:
                vector[i] = random.choices([0,1], weights = [1-p, p])[0]
        return vector

    def lenth_calculate(self):
        """
        Calculate the length distribution
        """
        print("Calculating length distribution...")
        max_length = 0
        max_length = max([len(traj) for traj in self.grid_data])

        length_dist = np.zeros(max_length + 1)    # the length distribution
        # add the epsilon differential privacy
        for traj in self.grid_data:
            encod_vector = np.zeros(max_length + 1)
            encod_vector[len(traj)] = 1
            length_dist += self.lenth_preturb(encod_vector)
        length_dist  = length_dist / sum(length_dist)
        self.length_dist = length_dist
        
    def transition_preturb(self, vector):
        """
        preturb the transition distribution
        """
        p = 1/2; q = 1.0/(math.exp(self.epsilon_2/self.max_grid_lenth) + 1.0)
        for i in range(len(vector)):
            if vector[i] == 0:
                vector[i] = random.choices([0,1], weights = [1-q, q])[0]
            else:
                vector[i] = random.choices([0,1], weights = [1-p, p])[0]
        return vector
        
    def transition_calculate(self):
        """
        Calculate the transition distribution
        For each grid, calculate the transition distribution to the 8 adjacent grids
        6 7 8
        3 * 5
        0 1 2
        """
        print("Calculating transition distribution...")
        transition_dist = np.zeros((self.grid_num + 1, 8))    # the transition distribution, 8 adjacent grids
        # add a map to project the value to the vector
        mapping = {0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 7:6, 8:7}
        # add the epsilon differential privacy
        for traj in self.grid_data:
            for i in range(len(traj) - 1): # read all the adjacent grids
                encode_vector = np.zeros(8)
                # print(traj[i], traj[i+1])
                dx = traj[i+1][0] - traj[i][0]
                dy = traj[i+1][1] - traj[i][1]
                # print(dx, dy)
                if dx < -1 or dx > 1 or dy < -1 or dy > 1:
                    continue
                encode_vector[mapping[dx + 1 + (dy + 1) * 3]] = 1
                transition_dist[traj[i][2]] += self.transition_preturb(encode_vector)
        # normalize the transition distribution
        transition_dist[transition_dist == 0] = 1e-10    # avoid the 0 value
        transition_dist = transition_dist / np.sum(transition_dist, axis = 1, keepdims = True)
        self.transition_dist = transition_dist

    def startend_preturb(self, vector):
        """
        Preturb the start/end distribution
        """
        # print("Preturbing start/end distribution...")
        p = 1/2; q = 1.0/(math.exp(self.epsilon_3) + 1.0)
        if vector[0] == 0:
            vector[0] = random.choices([0,1], weights = [1-q, q])[0]
        else:
            vector[0] = random.choices([0,1], weights = [1-p, p])[0]
        return vector

    def startend_calculate(self):
        """
        Calculate the start/end distribution
        """
        print("Calculating start/end distribution...")
        start_dist = np.zeros(self.grid_num + 1)    # the start distribution
        end_dist = np.zeros(self.grid_num + 1)    # the end distribution
        # add the epsilon differential privacy
        for traj in self.grid_data:
            start_tmp = np.zeros(self.grid_num + 1)
            end_tmp = np.zeros(self.grid_num + 1)
            start_tmp[traj[0][2]] = 1
            end_tmp[traj[-1][2]] = 1
            start_dist += self.startend_preturb(start_tmp)
            end_dist += self.startend_preturb(end_tmp)
        start_dist[0] = 0; end_dist[0] = 0    # set the 0 row to 0, because the 0 row is not used
        start_dist = start_dist / sum(start_dist)
        end_dist = end_dist / sum(end_dist)

        self.start_dist = start_dist
        self.end_dist = end_dist

    def generate(self):
        """
        Generate the distribution
        """
        print("Generating distribution...")
        if self.grid_data is None:
            print("Error: No grid data")
            exit()
        self.lenth_calculate()
        self.transition_calculate()
        self.startend_calculate()

    def save(self):
        """
        Save the distribution
        """
        print("Saving distribution...")
        if not os.path.exists("./data/ldps/{}/{}/{}/".format(self.dataset, self.delta, self.epsilon)):
            os.makedirs("./data/ldps/{}/{}/{}/".format(self.dataset, self.delta, self.epsilon), exist_ok=True)
        with open("./data/ldps/{}/{}/{}/length_dist".format(self.dataset, self.delta, self.epsilon), "wb") as fp:
            pickle.dump(self.length_dist, fp)
        with open("./data/ldps/{}/{}/{}/transition_dist".format(self.dataset, self.delta, self.epsilon), "wb") as fp:
            pickle.dump(self.transition_dist, fp)
        with open("./data/ldps/{}/{}/{}/start_dist".format(self.dataset, self.delta, self.epsilon), "wb") as fp:
            pickle.dump(self.start_dist, fp)
        with open("./data/ldps/{}/{}/{}/end_dist".format(self.dataset, self.delta, self.epsilon), "wb") as fp:
            pickle.dump(self.end_dist, fp)
        with open("./data/ldps/{}/{}/{}/info".format(self.dataset, self.delta, self.epsilon), "wb") as fp:
            pickle.dump(self.info, fp)

    def load(self):
        """
        Load the distribution
        """
        print("Loading distribution...")
        with open("./data/ldps/{}/{}/{}/length_dist".format(self.dataset, self.delta, self.epsilon), "rb") as fp:
            self.length_dist = pickle.load(fp)
        with open("./data/ldps/{}/{}/{}/transition_dist".format(self.dataset, self.delta, self.epsilon), "rb") as fp:
            self.transition_dist = pickle.load(fp)
        with open("./data/ldps/{}/{}/{}/start_dist".format(self.dataset, self.delta, self.epsilon), "rb") as fp:
            self.start_dist = pickle.load(fp)
        with open("./data/ldps/{}/{}/{}/end_dist".format(self.dataset, self.delta, self.epsilon), "rb") as fp:
            self.end_dist = pickle.load(fp)
        with open("./data/ldps/{}/{}/{}/info".format(self.dataset, self.delta, self.epsilon), "rb") as fp:
            self.info = pickle.load(fp)

    def get_dist(self):
        """
        Get the distribution
        """
        if self.length_dist is None or self.transition_dist is None or self.start_dist is None or self.end_dist is None:
            print("Error: No distribution")
            exit()
        return self.length_dist, self.transition_dist, self.start_dist, self.end_dist

    def get_info(self):
        """
        Get the info
        """
        if self.info is None:
            print("Error: No info")
            exit()
        return self.info

class GenClient(LDPServer):
    def __init__(self, dataset, epsilon, delta, info = None):
        """
        Get the data distribution of the given dataset
        Trajectory Synthesis
        """
        super().__init__(dataset, epsilon, delta, info)
        self.length_dist = None
        self.transition_dist = None
        self.start_dist = None
        self.end_dist = None
        self.syn_traj = None
        self.info = None

    def load_dist(self, LDPserver):
        """
        Read the distribution from the server
        """
        self.length_dist, self.transition_dist, self.start_dist, self.end_dist = LDPserver.get_dist()
        print(self.transition_dist)
        self.info = LDPserver.get_info()
        print(self.info)

    def single_synthesis(self, length_list, start_list, end_list, grid_list):
        """
        Synthesize a trajectory
        """
        # print("Synthesizing a trajectory...")
        mapping = {0:0, 1:1, 2:2, 3:3, 4:5, 5:6, 6:7, 7:8}    # map the vector to the adjacent grid
        traj = []; pre_cell = -1
        # sample the length
        length = np.random.choice(length_list, p = self.length_dist)
        if length == 0:
            return traj
        # sample the start
        start = np.random.choice(start_list, p = self.start_dist)
        traj.append(start); pre_cell = start
        # sample the end
        end = np.random.choice(end_list, p = self.end_dist)
        # sample the trajectory between start and end
        adj_list = np.arange(8)
        for i in range(length - 1):
            cur_cell = -1    # initialize the current cell
            while cur_cell < 0 or cur_cell > self.info['grid_num']:    # if the cell is out of the grid, resample
                cur_adj = np.random.choice(adj_list, p = self.transition_dist[pre_cell])
                cur_adj = mapping[cur_adj]
                cur_cell = pre_cell + (cur_adj % 3 - 1) + (cur_adj // 3 - 1) * self.info['x_num']   # get the current cell from the adjacent cell
            traj.append(cur_cell); pre_cell = cur_cell
            if cur_cell == end:
                break
        return traj
    
    def trajs_synthesis(self, num):
        """
        Synthesize trajectories
        """
        print("Synthesizing trajectories...")
        syn_traj = []

        # get the length list, start list, end list from the distribution
        lenth_list = [i for i in range(len(self.length_dist))]
        start_list = [i for i in range(len(self.start_dist))]
        end_list = [i for i in range(len(self.end_dist))]
        grid_list = [i for i in range(len(self.transition_dist))]

        # trajectory synthesis
        for i in range(num):
            syn_traj.append(self.single_synthesis(lenth_list, start_list,
                                                   end_list, grid_list))
            if i %500 == 0:
                print("Synthesized {} trajectories".format(i))
        self.syn_traj = syn_traj

    def save(self):
        """
        Save the synthetic trajectories
        """
        print("Saving synthetic trajectories...")
        if not os.path.exists("./data/syntrajs/{}/{}/{}/".format(self.dataset, self.delta, self.epsilon)):
            os.makedirs("./data/syntrajs/{}/{}/{}/".format(self.dataset, self.delta, self.epsilon), exist_ok=True)
        with open("./data/syntrajs/{}/{}/{}/syn_traj".format(self.dataset, self.delta, self.epsilon), "wb") as fp:
            pickle.dump(self.syn_traj, fp)
        with open("./data/syntrajs/{}/{}/{}/syn_info".format(self.dataset, self.delta, self.epsilon), "wb") as fp:
            pickle.dump(self.info, fp)
        
    def load(self):
        """
        Load the synthetic trajectories
        """
        print("Loading synthetic trajectories...")
        with open("./data/syntrajs/{}/{}/{}/syn_traj".format(self.dataset, self.delta, self.epsilon), "rb") as fp:
            self.syn_traj = pickle.load(fp)
        with open("./data/syntrajs/{}/{}/{}/syn_info".format(self.dataset, self.delta, self.epsilon), "rb") as fp:
            self.info = pickle.load(fp)

    def get_syn_trajs(self):
        """
        Get the synthetic trajectories
        """
        if self.syn_traj is None:
            if os.path.exists("./data/syntrajs/{}/{}/{}/syn_traj".format(self.dataset, self.delta, self.epsilon)):
                self.load()
            else:
                print("Error: No synthetic trajectories")
                exit()
        return self.syn_traj





        


        
