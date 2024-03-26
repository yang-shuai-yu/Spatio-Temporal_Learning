import os, pickle
import numpy as np

EARTH_RADIUS = 6371.0

def hav(theta):
    s = np.sin(theta / 2)
    return s * s

def get_distance_hav(lon0, lat0, lon1, lat1):    # [lon, lat], [lon, lat]    
    lat0 = np.radians(lat0)
    lat1 = np.radians(lat1)
    lon0 = np.radians(lon0)
    lon1 = np.radians(lon1)

    dlat = lat0 - lat1
    dlon = lon0 - lon1

    h = hav(dlat) + np.cos(lat0) * np.cos(lat1) * hav(dlon)
    distance = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(h))
    return distance

def get_journey_distance(gps_data):
    journey_distance = []
    # use map function to get the distance among all the points
    for i in range(len(gps_data)):
        # map all the points in one trajectory to the distance
        distance = list(map(lambda x, y: get_distance_hav(x[0], x[1], y[0], y[1]), gps_data[i][:-1], gps_data[i][1:]))
        journey_distance.append(np.sum(distance))
    journey_distance = np.array(journey_distance)
    return journey_distance


def save_journey_distance(journey_distance, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(journey_distance, f)
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    print(os.getcwd()) # /home/yangshuaiyu6791/RepresentAttack/inversion_dist
    current_path = os.getcwd()
    gps_path = 'data/gps/testgps'
    save_path = 'data/dist/testdist'
    gps_data = pickle.load(open(os.path.join(current_path, gps_path), 'rb'))
    
    # compute the distance between start and end point
    distance = get_journey_distance(gps_data)    # numpy array, shape: (num_traj, )
    print(len(distance))
    print(distance[:10])
    save_journey_distance(distance, os.path.join(current_path, save_path))
