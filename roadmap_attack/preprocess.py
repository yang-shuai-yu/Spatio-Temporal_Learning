'''
    本文件用于生成每条gps轨迹对应的路网序号向量
'''
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import ast, pickle
from shapely.geometry import Point, LineString

def load_geo_file(geo_file_path):
    df = pd.read_csv(geo_file_path)
    geometry = [LineString(ast.literal_eval(line)) for line in df['coordinates']]
    return gpd.GeoDataFrame(df, geometry=geometry)

def load_rel_file(rel_file_path):
    return pd.read_csv(rel_file_path)
    
def load_pickled_gps_file(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        gps_points = pickle.load(f)
    return gps_points

def match_gps_to_road_segments(gps_trajs, road_segments):
    matched_segments = []
    counter = 0
    for traj in gps_trajs:
        counter += 1
        traj_segments = []
        for point in traj:
            # point: [[longitude, latitude] ...]
            point = Point(point[0], point[1])  # Swap latitude and longitude if needed
            # Find the closest road segment
            closest_segment = road_segments.geometry.distance(point).idxmin()
            # Get the details of the closest segment
            traj_segments.append(road_segments.loc[closest_segment]['geo_id'])
        matched_segments.append(traj_segments)
        if counter % 10 == 0:
            print(f'Processed {counter} trajectories')

    return matched_segments

if __name__ == '__main__':
    geo_file_path = 'align_data/porto_roadmap_edge/porto_roadmap_edge.geo'
    rel_file_path = 'align_data/porto_roadmap_edge/porto_roadmap_edge.rel'
    # gps_points_file_path = 'data/gps/traingps'
    # save_path = 'align_data/aligned_gps'
    gps_points_file_path = 'data/START/gps/traingps'
    save_path = 'align_data/START/aligned_gps'
    save_name = 'aligned_traingps'

    # Load data
    current_path = os.path.dirname(os.path.abspath(__file__))
    road_segments = load_geo_file(os.path.join(current_path, geo_file_path))
    print(road_segments.head())
    print(road_segments.info())

    rel_table = load_rel_file(os.path.join(current_path, rel_file_path))
    print("***************")
    print(rel_table.head())
    print(rel_table.info())
    
    # Read GPS points from pickle file
    gps_points = load_pickled_gps_file(os.path.join(current_path, gps_points_file_path))    # longitude, latitude ...
    gps_points = gps_points[:60000]    # too slow to process all the gps points, only use for training data

    # Match GPS points to road segments
    matched_segments = match_gps_to_road_segments(gps_points, road_segments)

    # If your .rel file includes relationship information, you can use it to refine the results.
    # Save the results to a new file
    print(matched_segments[:5])

    with open(os.path.join(current_path, save_path, save_name), 'wb') as f:
        pickle.dump(matched_segments, f)