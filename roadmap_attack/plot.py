import numpy as np
import matplotlib.pyplot as plt
import torch, folium, argparse, os
import geopandas as gpd
import pandas as pd
import ast, pickle
from shapely.geometry import Point, LineString

from dataloader import *
from model import MLP
from torch.utils.data import DataLoader

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

def load_pickled_emb_file(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        embs = pickle.load(f)
    return embs

def evaluate(test_loader, model, device, args):
    output_list = []
    segment_list = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for emb, road_seg in test_loader:
            emb, road_seg = emb.to(device), road_seg.to(device)
            output = model(emb)
            output_list.append(output.cpu().numpy())
            segment_list.append(road_seg.cpu().numpy())
    output_list = np.concatenate(np.array(output_list), axis=0)
    segment_list = np.concatenate(np.array(segment_list), axis=0)
    
    return segment_list, output_list
            
def load_model(model_path):
    model = MLP(128, 512, 11095)
    model.load_state_dict(torch.load(model_path))
    return model

def get_gps_from_segment(selected_labels, selected_outputs, road_segments, rel_table):
    label_gpslist = []
    output_gpslist = []
    for i in range(len(selected_labels)):
        label_gps, output_gps = segment2gps(selected_labels[i],
                                             selected_outputs[i], road_segments, rel_table)
        # output_gps = segment2gps(selected_outputs[i], road_segments, rel_table)
        label_gpslist.append(label_gps)
        output_gpslist.append(output_gps)
    return label_gpslist, output_gpslist


def segment2gps(labels, segments, road_segments, rel_table):
    labels_idxes = np.where(labels == 1)[0]
    # print('labels_idxes:', labels_idxes)
    labels_num = np.sum(labels).astype(int)    # total number of segments
    _, predicted_idxes = torch.topk(torch.tensor(segments), labels_num)
    predicted_idxes = predicted_idxes.cpu().numpy()
    # print('predicted_idxes:', predicted_idxes)

    label_seg_slice = labels[predicted_idxes]
    correct = label_seg_slice.sum().item()
    accuracy = 100 * correct / labels_num
    print(correct, labels_num, accuracy)

    # get all road segments with indexes
    label_gps_list = []
    output_gps_list = []
    for index in labels_idxes:
        segment = road_segments.iloc[index]
        # get all gps points in the segment
        segment_gps = ast.literal_eval(segment['coordinates'])
        label_gps_list.append(np.array(segment_gps))
    label_gps_list = np.concatenate(np.array(label_gps_list), axis=0)
    
    for index in predicted_idxes:
        segment = road_segments.iloc[index]
        # get all gps points in the segment
        segment_gps = ast.literal_eval(segment['coordinates'])
        output_gps_list.append(np.array(segment_gps))
    output_gps_list = np.concatenate(np.array(output_gps_list), axis=0)
    
    return label_gps_list, output_gps_list

def load_data(geo_file_path, rel_file_path, gps_path):
    road_segments = load_geo_file(geo_file_path)
    rel_table = load_rel_file(rel_file_path)
    gps_lists = load_pickled_gps_file(gps_path)    # list of gps trajectories
    return road_segments, rel_table, gps_lists

def plot_on_map(gps_list, color, m):
    for gps in gps_list:
        # reverse the two values
        # print(gps[0], gps[1])
        gps = [float(gps[1]), float(gps[0])]
        folium.CircleMarker(gps, radius=4, color=color).add_to(m)

    # # get the polyline and reverse the two values
    # folium.PolyLine(gps_list[:, ::-1], color=color).add_to(m)

    return m

def test(gps_lists, outputs, labels, geo_file, rel_table, args):
    idx_list = np.random.choice(len(gps_lists), 5, replace=False)
    print(np.shape(labels), np.shape(outputs))
    selected_labels = labels[idx_list]
    selected_outputs = outputs[idx_list]
    selected_gps = np.array(gps_lists)[idx_list]

    label_gps, outputs_gps = get_gps_from_segment(selected_labels, selected_outputs, geo_file, rel_table)

    for i in range(len(selected_gps)):
        m = folium.Map(location=[selected_gps[i][0][1], selected_gps[i][0][0]], zoom_start=15)
        print(len(label_gps[i]), len(outputs_gps[i])) 
        m = plot_on_map(label_gps[i], 'blue', m)
        m = plot_on_map(outputs_gps[i], 'red', m)
        m.save('results/{}_{}.html'.format(args.emb_model, i))
        print('results/{}_{}.html'.format(args.emb_model, i))

def get_args():
    # get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned_path', type=str, default='align_data/aligned_gps')
    parser.add_argument('--emb_path', type=str, default='data/')
    parser.add_argument('--data_name', type=str, default='aligned_testgps')
    parser.add_argument('--emb_name', type=str, default='start_128_test')
    parser.add_argument('--model_path', type=str, default='models/start128_')
    parser.add_argument('--emb_model', type=str, default='start')

    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)

    # args = parser.parse_args()    # 调用parser.parse_args()会读取系统参数：sys.argv[]，仅命令行调用时是正确参数
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = get_args()

    geo_file_path = 'align_data/porto_roadmap_edge/porto_roadmap_edge.geo'
    rel_file_path = 'align_data/porto_roadmap_edge/porto_roadmap_edge.rel'
    gps_path = 'data/gps/testgps'
    emb_path = 'data/{}_{}_test'.format(args.emb_model, args.emb_size)
    model_path = args.model_path + 'model.ckpt'
    # model_path = 'models/eph50_{}{}.ckpt/'.format(emb_model, emb_dim)

    ### ===== load data ===== ###
    test_dataset = TrajDataset(args.aligned_path, args.emb_path, args.data_name, args.emb_name, args.mode)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    model = MLP(args.emb_size, args.hidden_size, 11095)
    model.load_state_dict(torch.load(model_path))

    geo_file, rel_table, gps_lists = load_data(geo_file_path, rel_file_path, gps_path)
    labels, outputs = evaluate(test_loader, model, 'cuda:0', args)

    # get the heads of the rel_table
    rel_table.head()
    print(geo_file.head())    # coordinates: list of [lon, lat]

    test(gps_lists, outputs, labels, geo_file, rel_table, args)

