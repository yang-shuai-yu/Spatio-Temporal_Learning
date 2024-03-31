''' 
    evalueate the performance of the model
'''
import os, argparse
import numpy as np
import torch
from dataloader import TrajDataset
from model import MLP
from torch.utils.data import DataLoader
import torch.nn as nn

def evaluate(test_loader, model, criterion, args):
    model.load_state_dict(torch.load(args.model_path+'model.ckpt'))
    model.eval()
    with torch.no_grad():
        acc_list = []
        for emb, road_seg in test_loader:
            emb = emb.float()
            road_seg = road_seg.float()
            outputs = model(emb)
            # get the max k index for each row, k is the number of road segment for each line
            k = torch.sum(road_seg, 1)    # k is the list of the number of road segment for each line
            for i in range(len(k)):
                correct = 0
                total = k[i].item()
                _, predicted = torch.topk(outputs[i], int(k[i]))
                # get the slice of index by the predicted
                road_seg_slice = road_seg[i][predicted]
                # predicted is the index of the max k value
                correct = road_seg_slice.sum().item()
                accuracy = 100 * correct / total
                print(f'Accuracy of the network on the test data: {accuracy} %')
                acc_list.append(accuracy)
        print(f'Accuracy of the network on the test data: {np.mean(acc_list)} %')

def loss_evaluate(test_loader, model, criterion, args):
    model.load_state_dict(torch.load(args.model_path+'model.ckpt'))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_list = []
        for emb, road_seg in test_loader:
            emb = emb.float()
            road_seg = road_seg.float()
            outputs = model(emb)
            loss = criterion(outputs, road_seg)
            loss_list.append(loss.item())
        print(f'Loss of the network on the test data: {np.mean(loss_list)}')

        

def prob_evaluate(test_loader, model, criterion, args):
    model.load_state_dict(torch.load(args.model_path+'model.ckpt'))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for emb, road_seg in test_loader:
            emb = emb.float()
            road_seg = road_seg.float()
            outputs = model(emb)
            # get the probability of the road segment for each line
            for i in range(len(road_seg)):
                prob = outputs[i][road_seg[i] == 1]
                correct += prob.sum().item()
                total += len(prob)
        print(f'Probability of the network on the test data: {100 * correct / total} %')

def points2line(points):
    # points: [batch_size, 2, 2]
    # return: [batch_size, 2]
    x1 = points[:, 0, 0]
    y1 = points[:, 0, 1]
    x2 = points[:, 1, 0]
    y2 = points[:, 1, 1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    '''
        import numpy as np
        from shapely.geometry import LineString, Point
        import folium

        # 假设已有轨迹点集，这里用随机生成的点集代替
        np.random.seed(42)
        num_points = 100
        min_lat, max_lat = 40, 41
        min_lon, max_lon = -74, -73
        trajectory_points = [Point(np.random.uniform(min_lon, max_lon), np.random.uniform(min_lat, max_lat)) for _ in range(num_points)]

        # 根据点之间的距离排序轨迹点
        sorted_trajectory_points = [trajectory_points[0]]
        remaining_points = trajectory_points[1:]
        while remaining_points:
            last_point = sorted_trajectory_points[-1]
            nearest_point_index = min(range(len(remaining_points)), key=lambda i: remaining_points[i].distance(last_point))
            sorted_trajectory_points.append(remaining_points.pop(nearest_point_index))

        # 将轨迹点连接成一条线路
        trajectory_line = LineString([point.coords[0] for point in sorted_trajectory_points])

        # 在地图上绘制轨迹线路
        map_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
        mymap = folium.Map(location=map_center, zoom_start=12)
        folium.GeoJson(data=trajectory_line.__geo_interface__, name="trajectory").add_to(mymap)
        mymap.save("sorted_trajectory_map.html")

    '''
    return torch.stack([a, b, c], dim=1)

if __name__ == '__main__':
    # get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned_path', type=str, default='align_data/aligned_gps')
    parser.add_argument('--emb_path', type=str, default='data/')
    parser.add_argument('--data_name', type=str, default='aligned_testgps')
    parser.add_argument('--emb_name', type=str, default='start_128_test')
    parser.add_argument('--model_path', type=str, default='models/start128_')

    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    test_dataset = TrajDataset(args.aligned_path, args.emb_path, args.data_name, args.emb_name, args.mode)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    # model
    model = MLP(args.emb_size, args.hidden_size, 11095)
    criterion = nn.BCELoss()

    # evaluate
    evaluate(test_loader, model, criterion, args)
    prob_evaluate(test_loader, model, criterion, args)
    loss_evaluate(test_loader, model, criterion, args)