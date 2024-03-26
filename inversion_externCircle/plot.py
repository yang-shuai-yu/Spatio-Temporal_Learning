import torch
import torch.nn as nn
import os, pickle, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random, folium, math

from model import InversionDistModel
from dataloader import DistDataset

def evaluate(model, dataloader, gps_data, args):
    output_list = []
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            # save as numpy in the list
            output_list.append(outputs.cpu().numpy())
    # concatenate
    output_list = np.concatenate(np.array(output_list), axis = 0, dtype = 'object')
    print(np.shape(output_list))
    return output_list

def plot(output_array, dataloader, gps_data, args):
    # select indexes
    index_array = np.random.choice(len(output_array), 5, replace = False)
    # output array to circle points and radius
    circle_x = output_array[:, 0] * dataloader.dataset.std_centerX.cpu().numpy() + dataloader.dataset.mean_centerX.cpu().numpy()
    circle_y = output_array[:, 1] * dataloader.dataset.std_centerY.cpu().numpy() + dataloader.dataset.mean_centerY.cpu().numpy()
    radius = output_array[:, 2] * dataloader.dataset.std_radius.cpu().numpy() + dataloader.dataset.mean_radius.cpu().numpy()
    # # get selected indexs
    # selected_gps = gps_data[index_array]
    # selected_circleX = circle_x[index_array]
    # selected_CircleY = circle_Y[index_array]
    # selected_radius = radius[index_array]

    for i in range(len(index_array)):
        plot_one(i,circle_x[index_array[i]], circle_y[index_array[i]], radius[index_array[i]], gps_data[index_array[i]], args)

    print("Plotting finished!")

def plot_one(index, circle_x, circle_y, radius, gps_data, args):
    # get circle by radius and angle (cos, sin)
    theta = np.array([2 * math.pi*i/72 for i in range(73)])    # get 72 points to draw a circle, use 73 to close the circle
    circle = np.array([np.cos(theta)*radius, np.sin(theta)*radius]).transpose()
    circle = np.array([circle_x, circle_y]) + circle
    # get the extern circle gps data
    gps_data = np.array(gps_data)
    # plot, reverse the data column
    m = folium.Map(location=[gps_data[0][1], gps_data[0][0]], zoom_start=15)
    folium.PolyLine(gps_data[:,::-1], color="blue", weight=2.5, opacity=1).add_to(m)
    folium.PolyLine(circle[:,::-1], color="red", weight=2.5, opacity=1).add_to(m)
    m.save(os.path.join(args.save_path + 'plot_{}.html'.format(index)))
    





if __name__ == '__main__':
    ### === Example === ###
    # python plot.py --emb_path data/t2vec_128_test --model_path models/externCircle_t2vec128.pth --dist_path data/circle/testcircle --save_path results/t2vec128
    ### === Example === ###
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Inversion Distance Model')
    parser.add_argument('--dist_path', type=str, default='data/circle/testcircle', help='Path to the distance data')
    parser.add_argument('--gps_path', type=str, default='data/gps/testgps', help='Path to the gps data')
    parser.add_argument('--emb_path', type=str, default='data/t2vec_256_test', help='Path to the embedding data')
    parser.add_argument('--model_path', type=str, default='models/externCircle_t2vec256.pth', help='Path to the model')
    parser.add_argument('--save_path', type=str, default='results/t2vec256', help='Path to save the figures')
    
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default = 512, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    # Set random seed  
    np.random.seed(args.seed)

    # Load the data
    print("Loading data...")
    current_path = os.getcwd()
    dist_path = os.path.join(current_path, args.dist_path)
    emb_path = os.path.join(current_path, args.emb_path)
    gps_path = os.path.join(current_path, args.gps_path)

    dist_dataset = DistDataset(dist_path, emb_path)
    dataloader = DataLoader(dist_dataset, batch_size=args.batch_size, shuffle=False)
    gps_data = np.array(pickle.load(open(gps_path, 'rb')))
    print("Data loaded!")

    # Load the model
    print("Loading model...")
    emb_dim = dist_dataset.emb_data.shape[1]
    model = InversionDistModel(emb_dim, args.hidden_dim, args.output_dim)
    criterion = nn.MSELoss()
    print("Model loaded!")

    # plot the results
    print("Plotting...")
    output_array = evaluate(model, dataloader, gps_data, args)
    plot(output_array, dataloader, gps_data, args)
