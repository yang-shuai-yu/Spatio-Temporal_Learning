import torch
import torch.nn as nn
import os, pickle, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

from model import InversionDistModel
from dataloader import DistDataset

def evaluate_loss(model, dataloader, criterion, args):
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            # Compute the loss by using the absolute value rather than the MSEloss
            loss = torch.abs(outputs - labels)
            running_loss += loss.mean().item()
    mean_loss = running_loss / len(dataloader)
    print('Mean loss: %.3f' % mean_loss)

def evaluate_shotrate(model, dataset, dataloader, gps_data, args):
    shotrate_list = []
    for i in range(3):
        shotrate_list.append(evaluate_shotrate_one(model, dataset, dataloader, gps_data, args))
    print('Mean shot accuracy: %.3f' % np.mean(shotrate_list))


def evaluate_shotrate_one(model, dataset, dataloader, gps_data, args):
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    shot_accuracy = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            gps_batch = gps_data[i*args.batch_size:(i+1)*args.batch_size]

            # denormalize the data
            center = torch.cat((outputs[:, 0].unsqueeze(1) * dataset.std_centerX + dataset.mean_centerX,
                                 outputs[:, 1].unsqueeze(1) * dataset.std_centerY + dataset.mean_centerY), 1)
            radius = outputs[:, 2].unsqueeze(1) * dataset.std_radius + dataset.mean_radius
            
            # compute the number of trajectory points within the circle
            for i in range(args.batch_size):
                gps = torch.tensor(gps_batch[i])
                distance = torch.sqrt(torch.sum((gps - center[i])**2, 1))
                shot_accuracy = torch.cat((shot_accuracy, (distance <= radius[i]).float()), 0)
            # break the loop
            if i >= len(dataloader)-1:
                break
    shot_accuracy = shot_accuracy.numpy()
    return shot_accuracy




if __name__ == '__main__':
    ### === Example === ###
    # python train.py --emb_path data/t2vec_128_train --model_path models/externCircle_t2vec128.pth --dist_path data/circle/traincircle --epochs 20
    # python train.py --emb_path data/t2vec_128_train --model_path models/eph50dim512_t2vec128.pth --dist_path data/circle/traincircle --epochs 50
    # python evaluate.py --emb_path data/t2vec_128_test --model_path models/externCircle_t2vec128.pth --dist_path data/circle/testcircle --epochs 20
    ### === Example === ###

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Inversion Distance Model')
    parser.add_argument('--dist_path', type=str, default='data/circle/testcircle', help='Path to the distance data')
    parser.add_argument('--gps_path', type=str, default='data/gps/testgps', help='Path to the gps data')
    parser.add_argument('--emb_path', type=str, default='data/t2vec_256_test', help='Path to the embedding data')
    parser.add_argument('--model_path', type=str, default='models/externCircle_t2vec256.pth', help='Path to the model')
    
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
    # # slice the gps data as the batch size as the same as the dataloader
    # gps_data = gps_data[:(len(dataloader)-1) * args.batch_size]
    print("Data loaded!")

    # Load the model
    print("Loading model...")
    emb_dim = dist_dataset.emb_data.shape[1]
    model = InversionDistModel(emb_dim, args.hidden_dim, args.output_dim)
    criterion = nn.MSELoss()
    print("Model loaded!")

    # evaluate the model
    print("Evaluating model...")
    evaluate_loss(model, dataloader, criterion, args)
    evaluate_shotrate(model, dist_dataset, dataloader, gps_data, args)
