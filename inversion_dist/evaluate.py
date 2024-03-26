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

def evaluate_distance(model, dataset, dataloader, args):
    mae_list = []
    for i in range(10):
        mae_list.append(evaluate_distance_one(model, dataset, dataloader, args))
    print('Mean absolute distance error: %.3f' % np.mean(mae_list))


def evaluate_distance_one(model, dataset, dataloader, args):
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    total_outputs = torch.tensor([])
    total_labels = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            # # gaussian random output
            # print("random output")
            # zero_list = torch.zeros_like(labels)
            # outputs = torch.normal(mean=zero_list, std=1)
            # get the mean and std of the distance data
            
            total_labels = torch.cat((total_labels, labels), 0)
            total_outputs = torch.cat((total_outputs, outputs), 0) 

    mean = dataset.mean
    std = dataset.std
    # denormalize the data
    outputs = outputs * std + mean
    labels = labels * std + mean
    # compute the mean absolute error
    mae = torch.abs(outputs - labels).mean()
    print('Mean absolute error: %.3f' % mae)
    return mae

if __name__ == '__main__':
    '''
    Usage:
        python evaluate.py --emb_path data/start_128_test --model_path models/eph20dim256_start128.pth --hidden_dim 256 --epochs 20
    '''
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Inversion Distance Model')
    parser.add_argument('--dist_path', type=str, default='data/dist/testdist', help='Path to the distance data')
    parser.add_argument('--emb_path', type=str, default='data/t2vec_256_test', help='Path to the embedding data')
    parser.add_argument('--model_path', type=str, default='models/inversion_dist_t2vec256.pth', help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default = 256, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    # Set random seed  
    np.random.seed(args.seed)

    # Load the data
    print("Loading data...")
    current_path = os.getcwd()
    dist_path = os.path.join(current_path, args.dist_path)
    emb_path = os.path.join(current_path, args.emb_path)
    dist_dataset = DistDataset(dist_path, emb_path)
    dataloader = DataLoader(dist_dataset, batch_size=args.batch_size, shuffle=True)
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
    evaluate_distance(model, dist_dataset, dataloader, args)
