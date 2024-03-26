import torch
import torch.nn as nn
import os, pickle, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model import InversionDistModel
from dataloader import DistDataset

def train(model, dataloader, criterion, optimizer, args):
    min_loss = float('inf')
    mean_loss = 0.0
    for epoch in range(args.epochs):
        running_loss = 0.0
        mean_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        mean_loss = running_loss / len(dataloader)
        if mean_loss < min_loss:
            min_loss = mean_loss
            torch.save(model.state_dict(), args.model_path)
    print("Finished training")

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Inversion Distance Model')
    parser.add_argument('--dist_path', type=str, default='data/dist/traindist', help='Path to the distance data')
    parser.add_argument('--emb_path', type=str, default='data/t2vec_256_train', help='Path to the embedding data')
    parser.add_argument('--model_path', type=str, default='models/trajdist_t2vec256.pth', help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default = 512, help='Hidden dimension')
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

    # load the model
    print("Loading model...")
    emb_dim = dist_dataset.emb_data.shape[1]
    print(emb_dim, args.hidden_dim, args.output_dim)
    model = InversionDistModel(emb_dim, args.hidden_dim, args.output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model loaded!")

    # Train the model
    print("Training model...")
    train(model, dataloader, criterion, optimizer, args)