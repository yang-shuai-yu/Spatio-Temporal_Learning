import os, argparse
import numpy as np
import torch
from dataloader import TrajDataset
from model import MLP
from torch.utils.data import DataLoader
import torch.nn as nn

def train(train_loader, model, criterion, optimizer, args):
    for epoch in range(args.num_epochs):
        for i, (emb, road_seg) in enumerate(train_loader):
            emb = emb.float()
            road_seg = road_seg.float()
            # Forward pass
            outputs = model(emb)
            loss = criterion(outputs, road_seg)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # Save the model checkpoint
    torch.save(model.state_dict(), args.model_path+'model.ckpt')


if __name__ == '__main__':
    '''
        Usage:
            python train.py --emb_name transformer_128_train --model_path models/trans128_ --emb_size 128
    '''

    # get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned_path', type=str, default='align_data/aligned_gps')
    parser.add_argument('--data_name', type=str, default='aligned_traingps_60000')
    parser.add_argument('--emb_path', type=str, default='data/')
    parser.add_argument('--emb_name', type=str, default='neutraj_128_train')
    parser.add_argument('--model_path', type=str, default='models/neutraj128_')

    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    train_dataset = TrajDataset(args.aligned_path, args.emb_path, args.data_name, args.emb_name, args.mode)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # model
    model = MLP(args.emb_size, args.hidden_size, 11095)    # hidden_size = 512, output_size = 11095
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    train(train_loader, model, criterion, optimizer, args)

