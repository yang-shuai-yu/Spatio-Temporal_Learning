import torch
import os, sys, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

def train(args, trainLoader, valLoader, models, optimizers, criterion, device):
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    max_train_loss = [0x3f3f3f3f for i in range(len(models))]
    max_val_loss = [0x3f3f3f3f for i in range(len(models))]
    model_list = ['sx', 'sy', 'ex', 'ey']
    best_models = [None for i in range(len(models))]
    for epoch in range(args.epochs):
        for i, (emb, xys) in enumerate(trainLoader):    # xys: [sx,sy,ex,ey]
            loss_list = []
            for j in range(len(models)):
                models[j].to(device);    models[j].train()    # train mode
                tmp_emb = emb.to(device)
                tmp_xys = xys[:][j].to(device)    # the single x or y
                tmp_xys = tmp_xys.float()
                optimizers[j].zero_grad()
                tmp_output = models[j](tmp_emb)
                tmp_loss = criterion(tmp_output, tmp_xys)
                tmp_loss.backward()
                optimizers[j].step()
                loss_list.append(tmp_loss.item())
                if tmp_loss.item() < max_train_loss[j]:
                    max_train_loss[j] = tmp_loss.item()
                    best_models[j] = models[j]
            if i % 100 == 0:    # when epoch, just print the first for every model
                print('outputs, labels ', tmp_output[0], tmp_xys[0], tmp_output[0] - tmp_xys[0])
                print("Epoch: {}/{}, Iteration: {}/{}, Loss: {}".format(epoch, args.epochs, i, len(trainLoader), np.mean(loss_list)))
        if epoch % 10 == 0:
            os.makedirs('./models/{}/{}/'.format(args.model, args.emb_dim), exist_ok = True)
            # validation
            for j in range(len(models)):
                models[j].eval()
                val_loss = 0
                iter_num = 0
                with torch.no_grad():
                    for i, (emb, xys) in enumerate(valLoader):
                        emb = emb.to(device)
                        xys = xys[:][j].to(device)
                        output = models[j](emb)
                        loss = criterion(output, xys)
                        val_loss += loss.item()
                        iter_num += 1
                val_loss /= iter_num
                print('outputs, labels ', output[0], xys[0], output[0] - xys[0])
                print("Epoch: {}/{}, Validation Loss: {}".format(epoch, args.epochs, val_loss))
                if val_loss < max_val_loss[j]:
                    max_val_loss[j] = val_loss
                    torch.save(models[j].state_dict(), './models/{}/{}/d0_001_val_{}_{}.pth'.format(args.model, args.emb_dim, model_list[j], args.hidden_dim))
                models[j].train()

            # save the best models every 10 epochs
            for j in range(len(models)):
                torch.save(best_models[j].state_dict(), './models/{}/{}/d0_001_train_{}_{}.pth'.format(args.model, args.emb_dim, model_list[j], args.hidden_dim))
