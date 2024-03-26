import torch
import os, sys, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from modules.models import sPoints_net, ePoints_net

def test(args, testLoader, models, criterion, device):
    if device == 'gpu':
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model_list = ['sx', 'sy', 'ex', 'ey']
    for i in range(len(models)):
        models[i].to(device)
        models[i].load_state_dict(torch.load('./models/{}/{}/d0_001_val_{}_{}.pth'.format(args.model, args.emb_dim, model_list[i], args.hidden_dim)),
                                map_location = device)
        models[i].eval()

    test_loss = 0
    iter_num = 0
    test_loss = [0 for i in range(len(models))]
    with torch.no_grad():
        for i, (emb, xys) in enumerate(testLoader):
            for j in range(len(models)):
                emb = emb.to(device)
                xys = xys[:, j].to(device)
                output = models[j](emb)
                loss = criterion(output, xys)
                test_loss[j] += loss.item()
            iter_num += 1
    for i in range(len(models)):
        test_loss[i] /= iter_num
        print("Test Loss: {}".format(test_loss[i]))
    return test_loss
    