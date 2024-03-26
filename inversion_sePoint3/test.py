import torch
import os, sys, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from modules.models import sPoints_net, ePoints_net

def test(args, testLoader, se_model, criterion, device):
    if device == 'gpu':
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    test_loss = 0
    iter_num = 0
    se_model.to(device)
    se_model.eval()
    with torch.no_grad():
        for i, (emb, se_cells) in enumerate(testLoader):
            emb = emb.to(device)
            se_cells = se_cells.to(device)
            se_cells = torch.cat((se_cells[:, 0], se_cells[:, 1]), 1)
            se_output = se_model(emb)
            se_loss = criterion(se_output, se_cells)
            iter_num += 1
            test_loss += se_loss.item()
    test_loss /= iter_num
    print("Test Loss: {}".format(test_loss))
    return test_loss
    
    