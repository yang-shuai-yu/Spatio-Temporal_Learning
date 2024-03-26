import torch
import os, sys, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from modules.models import sPoints_net, ePoints_net

def test(args, testLoader, s_model, e_model, criterion1, criterion2, device):
    if device == 'gpu':
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    s_model.to(device); e_model.to(device)
    s_model.load_state_dict(torch.load('./models/d0_001_val_sPoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim)),
                            map_location = device)
    e_model.load_state_dict(torch.load('./models/d0_001_val_ePoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim)),
                            map_location = device)
    s_model.eval(); e_model.eval()

    test_loss = 0
    iter_num = 0
    with torch.no_grad():
        for i, (emb, se_cells) in enumerate(testLoader):
            emb = emb.to(device)
            se_cells = se_cells.to(device)
            s_output = s_model(emb); e_output = e_model(emb)
            s_loss = criterion1(s_output, se_cells[:, 0])
            e_loss = criterion2(e_output, se_cells[:, 1])
            test_loss += s_loss.item() + e_loss.item()
            iter_num += 1
    test_loss /= iter_num
    print("Test Loss: {}".format(test_loss))
    return test_loss
    
    