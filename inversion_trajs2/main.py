import torch
import os, sys, logging, time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from dataloader.dataloader import TrajDataset
from config import get_args, Vocab_size, MAX_LEN
from train import train
from test import test

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    print("Current working directory: ", os.getcwd())
    args = get_args()
    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    logging.getLogger("python_config_logger")
    logging.basicConfig(filename='logs/{}{}_training.log'.format(args.emb_model, args.d_model), 
                        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(args)

    trainDataset = TrajDataset(args.emb_path, args.grid_path, args.emb_model, args.d_model, mode = 'train')
    testDataset = TrajDataset(args.emb_path, args.grid_path, args.emb_model, args.d_model, mode = 'test')
    valDataset = TrajDataset(args.emb_path, args.grid_path, args.emb_model, args.d_model, mode = 'val')
    print("Loading finished!")

    if args.mode == 'train':
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = args.batch_size, shuffle = True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size = args.batch_size, shuffle = True)
        print("DataLoader finished!")

        transformer = Transformer(
            args.src_vocab_size,
            args.trg_vocab_size,
            src_pad_idx=args.src_pad_idx,
            trg_pad_idx=args.trg_pad_idx,
            trg_emb_prj_weight_sharing=args.proj_share_weight,
            emb_src_trg_weight_sharing=args.embs_share_weight,
            d_k=args.d_k,
            d_v=args.d_v,
            d_model=args.d_model,
            d_word_vec=args.d_word_vec,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout,
            scale_emb_or_prj=args.scale_emb_or_prj).to(device)

        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, transformer.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            args.lr_mul, args.d_model, args.n_warmup_steps)
        
        train(transformer, trainLoader, valLoader, optimizer, device, args)
        logging.info("the date of the experiment: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    else:
        testLoader = torch.utils.data.DataLoader(testDataset, batch_size = args.batch_size, shuffle = True)
        print("DataLoader finished!")

        transformer = Transformer(
            args.src_vocab_size,
            args.trg_vocab_size,
            src_pad_idx=args.src_pad_idx,
            trg_pad_idx=args.trg_pad_idx,
            trg_emb_prj_weight_sharing=args.proj_share_weight,
            emb_src_trg_weight_sharing=args.embs_share_weight,
            d_k=args.d_k,
            d_v=args.d_v,
            d_model=args.d_model,
            d_word_vec=args.d_word_vec,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout,
            scale_emb_or_prj=args.scale_emb_or_prj).to(device)
        
        # transformer.load_state_dict(torch.load('results/transformer_best.pth'))
        test(transformer, testLoader, device, args)    # transformer is not loaded with the best model
        logging.info("the date of the experiment: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
