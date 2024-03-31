import argparse
import os, sys
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

Vocab_size = 18866
MAX_LEN = 20 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_path', type=str, default='data/embs/')
    parser.add_argument('-grid_path', type=str, default='data/grids/t2vec_partition/')
    parser.add_argument('-emb_model', type=str, default='transformer')
    parser.add_argument('-mode', type=str, default='train')
    
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_inner', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-seed', type=int, default=1234)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default='models/')
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-src_vocab_size', type=int, default=Vocab_size)
    parser.add_argument('-trg_vocab_size', type=int, default=Vocab_size)
    parser.add_argument('-src_pad_idx', type=int, default=Constants.PAD)
    parser.add_argument('-trg_pad_idx', type=int, default=Constants.PAD)
    parser.add_argument('-lr_mul', type=float, default=2.0)

    args = parser.parse_args()

    return args