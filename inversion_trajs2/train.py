import torch
import os, sys, argparse, math, time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging

from tqdm import tqdm
import random
import torch.nn.functional as F
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='mean')
    return loss

def train_one_epoch(transformer, trainLoader, optimizer, device, args):
    transformer.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    for batch in tqdm(trainLoader, mininterval=2, desc='  - (Training)   ', leave=False):
        embeddings, grid = batch
        gold = grid.to(device)
        embeddings = embeddings.to(torch.float32).to(device)

        optimizer.zero_grad()
        pred, _ = transformer(embeddings)
        loss, n_correct, n_word = cal_performance(pred, gold, Constants.PAD, smoothing=args.label_smoothing)
        loss.backward()
        optimizer.step_and_update_lr()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    print("Loss %.4f, Acc %.2f" % (total_loss/n_word_total, n_word_correct/n_word_total))
    logging.info("Loss %.4f, Acc %.2f" % (total_loss/n_word_total, n_word_correct/n_word_total))

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def val_one_epoch(transformer, valLoader, device, args):
    transformer.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    for batch in tqdm(valLoader, mininterval=2, desc='  - (Training)   ', leave=False):
        embeddings, grid = batch
        gold = grid.to(device)
        embeddings = embeddings.to(torch.float32).to(device)

        pred, _ = transformer(embeddings)
        loss, n_correct, n_word = cal_performance(pred, gold, Constants.PAD, smoothing=args.label_smoothing)

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        # if i % 100 == 0:
        #     print("Batch %d, Loss %.4f, Acc %.2f" % (i, loss.item(), n_correct/n_word))

    print("Loss %.4f, Acc %.2f" % (total_loss/n_word_total, n_word_correct/n_word_total))
    logging.info("Loss %.4f, Acc %.2f" % (total_loss/n_word_total, n_word_correct/n_word_total))

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(transformer, trainLoader, valLoader, optimizer, device, args):
    print("Start training...")
    logging.info("Start training...")

    def print_performance(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))
        
    valid_losses = []
    train_losses = []  
    for epoch in range(args.epoch):
        print("Epoch %d" % epoch)
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            transformer, trainLoader, optimizer, device, args)
        train_ppl = math.exp(min(train_loss, 100))
        train_losses.append(train_loss)
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performance('Train', train_ppl, train_acc, start, lr)

        start = time.time()
        with torch.no_grad():
            valid_loss, valid_acc = val_one_epoch(transformer, valLoader, device, args)
            valid_ppl = math.exp(min(valid_loss, 100))
            print_performance('Valid', valid_ppl, valid_acc, start, lr)
            valid_losses.append(valid_loss)
            checkpoint = {'epoch': epoch, 'transformer': transformer.state_dict(), 'optimizer': optimizer._optimizer.state_dict()}
            if args.save_mode == 'all':
                torch.save(checkpoint, f'{args.output_dir}/transformer_{epoch}.pth')
            elif args.save_mode == 'best':
                model_name = f'{args.output_dir}/transformer_best.pth'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print(f"New best model saved to {model_name}")
            else:
                raise ValueError(f'Unrecognized save mode {args.save_mode}')
    return