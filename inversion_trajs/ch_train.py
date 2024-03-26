import logging
from numpy import inf
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from models import MLPDecoder
from data_utils import TrajDataset
import os,shutil, sys, h5py, logging, time
import numpy as np


os.chdir(sys.path[0])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


batch = 64
num_iteration = 21    ## 21个epoch
print_freq = 50
save_freq = 5
hidden_size = 256
num_layers = 1


embedding_model = 't2vec'
embedding_size = 256
vocab_size = 18866
dropout = 0.2

partition = 't2vec_partition'

if partition == 'new_partition':
    vocab_size = 21105

path = "/home/yangshuaiyu6791/RepresentAttack/inversion-attack-new"
# 这里修改了路径
checkpoint = "./trained_models/checkpoint_{}_{}_{}_{}_{}_10000.pt".format(embedding_model, str(embedding_size), str(num_layers), str(hidden_size), partition)

# def get_neighbours(trg,V):
#     neighbours = []
#     for cell in trg:
#         neighbours += list(V[cell])
#     return neighbours

# def del_zero(trj):
#     while 0 in trj:
#         trj.remove(0)
#     while 3 in trj:
#         trj.remove(3)
#     return trj

# def genLoss(gendata, m0, m1, lossF):
#     """
#     One batch loss

#     Input:
#     gendata: a named tuple contains
#         gendata.src (batch, embedding_size): input tensor
#         gendata.trg (batch, seq_len): target tensor.
    
#     m0: map input to output.
#     m1: map the output of EncoderDecoder into the vocabulary space and do
#         log transform.
#     lossF: loss function.
#     ---
#     Output:
#     loss
#     """
#     input, target = gendata.src, gendata.trg  # (batch, embedding_size), (batch, seq_len)
#     if torch.cuda.is_available():
#         input, target = input.cuda(), target.cuda()

#     output, _ = m0(input)  # (seq_len, batch, hidden_size)

#     loss = 0
    
#     output = output.view(-1, output.size(2)) # （seq_len*batch, hidden_size)
#     output = m1(output)  # (seq_len*batch, vocab_size)
#     target = target.t().contiguous()  # (seq_len, batch)
#     target = target.view(-1)  # (seq_len*batch)
#     loss = lossF(output, target)

#     return loss, output, target

# def validate(valData, model, lossF, train_iteration):
#     """
#     valData (DataLoader)
#     """
#     if partition == 'new_partition':
#         vocab_dist_cell = h5py.File('./experiment/porto-vocab-dist-cell75.h5', 'r')
#     else:
#         vocab_dist_cell = h5py.File('./experiment/porto-vocab-dist-cell100.h5', 'r')
#     D = vocab_dist_cell['D']
#     V = vocab_dist_cell['V']

#     m0 = model
#     ## switch to evaluation mode
#     m0.eval()

#     sm = nn.Softmax(dim = 1)

#     num_iteration = valData.size // batch
#     if valData.size % batch > 0: num_iteration += 1

#     total_genloss = 0
#     in_acc = 0
#     total_in_acc = 0

#     # accuracy needed
#     if (train_iteration+1) % 40000 == 0:
#         for iteration in range(num_iteration):
#             gendata = valData.getbatch_one()
#             with torch.no_grad():
#                 genloss, output, target = genLoss(gendata, m0, m1, lossF)
#                 total_genloss += genloss.item() * gendata.trg.size(0)
                
#                 target = target.view(20, -1).t().contiguous()  # (batch, seq_len)
#                 out_seq = sm(output).argmax(dim = 1)  # (seq_len*batch) 
#                 out_seq = out_seq.view(20, -1).t().contiguous()  # (batch, seq_len)

#                 target = target.cpu()
#                 out_seq = out_seq.cpu()

#                 for i in range(batch):
#                     trg_seq = del_zero(list(target[i]))
#                     pre_seq = del_zero(list(out_seq[i]))
#                     neighbours = get_neighbours(trg_seq, V)
#                     pre_len = len(pre_seq)
#                     if pre_len == 0:
#                         continue
#                     in_count = 0
#                     for cell in pre_seq:
#                         if cell in neighbours:
#                             in_count += 1
#                     in_acc = in_count/pre_len
#                     total_in_acc += in_acc
#         in_acc = total_in_acc/(num_iteration*batch)

#     else:
#         for iteration in range(num_iteration):
#             gendata = valData.getbatch_one()
#             with torch.no_grad():
#                 genloss, _, _ = genLoss(gendata, m0, m1, lossF)
#                 total_genloss += genloss.item() * gendata.trg.size(0)

#         in_acc = "NONE"

#     ## switch back to training mode
#     m0.train()
    
#     return total_genloss / valData.size, in_acc


def savecheckpoint(state, is_best):
    torch.save(state, checkpoint)
    # 这里修改了路径
    if is_best:
        shutil.copyfile(checkpoint, os.path.join(path, 'trained_models/best_model_{}_{}_{}_{}_{}_100000.pt'.format(embedding_model, str(embedding_size), str(num_layers), str(hidden_size), partition)))

def validate(valData, model, lossF, train_iteration):
    model.eval()
    total_loss = 0
    for i, (src, trg) in enumerate(valData):
        t = torch.ones(trg.shape[0])
        if torch.cuda.is_available():
            src, trg = src.cuda(), trg.cuda()
            t = t.cuda()
        trg = torch.tensor(trg, dtype=torch.long)
        pred = model(src)
        loss = lossF(pred, trg, target=t)
        total_loss += loss.item()
    model.train()
    return total_loss / len(valData)

def train(checkpoint):
    logging.basicConfig(filename="training.log", level=logging.INFO)
    """
    embedding_model: 't2vec', 'NeuTraj', 'SRN'
    embedding_size: 64, 128, 192, 256
    """


    trainsrc = './data/source/{}/train_{}.src'.format(embedding_model, str(embedding_size))
    print('trainsrc', trainsrc)
    traintrg = "./data/target/{}/train.trg".format(partition)
    print("Reading training data...")
    trainDataset = TrajDataset(vocab_size, trainsrc, traintrg)
    trainData = DataLoader(trainDataset, batch_size=batch, shuffle=True, num_workers=4)


    valsrc = './data/source/{}/val_{}.src'.format(embedding_model, str(embedding_size))
    print('valsrc:', valsrc)
    valtrg = "./data/target/{}/val.trg".format(partition)
    print("Reading validation data...")
    valDataset = TrajDataset(vocab_size, valsrc, valtrg)
    valData = DataLoader(valDataset, batch_size=batch, shuffle=False, num_workers=4)

    # create criterion, model, optimizer
    criterion = nn.CosineEmbeddingLoss()

    # m0 = Decoder(embedding_size, hidden_size, num_layers, dropout)
    # m1 = nn.Sequential(nn.Linear(hidden_size, vocab_size))
    m0 = MLPDecoder(embedding_size, vocab_size, hidden_size, num_layers, dropout)
    
    if torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
        criterion.cuda()
    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=0.01)

    if os.path.isfile(checkpoint):
        print("=> loading checkpoint '{}'".format(checkpoint))
        logging.info("Restore training @ {} {} {} {} {}".format(time.ctime(), embedding_model, embedding_size, num_layers, hidden_size))

        checkpoint = torch.load(checkpoint)
        start_iteration = checkpoint["iteration"]
        best_prec_loss = checkpoint["best_prec_loss"]
        m0.load_state_dict(checkpoint["m0"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
    
    else:
        print('There is no checkpoint!')
        logging.info("Start training @ {} {} {} {} {}".format(time.ctime(), embedding_model, embedding_size, num_layers, hidden_size))
        start_iteration = 0
        best_prec_loss = float(inf)


    print("Iteration starts at {} "
          "and will end at {}".format(start_iteration, num_iteration-1))

    ## training
    m0.train()
    for iteration in range(start_iteration, num_iteration):
        avg_genloss = []
        for i, (src, trg) in enumerate(trainData):
            t = torch.ones(trg.shape[0])
            if torch.cuda.is_available():
                src, trg = src.cuda(), trg.cuda()
                t = t.cuda()
            pred = m0(src)
            trg  = torch.tensor(trg, dtype=torch.long)
            # print(pred.shape, trg.shape)
            # print(pred, trg)
            loss = criterion(pred, trg, target=t)
            m0_optimizer.zero_grad()
            loss.backward()

            ## clip the gradients
            clip_grad_norm_(m0.parameters(), 5.0)

            ## one step optimization
            m0_optimizer.step()

            if i % 1000 == 0:
                print("Iteration: {}, batch: {}, loss: {}".format(iteration, i, loss.item()))
                avg_genloss.append(loss.item())
                with open("inv loss", "a") as f:
                    f.writelines(str(loss.item())+"\n")


        if (iteration+1) % print_freq == 0:
            print("Iteration: {}, avg_genloss: {}".format(iteration, np.mean(avg_genloss)))
            with open("avg inv loss", "a") as f:
                f.writelines(str(np.mean(avg_genloss))+"\n")
            logging.info("Iteration: {}, avg_genloss: {}".format(iteration, np.mean(avg_genloss)))

            validate_loss = validate(valData, m0, criterion, iteration)
            if validate_loss < best_prec_loss:
                best_prec_loss = validate_loss
                is_best = True
                torch.save({
                    "iteration": iteration,
                    "m0": m0.state_dict(),
                    "m0_optimizer": m0_optimizer.state_dict(),
                    "best_prec_loss": best_prec_loss
                }, checkpoint)
                print("=> saved best model")
                logging.info("=> saved best model")
            else:
                is_best = False
            print("Iteration: {}, validate loss: {}".format(iteration, validate_loss))
train(checkpoint)
