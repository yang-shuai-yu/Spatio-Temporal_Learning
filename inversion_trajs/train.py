import logging
from numpy import inf
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import TransformerDecoder as Decoder
# from models import LSTMDeocder as Decoder
from data_utils import DataLoader
import os,shutil, sys, h5py, logging, time


os.chdir(sys.path[0])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


batch = 128
num_iteration = 50000    ## 我觉得50000轮足够了，不需要原来的100000轮
print_freq = 50
save_freq = 1000
hidden_size = 256
num_layers = 3


embedding_model = 't2vec'
embedding_size = 256
vocab_size = 18866
dropout = 0.2

partition = 't2vec_partition'

if partition == 'new_partition':
    vocab_size = 21105

path = "/home/yangshuaiyu6791/RepresentAttack/inversion_trajs"
# 这里修改了路径
checkpoint = "./trained_models/checkpoint_{}_{}_{}_{}_{}_10000.pt".format(embedding_model, str(embedding_size), str(num_layers), str(hidden_size), partition)

def get_neighbours(trg,V):
    neighbours = []
    for cell in trg:
        neighbours += list(V[cell])
    return neighbours

def del_zero(trj):
    while 0 in trj:
        trj.remove(0)
    while 3 in trj:
        trj.remove(3)
    return trj

def genLoss(gendata, m0, m1, lossF):
    """
    One batch loss

    Input:
    gendata: a named tuple contains
        gendata.src (batch, embedding_size): input tensor
        gendata.trg (batch, seq_len): target tensor.
    
    m0: map input to output.
    m1: map the output of EncoderDecoder into the vocabulary space and do
        log transform.
    lossF: loss function.
    ---
    Output:
    loss
    """
    input, target = gendata.src, gendata.trg  # (batch, embedding_size), (batch, seq_len)
    if torch.cuda.is_available():
        input, target = input.cuda(), target.cuda()

    # output = m0(input)  # (seq_len, batch, hidden_size)
    output, _ = m0(input)  # (seq_len, batch, hidden_size)

    loss = 0
    
    output = output.view(-1, output.size(2)) # （seq_len*batch, hidden_size)
    output = m1(output)  # (seq_len*batch, vocab_size)
    target = target.t().contiguous()  # (seq_len, batch)
    target = target.view(-1)  # (seq_len*batch)
    loss = lossF(output, target)

    return loss, output, target

def validate(valData, model, lossF, train_iteration):
    """
    valData (DataLoader)
    """
    if partition == 'new_partition':
        vocab_dist_cell = h5py.File('./experiment/porto-vocab-dist-cell75.h5', 'r')
    else:
        vocab_dist_cell = h5py.File('./experiment/porto-vocab-dist-cell100.h5', 'r')
    D = vocab_dist_cell['D']
    V = vocab_dist_cell['V']

    m0, m1 = model
    ## switch to evaluation mode
    m0.eval()
    m1.eval()

    sm = nn.Softmax(dim = 1)

    num_iteration = valData.size // batch
    if valData.size % batch > 0: num_iteration += 1

    total_genloss = 0
    in_acc = 0
    total_in_acc = 0

    # accuracy needed
    if (train_iteration+1) % 40000 == 0:
        for iteration in range(num_iteration):
            gendata = valData.getbatch_one()
            with torch.no_grad():
                genloss, output, target = genLoss(gendata, m0, m1, lossF)
                total_genloss += genloss.item() * gendata.trg.size(0)
                
                target = target.view(20, -1).t().contiguous()  # (batch, seq_len)
                out_seq = sm(output).argmax(dim = 1)  # (seq_len*batch) 
                out_seq = out_seq.view(20, -1).t().contiguous()  # (batch, seq_len)

                target = target.cpu()
                out_seq = out_seq.cpu()

                for i in range(batch):
                    trg_seq = del_zero(list(target[i]))
                    pre_seq = del_zero(list(out_seq[i]))
                    neighbours = get_neighbours(trg_seq, V)
                    pre_len = len(pre_seq)
                    if pre_len == 0:
                        continue
                    in_count = 0
                    for cell in pre_seq:
                        if cell in neighbours:
                            in_count += 1
                    in_acc = in_count/pre_len
                    total_in_acc += in_acc
        in_acc = total_in_acc/(num_iteration*batch)

    else:
        for iteration in range(num_iteration):
            gendata = valData.getbatch_one()
            with torch.no_grad():
                genloss, _, _ = genLoss(gendata, m0, m1, lossF)
                total_genloss += genloss.item() * gendata.trg.size(0)

        in_acc = "NONE"

    ## switch back to training mode
    m0.train()
    m1.train()
    
    return total_genloss / valData.size, in_acc


def savecheckpoint(state, is_best):
    torch.save(state, checkpoint)
    # 这里修改了路径
    if is_best:
        shutil.copyfile(checkpoint, os.path.join(path, 'trained_models/best_model_{}_{}_{}_{}_{}_100000.pt'.format(embedding_model, str(embedding_size), str(num_layers), str(hidden_size), partition)))



def train(checkpoint):
    logging.basicConfig(filename="training.log", level=logging.INFO)
    """
    embedding_model: 't2vec', 'NeuTraj', 'SRN'
    embedding_size: 64, 128, 192, 256
    """


    trainsrc = './data/source/{}/train_{}.src'.format(embedding_model, str(embedding_size))
    print('trainsrc', trainsrc)
    traintrg = "./data/target/{}/train.trg".format(partition)
    trainData = DataLoader(trainsrc, traintrg, batch)
    print("Reading training data...")
    trainData.load()

    valsrc = './data/source/{}/val_{}.src'.format(embedding_model, str(embedding_size))
    print('valsrc:', valsrc)
    valtrg = "./data/target/{}/val.trg".format(partition)
    print("Reading validation data...")
    valData = DataLoader(valsrc, valtrg, batch, True)
    valData.load()

    # create criterion, model, optimizer
    criterion = nn.CrossEntropyLoss(reduction= 'mean')

    m0 = Decoder(embedding_size, hidden_size, num_layers, dropout)
    m1 = nn.Sequential(nn.Linear(hidden_size, vocab_size))
    
    if torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
        m1.cuda()
        criterion.cuda()

    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=0.01)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=0.01)

    if os.path.isfile(checkpoint):
        print("=> loading checkpoint '{}'".format(checkpoint))
        logging.info("Restore training @ {} {} {} {} {}".format(time.ctime(), embedding_model, embedding_size, num_layers, hidden_size))

        checkpoint = torch.load(checkpoint)
        start_iteration = checkpoint["iteration"]
        best_prec_loss = checkpoint["best_prec_loss"]
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        m1_optimizer.load_state_dict(checkpoint["m1_optimizer"])
    
    else:
        print('There is no checkpoint!')
        logging.info("Start training @ {} {} {} {} {}".format(time.ctime(), embedding_model, embedding_size, num_layers, hidden_size))
        start_iteration = 0
        best_prec_loss = float(inf)


    print("Iteration starts at {} "
          "and will end at {}".format(start_iteration, num_iteration-1))

    ## training
    for iteration in range(start_iteration, num_iteration):
        try:
            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()

            ## generative loss
            gendata = trainData.getbatch_one()
            genloss, _, _ = genLoss(gendata, m0, m1, criterion)

            ## compute the gradients
            genloss.backward()

            ## clip the gradients
            clip_grad_norm_(m0.parameters(), 5.0)
            clip_grad_norm_(m1.parameters(), 5.0)

            ## one step optimization
            m0_optimizer.step()
            m1_optimizer.step()

            ## average loss for one word
            avg_genloss = genloss.item()

            if iteration % print_freq == 0:
                print("Iteration: {}, avg_genloss: {}".format(iteration, avg_genloss))
                
                with open("inv loss", "a") as f:
                    f.writelines(str(avg_genloss)+"\n")


            if (iteration+1) % save_freq == 0 and iteration > 0:
                prec_loss, in_acc = validate(valData, (m0, m1), criterion, iteration)
                print("prec_loss:{}, best_prec_loss:{}".format(prec_loss, best_prec_loss))
                if prec_loss < best_prec_loss:
                    best_prec_loss = prec_loss
                    print("Best model with loss {} and in_acc {} at iteration {}".format(best_prec_loss, in_acc, iteration))
                    logging.info("Best model with loss {} in_acc {} at iteration {} @ {}"\
                                 .format(best_prec_loss, in_acc, iteration, time.ctime()))
                    is_best = True
                else:
                    is_best = False
                print("Saving the model at iteration {} validation loss {} and in_acc {}".format(iteration, prec_loss, in_acc))
                savecheckpoint({
                    "iteration": iteration+1,
                    "best_prec_loss": best_prec_loss,
                    "m0": m0.state_dict(),
                    "m1": m1.state_dict(),
                    "m0_optimizer": m0_optimizer.state_dict(),
                    "m1_optimizer": m1_optimizer.state_dict()
                }, is_best)
        except KeyboardInterrupt:
            break

train(checkpoint)
