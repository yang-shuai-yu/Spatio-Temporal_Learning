import torch
import torch.nn as nn
from models import Decoder
from data_utils import DataLoader
import os, sys, pickle

os.chdir(sys.path[0])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_layers = 3
hidden_size = 256
batch = 100

embedding_model = 'NeuTrajat'
embedding_size = 256
vocab_size = 18866
dropout = 0.2

partition = 't2vec_partition'

if partition == 'new_partition':
    vocab_size = 21105

path = "/home/yangshuaiyu6791/RepresentAttack/inversion-attack-new"
exppath = "/home/yangshuaiyu6791/RepresentAttack/inversion-attack-new/experiment"
# 这里修改了路径
best_model = "./trained_models/best_model_{}_{}_{}_{}_{}_100000.pt".format(embedding_model, str(embedding_size), str(num_layers), str(hidden_size), partition)

def bb_inversion(best_model):

    m0 = Decoder(embedding_size, hidden_size, num_layers, dropout)
    m1 = nn.Sequential(nn.Linear(hidden_size, vocab_size))
    sm = nn.Softmax(dim = 1)

    expsrc = './data/source/{}/test_{}.src'.format(embedding_model, str(embedding_size))
    exptrg = './data/target/{}/test.trg'.format(partition)
    print("Reading experiment data...")
    expData = DataLoader(expsrc, exptrg, batch, True)
    expData.load()

    if os.path.isfile(best_model):
        print("=> loading best model '{}'".format(best_model))

        best_model = torch.load(best_model, map_location=torch.device('cpu'))
        
        m0.load_state_dict(best_model["m0"])
        m1.load_state_dict(best_model["m1"])

        if torch.cuda.is_available():
            m0.cuda()
            m1.cuda()
            sm.cuda()

        m0.eval()
        m1.eval()
        sm.eval()

        num_iteration = expData.size // batch

        test_embeddings = []

        # accuracy needed
        for iteration in range(num_iteration):
            gendata = expData.getbatch_one()
            with torch.no_grad():
                input, target = gendata.src, gendata.trg  # (batch, embedding_size), (batch, seq_len)
                if torch.cuda.is_available():
                    input, target = input.cuda(), target.cuda()

                output, _ = m0(input)  # (seq_len, batch, hidden_size)
                output = output.view(-1, output.size(2)) # （seq_len*batch, hidden_size)
                output = m1(output)  # (seq_len*batch, vocab_size)
                out_seq = sm(output).argmax(dim = 1)  # (seq_len*batch)

                target = target.t().contiguous()  # (seq_len, batch)
                target = target.view(-1)  # (seq_len*batch)
                
                out_seq = out_seq.view(20, -1).t().contiguous()
                target = target.view(20, -1).t().contiguous()
                
                for i in range(batch):
                    test_embeddings.append(out_seq[i].cpu().numpy())
        
        # 这里修改了路径
        with open('./experiment/inversions_{}_{}_{}_{}_{}_100000'.format(embedding_model, str(embedding_size), str(num_layers), str(hidden_size), partition), 'wb') as f:
            pickle.dump(test_embeddings, f)
        print('Inversion Finished!')

    else:
        print("=> There is no best_model found at '{}'".format(best_model))


bb_inversion(best_model)