from sklearn.utils import shuffle
from models import StackingMLP
import torch
import numpy as np
from numpy import inf
import torch.nn as nn
import torch.utils.data as Data
import pickle
import os, sys, argparse, time, shutil
import logging

os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="./data/",
    help="Path to training and validating data")

# parser.add_argument("-checkpoint", default="./models/checkpoint_300_t2vec_256.pt",
#     help="The saved checkpoint")

parser.add_argument("-batch_size", default= 50, type= int,
    help="batch size")

parser.add_argument("-num_layers", type=int, default=3,
    help="Number of layers in the MLPs")

parser.add_argument("-embedding_size", type=int, default=128,
    help="The input embeddings' size")

parser.add_argument("-hidden_size", type=int, default=128,
    help="The hidden size of hidden layers.")

parser.add_argument("-embedding_model", default="neutrajat_128",
    help="The attacked model.")

parser.add_argument("-epoch", default=500, type =int,
    help="The training epoch.")

parser.add_argument("-radius", default=300, type =int,
    help="The cluster radius.")

parser.add_argument("-mode", default = "train", help="Choose the mode")

# parser.add_argument("-best_model", default = "./models/best_model_300_t2vec_256.pt", help="The best mode")

args = parser.parse_args()

    

def genLoss(embeddings, labels, model, criterion):
    """
    One batch loss

    Input:
    gendata: a named tuple contains
        gendata.src (batch, embedding_size): input tensor
        gendata.trg (batch， seq_len): target tensor.
    
    m0: map input to output.
    m1: map the output of EncoderDecoder into the vocabulary space and do
        log transform.
    lossF: loss function.
    ---
    Output:
    loss
    """
     # (batch, embedding_size), (batch, seq_len)
    if torch.cuda.is_available():
        embeddings  = embeddings.cuda() # (batch, embedding_size)
        labels = labels.cuda()

    output = model(embeddings) # (batch, num_clusters)

    loss = 0
    
    output = output.view(output.size(0), output.size(1), 1) # (bacth, num_clusters, 1)
    labels = labels.view(labels.size(0), labels.size(1), 1) # (batch, num_clusters, 1)

    # print(output.dtype, labels.dtype)
    loss = criterion(output.float(), labels.float())

    return loss, output, labels

def evaluate(args):
    dataPath = args.data
    embedding_model = args.embedding_model
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_clusters = 97
    batch_size = args.batch_size
    radius = args.radius
    best_model = "./models/best_model_{}_{}.pt".format(radius, embedding_model)

    with open(dataPath + embedding_model +"_test", "rb") as f:
        embeddings = pickle.load(f)
    
    embeddings = torch.tensor(embeddings)
    
    model = StackingMLP(embedding_size, hidden_size, num_clusters, num_layers)

    if torch.cuda.is_available():
        embeddings  = embeddings.cuda() # (embedding_size)
        model.cuda()

    if os.path.isfile(best_model):
        print("=> loading best_model '{}'".format(best_model))
        best_model = torch.load(best_model, map_location=torch.device('cpu'))
        model.load_state_dict(best_model["model"])
    else:
        print('There is no best model!')
    
    model.eval()

    iteration = int(len(embeddings)/batch_size)
    results = np.zeros((len(embeddings), num_clusters))

    with torch.no_grad():
        for i in range(iteration):
            data = embeddings[i*batch_size:(i+1)*batch_size]
            output = model(data) # (batch, num_clusters)
            results[i*batch_size:(i+1)*batch_size] = output.cpu().numpy()
    
    print(results[0])

evaluate(args)