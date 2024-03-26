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

parser.add_argument("-batch_size", default= 64, type= int,
    help="batch size")

parser.add_argument("-num_layers", type=int, default=3,
    help="Number of layers in the MLPs")

parser.add_argument("-embedding_size", type=int, default=128,
    help="The input embeddings' size")

parser.add_argument("-hidden_size", type=int, default=256,
    help="The hidden size of hidden layers.")

parser.add_argument("-embedding_model", default="neutraj_128",
    help="The attacked model.")

parser.add_argument("-epoch", default=200, type =int,    # 原本设置epoch为500,但据我观察200epoch其实就差不多了
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

def validate(loader, model, criterion, epoch):
    """
    valData (DataLoader)
    """
    ## switch to evaluation mode
    
    model.eval()

    avg_loss = 0
    acc = 0

    total_loss = 0
    total_num = 0
    total_wrong = 0
    total_times = 0
    for step, (embeddings, labels) in enumerate(loader):
        
        if torch.cuda.is_available():
            embeddings  = embeddings.cuda() # (batch, embedding_size)
            labels = labels.cuda()

        step_loss, output, _ = genLoss(embeddings, labels, model, criterion)
        total_loss += (step_loss * len(embeddings))
        total_num += embeddings.size(0)

        output = output.round()
        labels = labels.view(labels.size(0), labels.size(1), 1)

        gap = torch.abs(output-labels) # (bacth, num_clusters, 1)
        wrong_num = gap.sum()
        total_wrong += wrong_num
        total_times += labels.size(0) * labels.size(1)

    avg_loss = total_loss/total_num
    acc = (total_times-total_wrong)/total_times*100

    print("epoch: {}, avg_loss: {}, acc: {}".format(epoch, avg_loss, acc))
    ## switch back to training mode
    model.train()
    return avg_loss, acc

def evaluate(args, name_id = -1):
    dataPath = args.data
    embedding_model = args.embedding_model
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_clusters = 97
    batch_size = args.batch_size
    radius = args.radius
    if name_id == -1:
        print("The name_id is not specified!")
        best_model = "./models/best_model_{}_{}.pt".format(radius, embedding_model)
    else:
        best_model = "./results/{}_training/best_model_{}_{}.pt".format(name_id, radius, embedding_model)

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
            predLabel = output.round()
            results[i*batch_size:(i+1)*batch_size] = predLabel.cpu().numpy()
    
    # 为了防止报错，这里我先给改成了这样，冗余代码量很大
    if name_id == -1:
        with open("./data/pred_test_{}_{}".format(radius, embedding_model), "wb") as f:
            pickle.dump(results, f)
    else:
        with open("./results/{}_training/pred_test_{}_{}".format(name_id, radius, embedding_model), "wb") as f:
            pickle.dump(results, f)

def train(args, name_id):
    ## hyperparameters
    os.makedirs("./results/{}_training".format(name_id), exist_ok=True)
    logging.basicConfig(filename="./results/{}_training/log.txt".format(name_id), level=logging.INFO)
                                                                                    
    dataPath = args.data
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    embedding_model = args.embedding_model
    epoch = args.epoch
    radius = args.radius
    batch_size = args.batch_size
    # checkpoint_path = "./models/checkpoint_{}_{}.pt".format(radius, embedding_model)
    checkpoint_path = "./results/{}_training/checkpoint_{}_{}.pt".format(name_id, radius, embedding_model)

    logging.info("The checkpoint is saved at {}".format(checkpoint_path))
    logging.info("Other hyperparameters: epoch: {}, radius: {}, batch_size: {}".format(epoch, radius, batch_size))

    with open(dataPath + embedding_model +"_train", "rb") as f:
        embeddings = pickle.load(f)
    with open(dataPath + "r{}_label_train".format(radius), "rb") as f:
        labels = pickle.load(f)
    
    with open(dataPath + embedding_model +"_val", "rb") as f:
        val_embeddings = pickle.load(f)
    with open(dataPath + "/r{}_label_val".format(radius), "rb") as f:
        val_labels = pickle.load(f)
    
    embeddings = torch.tensor(embeddings[0:100000])
    labels = torch.tensor(labels[0:100000])

    val_embeddings = torch.tensor(val_embeddings)
    val_labels = torch.tensor(val_labels)

    # print(labels.shape)
    num_clusters = labels.shape[1]
    # print(labels.shape)
    print(embeddings.shape)
    print(labels.shape)
    dataset = Data.TensorDataset(embeddings, labels)
    val_dataset = Data.TensorDataset(val_embeddings, val_labels)

    dataLoader = Data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)
    val_dataLoader = Data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)

    model = StackingMLP(embedding_size, hidden_size, num_clusters, num_layers)
    criterion = nn.BCELoss(reduction="mean")

    if torch.cuda.is_available():
        print("=> training with GPU")
        model.cuda()
        criterion.cuda()
    else:
        print("=> training with CPU")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        logging.info("Restore training @ {} {} {} {} {}".format(time.ctime(), embedding_model, embedding_size, num_layers, hidden_size))

        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"] + 1
        avg_loss = checkpoint["avg_loss"]
        best_prec_loss = checkpoint["best_prec_loss"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    else:
        print('There is no checkpoint!')
        logging.info("Start training @ {} {} {} {} {}".format(time.ctime(), embedding_model, embedding_size, num_layers, hidden_size))
        start_epoch = 0
        best_prec_loss = float(inf)
    
    print("Iteration starts at {} "
          "and will end at {}".format(start_epoch, epoch-1))

    ## training
    for i in range(start_epoch, epoch):
        try:
            optimizer.zero_grad()

            for step, (embeddings, labels) in enumerate(dataLoader):
                loss, _, _ = genLoss(embeddings, labels, model, criterion)

                # print(step, loss)
                loss.backward()

                # # clip the gradients
                # clip_grad_norm_(model.parameters(), 5.0)

                ## one step optimization
                optimizer.step()

                # average loss for one word
                avg_loss = loss.item()

                if step % 1000 == 0:
                    print("Step: {}, avg_genloss: {}".format(step, avg_loss))

            avg_loss, acc = validate(val_dataLoader, model, criterion, i)
            
            if avg_loss < best_prec_loss:
                best_prec_loss = avg_loss
                print("Best model with loss {} and in_acc {} at epoch {}".format(best_prec_loss, acc, i))
                logging.info("Best model with loss {} in_acc {} at epoch {} @ {}"\
                                 .format(best_prec_loss, acc, i, time.ctime()))
                is_best = True
            else:
                is_best = False
            
            print("Saving the model at epoch {} validation loss {} and acc {}".format(i, avg_loss, acc))

            state = {"epoch": i,
                    "avg_loss": avg_loss,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_prec_loss": best_prec_loss
                    }

            # 其实这里相当于保存了双份的model，一份是checkpoint，一份是best_model
            torch.save(state, checkpoint_path)
            if is_best:
                shutil.copyfile(checkpoint_path, "./models/best_model_{}_{}.pt".format(radius, embedding_model))
                # added by me
                torch.save(state, "./results/{}_training/best_model_{}_{}.pt".format(name_id, radius, embedding_model))
        except KeyboardInterrupt:
            break

    # evaluating after training
    evaluate(args, name_id)


if(args.mode == "train"):
    name_id = "20231129_7"
    train(args, name_id)
elif(args.mode == "evaluate"):
    evaluate(args)