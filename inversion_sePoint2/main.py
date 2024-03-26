import torch
import os, sys, argparse, logging, time
from DataLoader.dataloader import TrajDataset
from modules.models import xy_model
from train import train
from test import test

def get_args():
    parser = argparse.ArgumentParser(description='sePoints_Attack')
    parser.add_argument('--data_name', type=str, default='porto', help='dataset name')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--model', type=str, default='neutraj', help='embedding model name')
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--device', type=str, default='gpu', help='cpu or gpu')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("Current working directory: ", os.getcwd())
    logging.basicConfig(filename='d0_001_training.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    args = get_args()
    data_path = args.data_path;    emb_model = args.model
    emb_dim = args.emb_dim;   hid_dim = args.hidden_dim 
    logging.info("data_path: {}, emb_model: {}, emb_dim: {}, hid_dim: {}".format(data_path, emb_model, emb_dim, hid_dim))
    logging.info("the date of the experiment: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    
    print("Loading data...")
    trainDataset = TrajDataset(data_path, emb_model, emb_dim, hid_dim, mode = 'train')
    testDataset = TrajDataset(data_path, emb_model, emb_dim, hid_dim, mode = 'test')
    valDataset = TrajDataset(data_path, emb_model, emb_dim, hid_dim, mode = 'val')
    print("Loading finished!")
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = args.batch_size, shuffle = True)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size = args.batch_size, shuffle = True)
    valLoader = torch.utils.data.DataLoader(valDataset, batch_size = args.batch_size, shuffle = True)
    print("DataLoader finished!")

    if args.mode == 'train':
        print("Training...")
        sx_model = xy_model(emb_dim, hid_dim); sy_model = xy_model(emb_dim, hid_dim)
        ex_model = xy_model(emb_dim, hid_dim); ey_model = xy_model(emb_dim, hid_dim)

        optimizer1 = torch.optim.Adam(sx_model.parameters(), lr = args.lr, weight_decay = 0.0001); optimizer2 = torch.optim.Adam(sy_model.parameters(), lr = args.lr, weight_decay = 0.0001)
        optimizer3 = torch.optim.Adam(ex_model.parameters(), lr = args.lr, weight_decay = 0.0001); optimizer4 = torch.optim.Adam(ey_model.parameters(), lr = args.lr, weight_decay = 0.0001)

        models = [sx_model, sy_model, ex_model, ey_model]
        optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
        criterion = torch.nn.MSELoss(reduction = 'mean')
        train(args, trainLoader, valLoader, models, optimizers, criterion, device = args.device)
        logging.info("the date of the experiment: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    else:
        # print("Testing...")
        sx_model = xy_model(emb_dim, hid_dim); sy_model = xy_model(emb_dim, hid_dim)
        ex_model = xy_model(emb_dim, hid_dim); ey_model = xy_model(emb_dim, hid_dim)
        models = [sx_model, sy_model, ex_model, ey_model]
        criterion = torch.nn.MSELoss(reduction = 'mean')
        for i in range(len(models)):
            models[i].load_state_dict(torch.load('./models/d0_001_train_sePoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim)))
        test_loss = test(args, testLoader, models, criterion, device = args.device)
        logging.info("the date of the test experiment: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logging.info("test_loss: {}".format(test_loss))



        # se_model = se_net(emb_dim, hid_dim)
        # se_model.load_state_dict(torch.load('./models/d0_001_train_sePoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim)))
        # criterion = torch.nn.MSELoss(reduction = 'sum')
        # test_loss = test(args, testLoader, se_model, criterion, device = args.device)
        # logging.info("the date of the test experiment: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # logging.info("test_loss: {}".format(test_loss))


