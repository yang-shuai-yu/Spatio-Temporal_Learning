import torch
import os, sys, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

def train(args, trainLoader, valLoader, se_model,
          optimizer, criterion, device):
    if device == 'gpu':
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')    # 这里我没写好，只能手动调整用哪一块gpu
    else:
        device = torch.device('cpu')

    max_train_loss = 0x3f3f3f3f; max_val_loss = 0x3f3f3f3f
    se_model.to(device)
    se_model.train()
    for epoch in range(args.epochs):
        for i, (emb, se_cells) in enumerate(trainLoader):
            emb = emb.to(device)
            se_cells = se_cells.to(device)
            se_cells = torch.cat((se_cells[:, 0], se_cells[:, 1]), 1)
            optimizer.zero_grad()
            se_output = se_model(emb)
            se_loss = criterion(se_output, se_cells)
            se_loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('se_outputs, labels ', se_output[0], se_cells[0], se_output[0] - se_cells[0])
                print("Epoch: {}/{}, Iteration: {}/{}, Loss: {}".format(epoch, args.epochs, i, len(trainLoader), se_loss.item()))
            if se_loss.item() < max_train_loss:
                max_train_loss = se_loss.item()
                torch.save(se_model.state_dict(), './models/d0_001_train_sePoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim))
        if epoch % 10 == 0:
            # validation
            se_model.eval()
            val_loss = 0
            iter_num = 0
            with torch.no_grad():
                for i, (emb, se_cells) in enumerate(valLoader):
                    emb = emb.to(device)
                    se_cells = se_cells.to(device)
                    se_cells = torch.cat((se_cells[:, 0], se_cells[:, 1]), 1)
                    se_output = se_model(emb)
                    se_loss = criterion(se_output, se_cells)
                    val_loss += se_loss.item()
                    iter_num += 1
            val_loss /= iter_num
            print('se_outputs, labels ', se_output[0], se_cells[0], se_output[0] - se_cells[0])
            print("Epoch: {}/{}, Validation Loss: {}".format(epoch, args.epochs, val_loss))
            if val_loss < max_val_loss:
                max_val_loss = val_loss
                torch.save(se_model.state_dict(), './models/d0_001_val_sePoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim))
            se_model.train()


# def train(args, trainLoader, valLoader, s_model, e_model, 
#           optimizer1, optimizer2, criterion1, criterion2, device):
#     if device == 'gpu':
#         device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device('cpu')

#     max_train_loss = 0x3f3f3f3f; max_val_loss = 0x3f3f3f3f
#     s_model.to(device); e_model.to(device)
#     s_model.train(); e_model.train()
#     for epoch in range(args.epochs):
#         for i, (emb, se_cells) in enumerate(trainLoader):
#             emb = emb.to(device)
#             se_cells = se_cells.to(device)
#             optimizer1.zero_grad(); optimizer2.zero_grad()
#             s_output = s_model(emb); e_output = e_model(emb)
#             s_loss = criterion1(s_output, se_cells[:, 0])
#             e_loss = criterion2(e_output, se_cells[:, 1])
#             s_loss.backward(); e_loss.backward()
#             optimizer1.step(); optimizer2.step()
#             if i % 100 == 0:
#                 print("Epoch: {}/{}, Iteration: {}/{}, Loss: {},{}".format(epoch, args.epochs, i, len(trainLoader), s_loss.item(), e_loss.item()))
#             if s_loss.item() + e_loss.item() < max_train_loss:
#                 max_train_loss = s_loss.item() + e_loss.item()
#                 torch.save(s_model.state_dict(), './models/d0_001_train_sPoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim))
#                 torch.save(e_model.state_dict(), './models/d0_001_train_ePoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim))
#         if epoch % 10 == 0:
#             # validation
#             s_model.eval(); e_model.eval()
#             val_loss = 0
#             iter_num = 0
#             with torch.no_grad():
#                 for i, (emb, se_cells) in enumerate(valLoader):
#                     emb = emb.to(device)
#                     se_cells = se_cells.to(device)
#                     s_output = s_model(emb); e_output = e_model(emb)
#                     s_loss = criterion1(s_output, se_cells[:, 0])
#                     e_loss = criterion2(e_output, se_cells[:, 1])
#                     val_loss += s_loss.item() + e_loss.item()
#                     iter_num += 1
#             val_loss /= iter_num
#             print("Epoch: {}/{}, Validation Loss1,2:{},{}".format(epoch, args.epochs, s_loss.item(), e_loss.item()))
#             print("Epoch: {}/{}, Validation Loss: {}".format(epoch, args.epochs, val_loss))
#             if val_loss < max_val_loss:
#                 max_val_loss = val_loss
#                 torch.save(s_model.state_dict(), './models/d0_001_val_sPoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim))
#                 torch.save(e_model.state_dict(), './models/d0_001_val_ePoints_{}_{}_{}.pth'.format(args.model, args.emb_dim, args.hidden_dim))
#             s_model.train(); e_model.train()