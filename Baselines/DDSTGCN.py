import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils import data, recorde
from utils import loss_function, evaluation
from utils.DDSTGCNmodel import ddstgcn as Network
import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
from torch.optim import lr_scheduler
import scipy.sparse as sp

def train_and_test_DDSTGCN(args):

    def sym_adj(adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    
    def load_adj(adj_mx):
        adj = [sym_adj(adj_mx), sym_adj(np.transpose(adj_mx))]
        return adj
    
    def load_hadj(adj_mx,top_k):

        hadj = adj_mx

        top = top_k

        hadj = hadj - np.identity(hadj.shape[0])
        hadj = torch.from_numpy(hadj.astype(np.float32))
        _, idx = torch.topk(hadj, top, dim=0)
        _, idy = torch.topk(hadj, top, dim=1)

        base_mx_lie = torch.zeros([hadj.shape[0], hadj.shape[1]])
        for i in range(hadj.shape[0]):
            base_mx_lie[idx[:, i], i] = hadj[idx[:, i], i]
        base_mx_hang = torch.zeros([hadj.shape[0], hadj.shape[1]])
        for j in range(hadj.shape[0]):
            base_mx_hang[j, idy[j, :]] = hadj[j, idy[j, :]]

        base_mx = torch.where(base_mx_lie != 0, base_mx_lie, base_mx_hang)

        hadj = base_mx + torch.eye(hadj.shape[0])
        hadj = hadj.numpy()

        n = hadj.shape[0]
        l = int((len(np.nonzero(hadj)[0])))
        H = np.zeros((l, n))
        H_a = np.zeros((l, n))
        H_b = np.zeros((l, n))
        lwjl = np.zeros((l,1))
        a=0

        for i in range(hadj.shape[0]):
            for j in range(hadj.shape[1]):
                if(hadj[i][j]!=0.0):
                    H[a, i] = 1.0
                    H[a, j] = 1.0
                    H_a[a, i] = 1.0
                    H_b[a, j] = 1.0
                    if(i==j):
                        lwjl[a, 0] = 1.0
                    else:
                        lwjl[a,0] = adj_mx[i,j]
                    a = a+1

        lwjl = 1.0-lwjl

        W = np.ones(n)

        DV = np.sum(H * W, axis=1)
        DE = np.sum(H, axis=0)
        DE_=np.power(DE, -1)
        DE_[np.isinf(DE_)] = 0.
        invDE = np.mat(np.diag(DE_))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        HT = sp.coo_matrix(HT)
        rowsum = np.array(HT.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        H_T_new = d_mat.dot(HT).astype(np.float32).todense()

        G0 = DV2 * H
        G1 = invDE * HT * DV2

        n = adj_mx.shape[0]
        l = int((len(np.nonzero(adj_mx)[0])))
        H_all = np.zeros((l, n))
        edge_1 = np.array([])
        edge_2 = np.array([])
        a=0

        for i in range(adj_mx.shape[0]):
            for j in range(adj_mx.shape[1]):
                if(adj_mx[i][j]!=0.0):
                    H_all[a, i] = 1.0
                    H_all[a, j] = 1.0
                    edge_1 = np.hstack((edge_1, np.array([i])))
                    edge_2 = np.hstack((edge_2, np.array([j])))
                    a = a+1

        W_all = np.ones(n)
        DV_all = np.sum(H_all * W_all, axis=1)
        DE_all = np.sum(H_all, axis=0)

        DE__all=np.power(DE_all, -1)
        DE__all[np.isinf(DE__all)] = 0.
        invDE_all = np.mat(np.diag(DE__all))
        DV2_all = np.mat(np.diag(np.power(DV_all, -0.5)))
        W_all = np.mat(np.diag(W_all))
        H_all = np.mat(H_all)
        HT_all = H_all.T

        HT_all = sp.coo_matrix(HT_all)
        rowsum_all = np.array(HT_all.sum(1)).flatten()
        d_inv_all = np.power(rowsum_all, -1).flatten()
        d_inv_all[np.isinf(d_inv_all)] = 0.
        d_mat_all = sp.diags(d_inv_all)
        H_T_new_all = d_mat_all.dot(HT_all).astype(np.float32).todense()

        G0_all = DV2_all * H_all
        G1_all = invDE_all * HT_all * DV2_all

        coo_hadj = adj_mx - np.identity(n)
        coo_hadj = sp.coo_matrix(coo_hadj)
        coo_hadj = coo_hadj.tocoo().astype(np.float32)

        indices = torch.from_numpy(np.vstack((edge_1, edge_2)).astype(np.int64))

        G0 = G0.astype(np.float32)
        G1 = G1.astype(np.float32)
        H = H.astype(np.float32)
        HT = H.T.astype(np.float32)
        H_T_new = torch.from_numpy(H_T_new.astype(np.float32))
        H_a = torch.from_numpy(H_a.astype(np.float32))
        H_b = torch.from_numpy(H_b.astype(np.float32))
        lwjl = torch.from_numpy(lwjl.astype(np.float32))

        G0_all = G0_all.astype(np.float32)
        G1_all = G1_all.astype(np.float32)

        return H_a, H_b, HT, lwjl ,G0,G1,indices, G0_all,G1_all

        # GPU加速
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    random_seed = args.random_seed
    feat_file = args.feat_file
    adj_file = args.adj_file
    seq_length = args.seq_length
    pre_length = args.pre_length
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    num_early_stop_epochs = args.num_early_stop_epochs
    l2_reg = args.l2_reg
    used_best_model = args.used_best_model
    args.top_k = 4
    

    #初始化随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. 读取数据
    adj, train_loader, val_loader, test_loader, max_value, min_value = data.preprocess_data().generate_data(args)
    num_nodes = adj.shape[0]

    adj_mx = load_adj(adj)
    supports = [torch.tensor(i).cuda() for i in adj_mx]
    H_a, H_b, H_T_new, lwjl, G0, G1, indices, G0_all, G1_all = load_hadj(adj, args.top_k)
    lwjl = (((lwjl.t()).unsqueeze(0)).unsqueeze(3)).repeat(args.batch_size, 1, 1, 1)
    H_a = H_a.cuda()
    H_b = H_b.cuda()
    G0 = torch.tensor(G0).cuda()
    G1 = torch.tensor(G1).cuda()
    H_T_new = torch.tensor(H_T_new).cuda()
    lwjl = lwjl.cuda()
    indices = indices.cuda()

    G0_all = torch.tensor(G0_all).cuda()
    G1_all = torch.tensor(G1_all).cuda()

    model = Network(batch_size, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl, num_nodes,
                 dropout = 0.3, supports = supports, in_dim = 1, out_dim = pre_length, residual_channels = 40, dilation_channels = 40,
                 skip_channels=320, end_channels = 640, kernel_size = 2, blocks = 3, layers=1)
    model = model.to(device)

    # 3. 定义损失函数和优化器
    criterion = loss_function.MSELossWithL2(reg_lambda = l2_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if args.lr_scheduler == True:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)
    else:
        scheduler = None

    # #自动生成一个log文件,如果没有,则自动创建
    log_filename = parent_dir + '/logs/' + args.model + '.log'
    logger = recorde.init_logger(args, log_filename)

    if args.mode == 'train':
        # 4. 训练模型
        print('Start Training DDSTGCN Model!')
        best_model, model = evaluation.train_val_seq_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, num_early_stop_epochs, max_value, min_value, scheduler, logger)
        print('Finished Training')
        if args.save_model == True:
            torch.save(best_model, parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl')
            logger.info('save_path %s', parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl')
            print('Finished Saving Model')
        
        rmse, mae, wmape = evaluation.test_seq_model(used_best_model, best_model, model, test_loader, max_value, min_value, logger)
        print('Finished Testing')
        # evaluation.write_results(args, rmse, mae, mape, smape)

    else:
        print('Just Testing')
        used_best_model = True
        path = parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl'
        logger.info('load_model_from : %s', path)
        
        try:
            best_model = torch.load(path)
            rmse, mae, wmape = evaluation.test_seq_model(used_best_model, best_model, model, test_loader, max_value, min_value, logger)
            print('Finished Testing')
            # evaluation.write_results(args, rmse, mae, mape, smape)
        
        except:
            print('No model found, please train first!')
