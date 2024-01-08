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
from utils.DDGCRNmodel import DDGCRN as Network
import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import lr_scheduler

def train_and_test_DDGCRN(args):
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

    #初始化随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. 读取数据
    train_loader, val_loader, test_loader, max_value, min_value = data.preprocess_data().generate_data(args)

    if args.feat_file == 'sz_speed.csv':
        time_interval = 15
    else:
        time_interval = 5
    # numerical time_in_day
    args.T = (60 / time_interval) * 24
    args.num_nodes = train_loader.dataset[0][0].shape[1]
    args.input_dim = 1
    args.output_dim = 1
    args.rnn_units = 64
    args.num_layers = 1
    args.embed_dim = 10
    args.use_day = True
    args.use_week = True
    args.default_graph = True
    args.cheb_k = 2

    model = Network(args)
    model = model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)   

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
    # logger.info('model_struture: %s', model)

    if args.mode == 'train':
        # 4. 训练模型
        print('Start Training DDGCRN Model!')
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