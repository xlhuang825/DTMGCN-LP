import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
import logging.handlers
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils import data, recorde
from utils import loss_function, evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import lr_scheduler

def train_and_test_GCN(args):
    # GPU加速
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # 读取参数
    model = args.model
    num_epochs = args.num_epochs
    num_early_stop_epochs = args.num_early_stop_epochs
    hidden_size = args.hidden_size
    used_best_model = args.used_best_model
    random_seed = args.random_seed
    feat_file = args.feat_file
    adj_file = args.adj_file
    seq_length = args.seq_length
    pre_length = args.pre_length
    batch_size = args.batch_size
    lr = args.lr
    l2_reg = args.l2_reg


    # 1. 读取数据
    train_loader, val_loader, test_loader, max_value, min_value = data.preprocess_data().generate_data(args)

    num_features = seq_length
    out_features = pre_length
    # 2. 构建模型

    #初始化随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 定义 GCN 模型
    class GCNRegression(nn.Module):
        def __init__(self, num_features, hidden_size, out_features):
            super(GCNRegression, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_size)
            self.linear = torch.nn.Linear(hidden_size, out_features)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = self.linear(x).squeeze(1)
            return x

    # 初始化模型
    model = GCNRegression(num_features = num_features, hidden_size = hidden_size, out_features = out_features).to(device)
    
    #criterion = nn.MSELoss()
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
        # 3. 训练模型
        print('Start Training GCN Model!')
        best_model, model = evaluation.train_val_graph_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, num_early_stop_epochs, max_value, min_value, scheduler, logger)
        print('Finished Training')
        
        if args.save_model == True:
            torch.save(best_model, parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl')
            logger.info('save_path %s', parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl')
            print('Finished Saving Model')
        
        rmse, mae, wmape = evaluation.test_graph_model(used_best_model, best_model, model, test_loader, max_value, min_value, logger)
        print('Finished Testing')
        # evaluation.write_results(args, rmse, mae, wmape)
    
    else:
        print('Just Testing')
        used_best_model = True
        path = parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl'
        logger.info('load_model_from : %s', path)
        
        try:
            best_model = torch.load(path)
            rmse, mae, wmape = evaluation.test_graph_model(used_best_model, best_model, model, test_loader, max_value, min_value, logger)
            print('Finished Testing')
            # evaluation.write_results(args, rmse, mae, wmape)
        
        except:
            print('No model found, please train first!')
