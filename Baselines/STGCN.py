import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils import data, STGCNutility, recorde
from utils import loss_function, evaluation
from utils import STGCNlayers
import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from torch.optim import lr_scheduler


def train_and_test_STGCN(args):

    random_seed = args.random_seed
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_early_stop_epochs = args.num_early_stop_epochs
    lr = args.lr
    l2_reg = args.l2_reg
    used_best_model = args.used_best_model
    feat_file = args.feat_file
    adj_file = args.adj_file
    seq_length = args.seq_length
    pre_length = args.pre_length

    # GPU加速
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    #初始化随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. 读取数据
    adj, n_vertex, train_loader, val_loader, test_loader, max_value, min_value = data.preprocess_data().generate_data(args)
    criterion = loss_function.MSELossWithL2(reg_lambda = l2_reg)

    # 2. 定义模型
    class STGCNChebGraphConv(nn.Module):
        def __init__(self, args, blocks, n_vertex):
            super(STGCNChebGraphConv, self).__init__()
            modules = []
            for l in range(len(blocks) - 3):
                modules.append(STGCNlayers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
            self.st_blocks = nn.Sequential(*modules)
            Ko = args.seq_length - (len(blocks) - 3) * 2 * (args.Kt - 1)
            self.Ko = Ko
            if self.Ko > 1:
                self.output = STGCNlayers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
            elif self.Ko == 0:
                self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
                self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
                self.relu = nn.ReLU()
                self.leaky_relu = nn.LeakyReLU()
                self.silu = nn.SiLU()
                self.dropout = nn.Dropout(p=args.droprate)

        def forward(self, x):
            x = self.st_blocks(x)
            if self.Ko > 1:
                x = self.output(x)
            elif self.Ko == 0:
                x = self.fc1(x.permute(0, 2, 3, 1))
                x = self.relu(x)
                x = self.fc2(x).permute(0, 3, 1, 2)
            
            return x.view(len(x), -1) 

    class STGCNGraphConv(nn.Module):
        def __init__(self, args, blocks, n_vertex):
            super(STGCNGraphConv, self).__init__()
            modules = []
            for l in range(len(blocks) - 3):
                modules.append(STGCNlayers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
            self.st_blocks = nn.Sequential(*modules)
            Ko = args.seq_length - (len(blocks) - 3) * 2 * (args.Kt - 1)
            self.Ko = Ko
            if self.Ko > 1:
                self.output = STGCNlayers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
            elif self.Ko == 0:
                self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
                self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
                self.relu = nn.ReLU()
                self.leaky_relu = nn.LeakyReLU()
                self.silu = nn.SiLU()
                self.do = nn.Dropout(p=args.droprate)

        def forward(self, x):
            x = self.st_blocks(x)
            if self.Ko > 1:
                x = self.output(x)
            elif self.Ko == 0:
                x = self.fc1(x.permute(0, 2, 3, 1))
                x = self.relu(x)
                x = self.fc2(x).permute(0, 3, 1, 2)
            
            return x.view(len(x), -1) 

    def get_parameters(args):
        args.gso_type = 'sym_norm_lap'
        args.Kt = 3
        args.stblock_num = 2
        args.act_func = 'glu'
        args.Ks = 3
        args.graph_conv_type = 'cheb_graph_conv'
        args.enable_bias = True
        args.droprate = 0.5
        args.step_size = 10

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        Ko = args.seq_length - (args.Kt - 1) * 2 * args.stblock_num

        # blocks: settings of channel size in st_conv_blocks and output layer,
        # using the bottleneck design in st_conv_blocks
        blocks = []
        blocks.append([1])
        for l in range(args.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([1])

        
        return args, device, blocks

    def prepare_model(args, blocks, n_vertex):

        if args.graph_conv_type == 'cheb_graph_conv':
            model = STGCNChebGraphConv(args, blocks, n_vertex).to(device)
        else:
            model = STGCNGraphConv(args, blocks, n_vertex).to(device)
        return model

    args, device, blocks = get_parameters(args)
    gso = STGCNutility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = STGCNutility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    # 创建一个STGCN模型实例
    model = prepare_model(args, blocks, n_vertex)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if args.lr_scheduler == True:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)
    else:
        scheduler = None

    # #自动生成一个log文件,如果没有,则自动创建
    log_filename = parent_dir + '/logs/' + args.model + '.log'
    logger = recorde.init_logger(args, log_filename)
    # logger.info('model_struture: %s', args.model)

    # 3. 训练模型
    if args.mode == 'train':
        print('Start Training STGCN Model!')
        best_model, model = evaluation.train_val_seq_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, num_early_stop_epochs, max_value, min_value, scheduler, logger)
        print('Finished Training')

        if args.save_model == True:
            torch.save(best_model, parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl')
            logger.info('save_path %s', parent_dir + '/model/' + args.model + args.feat_file + str(args.pre_length) + '.pkl')
            print('Finished Saving Model')
        
        rmse, mae, wmape = evaluation.test_seq_model(used_best_model, best_model, model, test_loader, max_value, min_value, logger)
        print('Finished Testing')
        # evaluation.write_results(args, rmse, mae, wmape)
    
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














