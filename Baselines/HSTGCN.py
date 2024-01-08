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
from utils.graph_conv import calculate_laplacian_with_self_loop
from utils import data, recorde
from utils import loss_function, evaluation
import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import lr_scheduler

def train_and_test_HSTGCN(args):
    # GPU加速
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 读取参数
    args = args
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


    #初始化随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # 1. 读取数据
    train_loader, val_loader, test_loader, max_value, min_value = data.preprocess_data().generate_data(args)

    # 2. 构建模型
    # 定义 TGCN 模型
    class TGCNGraphConvolution(nn.Module):
        def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
            super(TGCNGraphConvolution, self).__init__()
            self._num_gru_units = num_gru_units
            self._output_dim = output_dim
            self._bias_init_value = bias
            self.register_buffer(
                "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
            )
            self.weights = nn.Parameter(
                torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
            )
            self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weights)
            nn.init.constant_(self.biases, self._bias_init_value)

        def forward(self, inputs, hidden_state):
            batch_size, num_nodes = inputs.shape
            # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
            inputs = inputs.reshape((batch_size, num_nodes, 1))
            # hidden_state (batch_size, num_nodes, num_gru_units)
            hidden_state = hidden_state.reshape(
                (batch_size, num_nodes, self._num_gru_units)
            )
            # [x, h] (batch_size, num_nodes, num_gru_units + 1)
            concatenation = torch.cat((inputs, hidden_state), dim=2)
            # [x, h] (num_nodes, num_gru_units + 1, batch_size)
            concatenation = concatenation.transpose(0, 1).transpose(1, 2)
            # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
            concatenation = concatenation.reshape(
                (num_nodes, (self._num_gru_units + 1) * batch_size)
            )
            # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
            a_times_concat = self.laplacian @ concatenation
            # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
            a_times_concat = a_times_concat.reshape(
                (num_nodes, self._num_gru_units + 1, batch_size)
            )
            # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
            a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
            # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
            a_times_concat = a_times_concat.reshape(
                (batch_size * num_nodes, self._num_gru_units + 1)
            )
            # A[x, h]W + b (batch_size * num_nodes, output_dim)
            outputs = a_times_concat @ self.weights + self.biases
            # A[x, h]W + b (batch_size, num_nodes, output_dim)
            outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
            # A[x, h]W + b (batch_size, num_nodes * output_dim)
            outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
            return outputs

        @property
        def hyperparameters(self):
            return {
                "num_gru_units": self._num_gru_units,
                "output_dim": self._output_dim,
                "bias_init_value": self._bias_init_value,
            }


    class TGCNCell(nn.Module):
        def __init__(self, adj, input_dim: int, hidden_dim: int):
            super(TGCNCell, self).__init__()
            self._input_dim = input_dim
            self._hidden_dim = hidden_dim
            self.register_buffer("adj", torch.FloatTensor(adj))
            self.graph_conv1 = TGCNGraphConvolution(
                self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
            )
            self.graph_conv2 = TGCNGraphConvolution(
                self.adj, self._hidden_dim, self._hidden_dim
            )

        def forward(self, inputs, hidden_state):
            # [r, u] = sigmoid(A[x, h]W + b)
            # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
            concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
            # r (batch_size, num_nodes, num_gru_units)
            # u (batch_size, num_nodes, num_gru_units)
            r, u = torch.chunk(concatenation, chunks=2, dim=1)
            # c = tanh(A[x, (r * h)W + b])
            # c (batch_size, num_nodes * num_gru_units)
            c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
            # h := u * h + (1 - u) * c
            # h (batch_size, num_nodes * num_gru_units)
            new_hidden_state = u * hidden_state + (1.0 - u) * c
            return new_hidden_state, new_hidden_state

        @property
        def hyperparameters(self):
            return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


    class TGCN(nn.Module):
        def __init__(self, adj, hidden_dim: int, pre_length:int, **kwargs):
            super(TGCN, self).__init__()
            self._input_dim = adj.shape[0]
            self._hidden_dim = hidden_dim
            self.register_buffer("adj", torch.FloatTensor(adj))
            self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
            self.linear = nn.Linear(self._hidden_dim, pre_length)

        def forward(self, inputs):
            inputs = inputs.transpose(1, 2)
            batch_size, seq_len, num_nodes = inputs.shape
            assert self._input_dim == num_nodes
            hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
                inputs
            )
            output = None
            for i in range(seq_len):
                output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
                output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            output = self.linear(output)
            return output

        @staticmethod
        def add_model_specific_arguments(parent_parser):
            parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument("--hidden_dim", type=int, default=64)
            return parser

        @property
        def hyperparameters(self):
            return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

    # 读取adj.csv文件
    adj_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', adj_file)
    adj = pd.read_csv(adj_path, header = None).values
    
    # 创建一个TGCN模型实例
    model = TGCN(adj, hidden_dim = hidden_size, pre_length = pre_length)
    model.to(device)

    # 定义损失函数和优化器
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
        print('Start Training H-STGCN Model!')
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
            model = torch.load(path)
            rmse, mae, wmape = evaluation.test_seq_model(used_best_model, best_model, model, test_loader, max_value, min_value, logger)
            print('Finished Testing')
            # evaluation.write_results(args, rmse, mae, mape, smape)
        except:
            print('No model found, please train first!')


