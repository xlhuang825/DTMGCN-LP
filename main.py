import argparse
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from baseline.TGGNN import train_and_test_TGGNN 
from baseline.LSTM import train_and_test_LSTM
from baseline.GRU import train_and_test_GRU
from baseline.GCN import train_and_test_GCN
from baseline.TGCN import train_and_test_TGCN
from baseline.HA import train_and_test_HA
from baseline.ARIMA import train_and_test_ARIMA
from baseline.SVR import train_and_test_SVR
from baseline.STGCN import train_and_test_STGCN
from baseline.HSTGCN import train_and_test_HSTGCN
from baseline.GMAN import train_and_test_GMAN
from baseline.DGCRN import train_and_test_DGCRN
from baseline.DDGCRN import train_and_test_DDGCRN
from baseline.TMGCN import train_and_test_TMGCN
from baseline.OTGCN import train_and_test_OTGCN
from baseline.AGCRN import train_and_test_AGCRN
from baseline.DDSTGCN import train_and_test_DDSTGCN
from baseline.VAR import train_and_test_VAR
import time

if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Train and test model for traffic prediction')
    parser.add_argument('--model', type = str, default = 'DGCRN', help='model name')
    parser.add_argument('--feat_file', type = str, default = 'augsburg_flow.csv', choices=['london_flow.csv', 'augsburg_flow.csv', 'los_speed.csv'], help='path to the input file')
    # parser.add_argument('--adj_file', type = str, default = 'london_adj.csv', choices=['pemsd4_adj.csv', 'london_adj.csv', 'augsburg_adj.csv', 'pemsd4_s_adj.csv','pemsd3_adj.csv','pemsd3_s_adj.csv','sz_adj.csv', 'los_adj.csv','pemsd7_adj.csv'], help='path to the adjacency matrix file')
    parser.add_argument('--adj_file', type = str, default = 'augsburg_adj.csv', choices=['london_adj.csv', 'augsburg_adj.csv', 'los_adj.csv'], help='path to the adjacency matrix file')
    parser.add_argument('--seq_length', type = int, default = 12, help='sequence size of the prediction model')
    parser.add_argument('--pre_length', type = int, default = 12, help='prediction size of the prediction model')
    parser.add_argument('--random_seed', type = int, default = 2012, help='random seed')
    parser.add_argument('--batch_size', type = int, default = 32, help='batch size of the prediction model')
    parser.add_argument('--hidden_size', type = int, default = 64, help='hidden size of the prediction model')
    parser.add_argument('--lr', type = float, default = 0.01, help='learning rate of the optimizer')
    parser.add_argument('--num_epochs', type = int, default = 100, help='number of epochs to train the model')
    parser.add_argument('--num_early_stop_epochs', type = int, default = 10, help='number of epochs for early stopping')
    parser.add_argument('--l2_reg', type = float, default = 0.0001, help='l2 regularization')
    parser.add_argument('--used_best_model', type = bool, default = True, help='whether to use the best model')
    parser.add_argument('--train_ratio', type = float, default = 0.6, help='ratio of training data')
    parser.add_argument('--val_ratio', type = float, default = 0.2, help='ratio of validation data')
    parser.add_argument('--save_model', type = bool, default = False, choices=[True, False],help='whether to save the model')
    parser.add_argument('--mode', type = str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr_scheduler', type = bool, default = True, choices=[True, False], help='whether to use the learning rate scheduler')
    parser.add_argument('--step_size', type = int, default = 10, help='step size of the learning rate scheduler')
    parser.add_argument('--gamma', type = float, default = 0.1, help='gamma of the learning rate scheduler')
    parser.add_argument('--device', type = str, default = 'cuda', choices=['cuda', 'cpu'], help='device to use for training the model')
    args = parser.parse_args()
    
    if args.model == 'HA':
        train_and_test_HA(args = args)
    
    if args.model == 'ARIMA':
        train_and_test_ARIMA(args = args)

    if args.model == 'SVR':
        train_and_test_SVR(args = args)
    
    if args.model == 'GRU':
        train_and_test_GRU(args = args)

    if args.model == 'LSTM':
        train_and_test_LSTM(args = args)

    if args.model == 'GCN':
        train_and_test_GCN(args = args)
        
    if args.model == 'TGCN':
        train_and_test_TGCN(args = args)

    if args.model == 'OTGCN':
        train_and_test_OTGCN(args = args)

    if args.model == 'STGCN':
        train_and_test_STGCN(args = args)

    if args.model == 'H-STGCN':
        args.sc = 12 #5mins
        args.st = 3 #1days
        args.sp = 1  #1weeks
        train_and_test_HSTGCN(args = args)
    

    if args.model == 'TGGNN':
        train_and_test_TGGNN(args = args)

    if args.model == 'DGCRN':
        args.rnn_size = 64
        args.hyperGNN_dim = 64
        args.gcn_depth = 2
        args.dropout = 0.3
        args.subgraph_size = 20
        args.node_dim = 40
        args.in_dim = 2
        args.layers = 3
        args.tanhalpha = 3
        args.cl_decay_steps = 1000
        train_and_test_DGCRN(args = args)


    if args.model == 'DDGCRN':
        train_and_test_DDGCRN(args = args)

    if args.model == 'TMGCN':
        train_and_test_TMGCN(args = args)

    if args.model == 'DDSTGCN':
        train_and_test_DDSTGCN(args = args)
    
    
    end_time = time.time()
    print('Running time: %s Seconds'%(end_time-start_time))



