import pandas as pd
import numpy as np
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils import data, evaluation, recorde
import torch


def train_and_test_HA(args):
    
    model = args.model
    feat_file = args.feat_file
    adj_file = args.adj_file
    seq_length = args.seq_length
    pre_length = args.pre_length
    batch_size = args.batch_size
    random_seed = args.random_seed

    # #自动生成一个log文件,如果没有,则自动创建
    log_filename = parent_dir + '/logs/' + args.model + '.log'
    logger = recorde.init_logger(args, log_filename)
    # logger.info('model_struture: %s', model)

    # 1. 读取数据
    train_loader, val_loader, test_loader, max_value, min_value = data.preprocess_data().generate_data(args)
    logger.info('Testing...')
    print('Start Testing HA Model!')
    # 2. 使用HA模型进行预测
    RMSE = []
    MAE = []
    WMAPE = []
    for i, (inputs, labels) in enumerate(test_loader):
        outputs_list = []
        outputs = inputs.mean(dim=2)
        for j in range(pre_length):
            outputs_list.append(outputs)
        outputs = torch.stack(outputs_list, dim=2)
        rmse, mae, wmape = evaluation.test(outputs, labels, max_value, min_value)
        RMSE.append(rmse)
        MAE.append(mae)
        WMAPE.append(wmape)
    rmse = sum(RMSE) / len(RMSE)
    mae = sum(MAE) / len(MAE)
    wmape = sum(WMAPE) / len(WMAPE)
    print(f"RMSE: {rmse:.4f}",f"MAE: {mae:.4f}",f"WMAPE: {wmape:.4f}%")
    logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f},  WMAPE: {wmape:.4f}%")
    # evaluation.write_results(args, rmse, mae, wmape)
    print('Finished Testing')


