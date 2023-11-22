import numpy as np
import scipy.linalg as slin
import networkx as nx
import timeit
import numpy as np
# import tensorflow as tf
import torch 
torch.cuda.empty_cache()
import torch.nn as nn 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from sklearn import preprocessing
import copy 
import sys
sys.path.append(os.path.abspath('../benchmark_data/'))
sys.path.append(os.path.abspath('../'))
from BNGPU import NOBEARS_torch
from time import time
import benchmark_data_reader
from BNGPU.NOBEARS_torch import GolemTorch
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import torch.utils.data as utils
import yaml
from torch.utils.data import Subset
from torch._utils import _accumulate
import random 

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


if __name__ == '__main__':
    with open("config/abcd.yaml", 'r') as f:     
        config = yaml.load(f, Loader=yaml.Loader)
    benreader = benchmark_data_reader.BenchmarkReader()
    res = []
    lambda_1 = config["train"]["lambda_1"]
    lambda_2 = config["train"]["lambda_2"]
    lambda_3 = config["train"]["lambda_3"]
    lambda_init = config["train"]["lambda_init"]
    path = config["path"]
    pos_weight = 0.1
    neg_weight = 0.1
    batch_size = 32
    seed = config["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = f"cuda:{config['train']['gpu']}"
    equal_variances = True 
    save_dir = "results/"
    os.makedirs(save_dir, exist_ok = True)

    for i in benreader.get_dataset_name():    
        fc_data = np.load(config["data"]["time_series"], allow_pickle=True)
        pearson_data = np.load(config["data"]["pathnode_feature"], allow_pickle=True)
        label_df = pd.read_csv(config["data"]["label"])
        
        with open(config["data"]["node_id"], 'r') as f:
            lines = f.readlines()
            pearson_id = [line[:-1] for line in lines]
       
        with open(config["data"]["series_id"], 'r') as f:
            lines = f.readlines()
            fc_id = [line[:-1] for line in lines]

        id2pearson = dict(zip(pearson_id, pearson_data))

        id2gender = dict(zip(label_df['id'], label_df['sex']))

        final_fc, final_label, final_pearson = [], [], []

        for fc, l in zip(fc_data, fc_id):
            if l in id2gender and l in id2pearson:
                if np.any(np.isnan(id2pearson[l])) == False:
                    # print(fc.shape, )
                    final_fc.append(fc.T)
                    final_label.append(id2gender[l])
                    final_pearson.append(id2pearson[l])
        # exit()
        final_pearson = np.array(final_pearson)

        final_fc = np.array(final_fc)
        matrices = []

        final_label = np.array(final_label)
        print(final_pearson.shape, final_fc.shape, final_label.shape)

        final_pearson_normalized = copy.deepcopy(final_pearson) * 0.5
        final_pearson_normalized[final_pearson_normalized < 0.0] = 0

        encoder = preprocessing.LabelEncoder()

        encoder.fit(label_df["sex"])

        final_label = encoder.transform(final_label)

        for j in range(np.shape(final_label)[0]):
            data = benreader.read_data(dataset_name = i,  i = j, X = final_fc, y = final_label, W = final_pearson)
            W_true = data.W

            tic = time()
            
            ## training no-bears 
            # W_init = NOBEARS.W_reg_init(data.data).astype('float32')
            X = data.data.copy()
            # X = benchmark_data_reader.rank_transform(X)
            X = benchmark_data_reader.mean_var_normalize(X)
            # X = np.vstack([X] * 2)
            d, n = data.num_gene, data.num_sample
            # print(X.shape, W_init.shape)
            matrices.append(X)
            # exit()
        matrices = np.array(matrices)
        idxs = np.arange(final_fc.shape[0])
        
        final_fc, final_pearson, labels, final_pearson_normalized, idxs = [torch.from_numpy(
            data).float() for data in (final_fc, final_pearson, final_label, final_pearson_normalized, idxs)]

        dataset = utils.TensorDataset(
            final_fc,
            final_pearson,
            labels,
            final_pearson_normalized,
            idxs
        )
        print(final_fc.shape, final_pearson.shape, labels.shape, final_pearson_normalized.shape, idxs.shape)

        length = final_fc.shape[0]
        # if config["train"]["ratio"] < 1:    
        train_length = int(length * 0.7 * config["train"]["ratio"])
        # else:
        # train_length = int(length*0.7)
        val_length = int(length * 0.1)
        test_length = int(length * 0.2)

        # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        #     dataset, [train_length, val_length, length-train_length-val_length],generator=torch.Generator().manual_seed(0))
        
        lengths = [train_length, val_length, test_length]
        indices = torch.randperm(sum(lengths), generator=torch.Generator().manual_seed(64)).tolist() # 100
        # print(indices)
        train_dataset, val_dataset, test_dataset = \
            [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

        print(f"TRAIN: {len(train_dataset)}, DEV: {len(val_dataset)}, TEST: {len(test_dataset)}")
        # create dataloader
        # print(matrices.shape, y_full["data"].shape)
        model = GolemTorch(n = final_fc.shape[0], d = final_fc.shape[2], t = final_fc.shape[1], lambda_1 = lambda_1, lambda_2 = lambda_2, lambda_3 = lambda_3,  pos_weight = pos_weight, neg_weight = neg_weight, \
            equal_variances = equal_variances, train = train_dataset, valid = val_dataset, test = test_dataset, B_init = final_pearson_normalized,  config = config)
        
        #### pretrain #######
        model.to(device)
        model.init_pretrain_utils()
        if config["train"]["ratio"] < 1:    
            save_path = f'ckpts/pretrain_abcd_{config["train"]["ratio"]}_{seed}/'
        else:
            save_path = f'ckpts/pretrain_abcd_{seed}/'
        if not os.path.exists(save_path):
            best_model = model.gnn_model_pretrain(save_dir = save_dir, device = device)
            os.makedirs(save_path, exist_ok = True)
            # save_path = f'ckpts/pretrain_abcd_{seed}'
            torch.save(best_model.state_dict(), f"{save_path}/gnn_pretrain.pt")
        else:
            best_model = GolemTorch(n = final_fc.shape[0], d = final_fc.shape[2], t = final_fc.shape[1], lambda_1 = lambda_1, lambda_2 = lambda_2, lambda_3 = lambda_3,  pos_weight = pos_weight, neg_weight = neg_weight, \
                equal_variances = equal_variances, train = train_dataset, valid = val_dataset, test = test_dataset, B_init = final_pearson_normalized,  config = config)
            best_model.load_gnn(save_path)
            best_model = copy.deepcopy(best_model.gnn_model)
            best_model.to(device)

        train_DAG = model.causal_structure_learning_single(device = device, mode = 'train', best_model = best_model, w_contra= 200, w_target = 200)
        test_DAG  = model.causal_structure_learning_single(device = device, mode = 'test', best_model = best_model, w_contra= 200, w_target = 200) # best_model
        
        if config["train"]["load_prev_model"]:
            print("Load Prev Model")
            model.load_gnn(save_path)
            # model.train()
        else:
            pass 

        best_model = model.gnn_model_finetunes(save_dir = save_dir, device = device)
        


