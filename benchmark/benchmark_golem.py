import numpy as np
import scipy.linalg as slin
import networkx as nx
import timeit
import numpy as np
# import tensorflow as tf
import torch 
import torch.nn as nn 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.append(os.path.abspath('../benchmark_data/'))
sys.path.append(os.path.abspath('../'))
from BNGPU import NOBEARS
from time import time
import benchmark_data_reader

from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


if __name__ == '__main__':
    benreader = benchmark_data_reader.BenchmarkReader()
    res = []

    for i in benreader.get_dataset_name():
        matrices = []
        for j in range(514):
            data = benreader.read_data(i, j)
            W_true = data.W

            tic = time()
            
            ## training no-bears 
            W_init = NOBEARS.W_reg_init(data.data).astype('float32')
            X = data.data.copy()
            X = benchmark_data_reader.rank_transform(X)
            X = benchmark_data_reader.mean_var_normalize(X)
            X = np.vstack([X] * 2)
            d, n = data.num_gene, data.num_sample
            # print(X.shape, W_init.shape)
            matrices.append(X)
            # exit()
        matrices = np.array(matrices)
        print(matrices.shape)
        exit()


        # y_pred = np.abs(W_est.ravel())
        # y_true = np.abs(W_true.ravel()) > 1e-5
        
        # s0 = average_precision_score(y_true, y_pred)
        # s1 = roc_auc_score(y_true, y_pred)
        
        # y_pred = np.abs(W_est_init.ravel())        
        # s2 = average_precision_score(y_true, y_pred)
        # s3 = roc_auc_score(y_true, y_pred)
        # res.append((i, d, n, time() - tic, s0, s1, s2, s3))

        # print('Save:', W_est.shape, W_init.shape)

