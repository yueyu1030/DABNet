import pandas as pd
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(os.listdir(SCRIPT_DIR))
# from constants import dataset_list
# 
from scipy.io import loadmat
import numpy as np
from scipy.stats import rankdata

_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))         

def mean_var_normalize(X):
    ## X: [n_sample, n_feature] array
    ## normalize each feature to zero-mean, unit std    
    return (X - np.mean(X, axis = 0, keepdims=True)) / np.std(X, axis = 0, keepdims=True)
        
def rank_transform(X):
    ## X: [n_sample, n_feature] array
    ## apply rank transform to each feature independently
    n, d = X.shape
    for i in range(d):
        X[:, i] = rankdata(X[:, i]) / float(n)
    return X

def transform_data(X, transform):
    if transform == 'rank':
        return rank_transform(X)
    if transform == 'log2':
        return np.log2(X)
    return

class BenchmarkData():
    def __init__(self, X, W_true, y_true, id2gene = None):
        ## X: sample x gene numpy array
        ## W_true: gene x gene binary adjacent matrix, numpy array
        ## id2gene: list of gene names
        
        self.data = X.copy()
        self.W = W_true.copy()
        self.num_sample = X.shape[0]
        self.num_gene = X.shape[1]
        self.y_true = y_true
        
        if id2gene is None:
            self.id2gene = [i for i in range(self.num_gene)]
        else:
            self.id2gene = id2gene
            
class BenchmarkReader():
    def __init__(self):
        dataset_list = [ ('brain', ''),]
        self._dataset_list = dataset_list
        self.dataset_name = ['/'.join(s) for s in dataset_list]
        self._dataset_name2list = {'/'.join(s):s for s in dataset_list}
        
        return
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def read_data(self, dataset_name, i=0, X= None, y= None, W = None):
        try:
            id0, id1 = self._dataset_name2list[dataset_name]
        except:
            print('find no dataset: %s' % dataset_name)
            print('use get_dataset_name to see all available datasets')
            return
        
        if id0 == 'brain':
            return self._read_pnc_full(X, W, y, i) #self._read_pnc(i) 
        
    def _read_pnc(self, i):
        path = '/localscratch/Chao_lab/yyu414/brain_network_data/PNC_data/514_timeseries.npy'
        X = np.load(path,  allow_pickle=True).item()
        # X = np.load("../../../brain_network_data/PNC_data/514_timeseries.npy",  allow_pickle=True).item()
        X = X['data'][i]
        path = '/localscratch/Chao_lab/yyu414/brain_network_data/PNC_data/514_pearson.npy'
        y = np.load(path,  allow_pickle=True).item()

        y_true = y['data'][i]
        print(X.shape, y_true.shape)
        return BenchmarkData(X, y_true)
    
    def _read_pnc_full(self, X, W, y, i):
        # path = '/localscratch/Chao_lab/yyu414/brain_network_data/PNC_data/514_timeseries.npy'
        # X = np.load(path,  allow_pickle=True).item()
        # X = np.load("../../../brain_network_data/PNC_data/514_timeseries.npy",  allow_pickle=True).item()
        X = X[i]
        # path = '/localscratch/Chao_lab/yyu414/brain_network_data/PNC_data/514_pearson.npy'
        # y = np.load(path,  allow_pickle=True).item()
        y_true = y[i]
        W = W[i]
        # print(X.shape, y_true.shape)
        return BenchmarkData(X, W, y_true)
  