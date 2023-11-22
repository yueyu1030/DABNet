import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
import math


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x

class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        b, k, d = raw.shape
        x = raw.view((b*k, -1, self.kernel_size))
        x, h = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x

class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.embedding = nn.Parameter(torch.rand(node_input_dim, node_input_dim), requires_grad = True)
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim),
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear( 8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2),
        )


    def forward(self, m, node_feature, mix_index, mix_lam =0,mix_layer= 0):
        # m : adj matrix
        # node_feature: node attribute
        bz = m.shape[0]
        
        mix_layer = min(0, mix_layer)
        
        if mix_layer == 0:
            mixed_m = mix_lam * m + (1 - mix_lam) * m[mix_index, :]
            mixed_node_feature =  mix_lam * node_feature + (1 - mix_lam) * node_feature[mix_index, :]
            m = mixed_m
            node_feature = mixed_node_feature
        # print('m, feat', m.shape, node_feature.shape)
        x = torch.einsum('ijk,ijp->ijp', m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz * self.roi_num, -1))
        x = self.bn1(x)
       
        x = x.reshape((bz, self.roi_num, -1))
        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))
        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x)
        x = x.view(bz,-1)
        return self.fcn(x)

class FBNETGEN(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        self.graph_generation = model_config['graph_generation']
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'])
        # if self.graph_generation == "linear":
        #     self.emb2graph = Embed2GraphByLinear(
        #         model_config['embedding_size'], roi_num=roi_num)
        # elif self.graph_generation == "product":
        #     self.emb2graph = Embed2GraphByProduct(
        #         model_config['embedding_size'], roi_num=roi_num)

        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)

    def forward(self, t, label, nodes_adj, pearson = None, dag_adj = None, adj_matrix = True, mix = True, mix_index = [], mix_lam = 0, mix_layer= 0):
        # inputs: Time Series
        # nodes: Pearson
        # label: classification Label
        # nodes: Node 
        # normalize_pearson: PEARSON 
        if mix == False:
            mix_layer = -1
        x = self.extract(t)
        feat = x
        x = F.softmax(x, dim=-1)
        # m = self.emb2graph(x)
        # if self.graph_generation == "linear":
        #     m = gumbel_softmax(m, temperature=0.1, hard=True)

        m = m[:, :, :, 0]

        bz, _, _ = m.shape
        # print(torch.max(m), torch.min(m), torch.max(nodes), torch.min(nodes))
        # nodes = 0.04 * (nodes - torch.min(nodes))/(torch.max(nodes)-torch.min(nodes)) + 0.1
        # print(torch.max(m), torch.min(m), torch.max(nodes), torch.min(nodes))
        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))
        if pearson is not None:
            if dag_adj is None:
                return self.predictor(nodes_adj, pearson, mix_index = mix_index, mix_lam = mix_lam, mix_layer= mix_layer), m, edge_variance
            else:
                return self.predictor(dag_adj, pearson, mix_index = mix_index, mix_lam = mix_lam, mix_layer = mix_layer), m, edge_variance
        elif adj_matrix:
            return self.predictor(m, nodes_adj, mix_index = mix_index, mix_lam = mix_lam, mix_layer= mix_layer), m, edge_variance
        else:
            if dag_adj is None:
                return self.predictor(nodes_adj, nodes_adj, mix_index = mix_index, mix_lam = mix_lam, mix_layer= mix_layer), m, edge_variance
            else:
                return self.predictor(dag_adj, nodes_adj, mix_index = mix_index, mix_lam = mix_lam, mix_layer= mix_layer), m, edge_variance

    def mixup_data(self, x, alpha=1.0, device='cuda'):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        layer = np.random.randint(low = 0, high = 3)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        # mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
        # mixed_x = lam * x + (1 - lam) * x[index, :]
        # y_a, y_b = y, y[index]
        # mixed_dag = lam * dag + (1 - lam) * dag[index, :]
        # mixed_node_feat = lam * node_feat + (1 - lam) * node_feat[index, :]

        return index, lam, layer