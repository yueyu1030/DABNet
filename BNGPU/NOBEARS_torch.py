from pkg_resources import require
import torch
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats.stats import spearmanr
from tqdm import trange
import torch.nn.functional as F
import torch.nn as nn 
import torch.utils.data as utils
from tqdm import tqdm , trange
from torch.optim.lr_scheduler import StepLR
from .EncourageLoss import EncouragingLoss
import logging
import os
import copy
from .fbnetgen import GruKRegion, ConvKRegion
from .causal_model import CausalModel
from .causal_model_single import SingleCausalModel

from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support, f1_score
import json 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import seaborn as sns

def W_linear_reg(X, W, alpha, K, noise_level = 0.01):
    X0_tf = X + noise_level * torch.randn(X.shape)
    # alpha: torch.nn.paramerter with shape B * (K+1) * 1
    X1_tf = alpha[:, 0, :]
    for i in range(1, K+1):    
        if i == 0:
            X1_tf += (X0_tf) * alpha[:, i, :]
        else:
            X1_tf += (X0_tf ** (i)) * alpha[:, i, :]
    X2_tf = torch.einsum('ijk, ikm -> ijm', X1_tf, W) 
    return X2_tf 


def W_reg_init(X):
    n, d = X.shape
    rho, p = spearmanr(X)
    np.fill_diagonal(rho, 0)
    
    W_init = np.zeros((d, d))
    for i in range(d):
        ind = np.argsort(rho[:, i])[::-1][:min([int(n * 0.5), 30])]
        Xpar = X[:, ind]
        Xpar += 0.01 * np.random.randn(*Xpar.shape)
        yc = X[:, i]
        wpar = lsq_linear(Xpar, yc)['x']

        W_init[ind, i] = wpar
    W_init = W_init
    return W_init

def power_iteration_torch(W, power_iteration_rounds = 1 ):
    '''
    W: a batch of feature; B * M* M
    '''
    
    u_var = torch.randn(W.shape[0], W.shape[1], 1) * 0.1
    v_var = torch.randn(W.shape[0], W.shape[1], 1) * 0.1
    for _ in range(power_iteration_rounds):
        v = F.normalize(torch.bmm(W, v_var), p = 2, eps = 1e-12)
        u = F.normalize(torch.bmm(W.transpose(1, 2).contiguous(), u_var), p = 2, eps = 1e-12)
        v_var = v
        u_var = u
    
    W_med = torch.einsum("ijk, ijl -> ikl", u_var.detach(), W)
    W_final = torch.einsum("ijk, ikm -> ijm", W_med, v_var.detach())
    # W_final shape: 1*1
    W_final = W_final.flatten()
    norm_value = torch.einsum("ijk, ijp -> ikp", u_var.detach(), v_var.detach()).flatten()
    eigen_value = W_final/norm_value 
    # norm_value = tf.matmul(tf.matmul(v, W, transpose_a=True), u) / tf.reduce_sum(v * u)
    # norm_value.shape.assert_is_fully_defined()
    return u_var, v_var, eigen_value 

class GolemTorch(nn.Module):
    # def __init__(self, beta1 = 0.05, beta2 = 0.001, alpha_init = 0.01, rho_init = 1.0, poly_degree = 3, l1_ratio = .5, total_size = 10, total_regions=100,  batch_size = 16):
    _logger = logging.getLogger(__name__)
    def __init__(self, n, d, t, lambda_1, lambda_2, lambda_3, config, pos_weight=0.1, neg_weight = 0.1,  \
                equal_variances=True, seed=1, B_init = None, train = None, valid = None, test = None):
        super(GolemTorch, self).__init__() 
        self.n = n # number of brains
        self.d = d # number of nodes
        self.t = t # length of time series
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.equal_variances = equal_variances
        self.config = config
        # self.X shape: [number of brains, number of observations, number of nodes]
        # self.X = X 
        # self.y = y
        # self.B shape: [number of brains,  number of nodes, number of nodes]
        self.B_init = B_init 

        self.train = train 
        self.valid = valid 
        self.test = test 
        self.batch_size = int(config["data"]["batch_size"])
        self._gen_dataset()
        node_size = self.d 
        node_feature_size = self.d 
        timeseries_size = self.t
        ########### GNN for downstream task ##############
        self.gnn_model = FBNETGEN(self.config['model'], node_size,
                             node_feature_size, timeseries_size)
        self.loss_finetune_fn = EncouragingLoss(log_end=1) #torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_norm = "L1"
        ########### GNN for Generation task ##############
        if config["data"]["dataset"].lower() == "abcd":
            self.causal_model = SingleCausalModel(d = self.d, t = self.t, n = self.n, config = self.config, equal_variances = equal_variances, B_init = B_init)
        else:
            self.causal_model = CausalModel(d = self.d, t = self.t, n = self.n, config = self.config, equal_variances = equal_variances, B_init = B_init)

    def save_gnn(self, save_path):
        # save model parameters
        os.makedirs(f"{save_path}", exist_ok = True)
        torch.save(self.gnn_model.state_dict(), f"{save_path}/gnn_pretrain.pt")
        # torch.save(self.causal_model.state_dict(), f"{save_path}/causal_pretrain.pt")

    def load_gnn(self, load_path):
        # save model parameters
        if os.path.exists(f"{load_path}"):
            self.gnn_model.load_state_dict(torch.load(f"{load_path}/gnn_pretrain.pt"))
            # self.causal_model.load_state_dict(torch.load(f"{load_path}/causal_pretrain.pt"))
        else:
            print("No CKPT!") 

    def save_ckpt(self, save_path):
        # save model parameters
        os.makedirs(f"{save_path}", exist_ok = True)
        torch.save(self.gnn_model.state_dict(), f"{save_path}/gnn_pretrain.pt")
        torch.save(self.causal_model.state_dict(), f"{save_path}/causal_pretrain.pt")

    def load_ckpt(self, load_path):
        # save model parameters
        if os.path.exists(f"{load_path}"):
            self.gnn_model.load_state_dict(torch.load(f"{load_path}/gnn_pretrain.pt"))
            self.causal_model.load_state_dict(torch.load(f"{load_path}/causal_pretrain.pt"))
        else:
            print("No CKPT!") 
       

    def _gen_dataset(self):
        self.train_dataloader = utils.DataLoader(
            self.train, batch_size = self.batch_size, num_workers=12, shuffle=True, drop_last=True)

        self.train_gen_dataloader = utils.DataLoader(
            self.train, batch_size = 1, num_workers=12, shuffle=True, drop_last=True)

        self.val_dataloader = utils.DataLoader(
            self.valid, batch_size= self.batch_size,  num_workers=12, shuffle=True, drop_last=True)

        self.test_dataloader = utils.DataLoader(
            self.test, batch_size= self.batch_size, num_workers=12,  shuffle=True, drop_last=True)
        
        self.test_gen_dataloader = utils.DataLoader(
            self.test, batch_size = 1, num_workers=12, shuffle=True, drop_last=True)


    def init_param(self, X, alpha = None, weight_matrix = None):
        if not alpha:
            self.alpha = nn.Parameter(0.1 * torch.random.normal(self.total_size, self.poly_degree, 1))
        else:
            self.alpha = alpha 
        if weight_matrix:
            self.W = weight_matrix
        else:
            self.W = nn.Parameter(0.1 * torch.random.normal(self.total_size, self.total_regions, self.total_regions))
        self.X = X
    
    def contrast_loss1(self, B):
        B = B.reshape(B.shape[0], -1) 
        if self.loss_norm == 'L1':
            cdist = torch.cdist(B, B, p = 1)
            loss = torch.sum(cdist)
            # B =
        elif self.loss_norm == 'IP':
            pairwise_dot_product = torch.matmul(B, B.transpose(0, 1))
            loss = F.cross_entropy(pairwise_dot_product, np.arange(B.shape[0]))
        return loss 
        
        # L1 norm or dot product 
    def contrast_loss_label(self, B, labels, margin, mode = "train"):
        B = B #.reshape(B.shape[0], -1)           # (B * Length)
        neg_vec = self.causal_model.positive #.reshape(1, -1)  # (1 * Length)
        pos_vec = self.causal_model.negative #.reshape(1, -1)  # (1 * Length)
        if self.loss_norm == 'L1':
            # pass 
            self.loss = nn.MarginRankingLoss(margin = margin, reduction = 'none')
            
            neg_dist = torch.linalg.matrix_norm(B - neg_vec, ord = 1)
            pos_dist = torch.linalg.matrix_norm(B - pos_vec, ord = 1)
            # print(neg_dist, pos_dist, labels)
            if mode == 'test':
                loss = self.loss(neg_dist, pos_dist, 2*labels - 1)
                weight = F.softmax(torch.cat([neg_dist.reshape(-1,1), pos_dist.reshape(-1,1)], dim = 1))
                # print(weight)
                weight = torch.sum(-torch.log(weight+1e-6) * weight, dim = 1)
                # print(weight, weight.shape[-1], np.log(weight.shape[-1]))
                weight = (1 - weight / np.log(2)) ** 2
                # print(weight, loss)
                loss = loss * weight
                loss = loss.mean()
            else:
                loss = self.loss(neg_dist, pos_dist, 2*labels - 1).mean()

        elif self.loss_norm == 'IP':
            self.loss = nn.CrossEntropyLoss()
            pos_rel = torch.einsum("ij, kj->ik", B, pos_vec).squeeze() # (B, )
            neg_rel = torch.einsum("ij, kj->ik", B, neg_vec).squeeze() # (B, )
            pred = torch.cat([pos_rel, neg_rel], dim = 1)
            loss = self.loss(pred, labels)
        l1_penalty_pos = self.causal_model._compute_L1_penalty(self.causal_model.positive)
        h_pos = self.causal_model._compute_h_single(self.causal_model.positive)

        l1_penalty_neg = self.causal_model._compute_L1_penalty(self.causal_model.negative)
        h_neg = self.causal_model._compute_h_single(self.causal_model.negative)

        loss_pos = self.pos_weight * (self.lambda_1 * l1_penalty_pos + self.lambda_2 * h_pos) + \
             self.neg_weight * (self.lambda_1 * l1_penalty_neg + self.lambda_2 * h_neg)
        loss = loss + loss_pos

        return loss


    def init_pretrain_utils(self):
        self.pretrain_loss = TotalMeter()
        self.train_accuracy = TotalMeter()


    def causal_structure_learning_single(self, device, mode = 'train', best_model = None, w_contra= 100, w_target = 100):
        self._logger.info("--- Causal Structure Learning (single) ---")
       
        # plt.figure(figsize = [15, 12], dpi = 110)
        if mode == 'train':
            train_dataloader = self.train_gen_dataloader
        elif mode == 'test':
            train_dataloader = self.test_gen_dataloader


        casual_dags = torch.zeros([self.n, self.d, self.d])
        dags = []
        labels = []
        # print(self.n, len(train_dataloader))
        # exit()
        for i, (data_in, pearson, label, normalize_pearson, idx) in tqdm(enumerate(train_dataloader)):
            self.causal_model.train()
            
            data_in, pearson, label, normalize_pearson = data_in.to(
                device), pearson.to(device), label.to(device), normalize_pearson.to(device)
            idx = idx.long()
            # with torch.no_grad():
            self.causal_model._init_B(normalize_pearson.squeeze(0))
            # print('before', normalize_pearson.squeeze(0))
            optimizer = torch.optim.Adam(
                self.causal_model.parameters(), lr=self.config['train']['lr_causal'],
                weight_decay=self.config['train']['weight_decay'] * 0.1)
            
            for _ in trange(self.config['train']['gen_epoch']):
                loss, penalty1, penalty2 = self.causal_model(data_in, idx, label)
                final_loss = self.lambda_3 * loss + penalty1 * self.lambda_1 + self.lambda_2 * penalty2
                gen_dag = self.causal_model.B_tmp.unsqueeze(0) #[idx]
                # print(gen_dag, self.causal_model.B.requires_grad )
                # print(pearson.shape, gen_dag.shape)
                
                if best_model is not None:
                    best_model.eval()
                    ce_loss = nn.KLDivLoss(reduction='none', log_target= False)
                    
                    # mixed_data, mixed_pearson, mixed_dag, targets_a, targets_b, lam = mixup_data(
                    #     data_in, pearson, gen_dag, label, 1, device)
                    output_logit = best_model(t = data_in, label=label, \
                                                nodes_adj = normalize_pearson, pearson = pearson, \
                                                dag_adj = gen_dag, adj_matrix = True)
                    output = F.softmax(output_logit, dim = -1)
                    f = torch.sum(output, dim=0)
                    t = output**2 / f
                    t = t + 1e-10
                    p = t/torch.sum(t, dim=-1, keepdim=True)
                    target_loss = ce_loss(output.log(), p).mean()
                else:
                    target_loss = 0
                    # print(output, label)
                    # label = label.long()
                    # target_loss = ce_loss(output, label)
                    # print(output, p, target_loss)                                    
                loss = final_loss +  w_contra * self.contrast_loss_label(B = gen_dag, labels = label, margin = 1.0, mode = mode) + w_target * target_loss
                # print(loss, final_loss)
                optimizer.zero_grad()
                loss.backward()
                # grad_norm = {n: p.grad.detach() for (n, p) in self.causal_model.named_parameters()}
                nn.utils.clip_grad_norm_(self.causal_model.parameters(), 5.0)
                optimizer.step()
            # print(self.causal_model.B.data.detach().shape, self.causal_model.B.data.detach())
                # print('after',_, self.causal_model.B_tmp.data.detach().cpu())
            # print(self.causal_model.B.data.detach().cpu()-normalize_pearson.unsqueeze(0).cpu())
            casual_dags[idx.detach().cpu()] = self.causal_model.B_tmp.data.detach().cpu()
            self.causal_model.B[idx.detach().cpu()] = self.causal_model.B_tmp.data.detach()
            # print(loss, penalty1, penalty2, self.causal_model.B[1], self.causal_model.B[0])
            # plt.figure(figsize = [13, 7], dpi = 80)
            # print(self.causal_model.B_tmp.data.detach(), normalize_pearson.detach().cpu().numpy()[0])
            # plt.subplot(1, 2, 1)
            # ax = sns.heatmap(normalize_pearson.detach().cpu().numpy()[0], linewidth=0.02)
            # plt.subplot(1, 2, 2)
            # ax = sns.heatmap(self.causal_model.B_tmp.data.detach().cpu().numpy(), linewidth=0.02)
            # plt.savefig(f"results/abcd_{i}_causal_{self.config['train']['lambda_init']}_{self.lambda_1}_{self.lambda_2}_{self.lambda_3}.pdf")
            labels.append(label.detach().cpu())
            dags.append(self.causal_model.B_tmp.data.detach().cpu().numpy())
        labels = np.array(labels)
        dags = np.array(dags)
        print(labels.shape)
        print(dags.shape)
        np.save("ABCD_labels.npy", labels)
        np.save("ABCD_dags.npy", dags)
        return casual_dags.detach().cpu().numpy()

    def causal_structure_learning(self, device, mode = 'train', best_model = None, w_contra= 100, w_target = 100):
        self._logger.info("--- Causal Structure Learning ---")
        optimizer = torch.optim.Adam(
                    self.causal_model.parameters(), lr=self.config['train']['lr_causal'],
                    weight_decay=self.config['train']['weight_decay'] * 0.1)
        plt.figure(figsize = [15, 10], dpi = 90)
        if mode == 'train':
            train_dataloader = self.train_dataloader
        elif mode == 'test':
            train_dataloader = self.test_dataloader
        # results = torch.zeros()
        # train_id = []
        train_mat1 = []
        label_mat1 = []

        train_mat2 = []
        label_mat2 = []
        for _ in trange(self.config['train']['gen_epoch']):
            train_labels = {}
            test_label_id = []
            for i, (data_in, pearson, label, normalize_pearson, idx) in tqdm(enumerate(train_dataloader)):
                self.causal_model.train()
                
                data_in, pearson, label, normalize_pearson = data_in.to(
                    device), pearson.to(device), label.to(device), normalize_pearson.to(device)
                idx = idx.long()
                # train_id += list(idx.cpu().numpy())
                # print(idx)
                train_idx = idx.cpu().numpy()
                train_label = label.cpu().numpy()
                for (x, y) in zip(train_idx, train_label):
                    test_label_id.append(int(x))
                    train_labels[int(x)] = int(y)


                loss, penalty1, penalty2 = self.causal_model(data_in, idx, label)
                
                final_loss = self.lambda_3 * loss + penalty1 * self.lambda_1 + self.lambda_2 * penalty2
                gen_dag = self.causal_model.B[idx].to(device)
                # print(gen_dag, normalize_pearson, pearson )
                # print(pearson.shape, gen_dag.shape)
                
                if best_model is not None:
                    best_model.eval()
                    ce_loss = nn.KLDivLoss(reduction='none', log_target= False)
                    
                    # mixed_data, mixed_pearson, mixed_dag, targets_a, targets_b, lam = mixup_data(
                    #     data_in, pearson, gen_dag, label, 1, device)
                    output_logit = best_model(t = data_in, label=label, \
                                                nodes_adj = normalize_pearson, pearson = pearson, \
                                                dag_adj = gen_dag, adj_matrix = True)
                    
                    output = F.softmax(output_logit, dim = -1)
                    f = torch.sum(output, dim=0)
                    t = output**2 / f
                    t = t + 1e-10
                    p = t / torch.sum(t, dim=-1, keepdim=True)
                    target_loss = ce_loss(output.log(), p).mean()
                else:
                    target_loss = 0
                    # print(output, label)
                    # label = label.long()
                    # target_loss = ce_loss(output, label)
                    # print(output, p, target_loss)
                                    
                final_loss = final_loss +  w_contra * self.contrast_loss_label(B = gen_dag, labels = label, margin = 1.0, mode = mode) + w_target * target_loss
                optimizer.zero_grad()
                final_loss.backward()
                # grad_norm = {n: p.grad.detach() for (n, p) in self.causal_model.named_parameters()}
                # print("-------")
                # print("")
                # print(grad_norm["A"][idx], grad_norm["B"][idx])
                # print("")
                # print("-------")
                gnorm = nn.utils.clip_grad_norm_(self.causal_model.parameters(), 2.0)
                
                optimizer.step()
            # print(loss, penalty1, penalty2, self.causal_model.B[1], self.causal_model.B[0])
            # print(len(test_label_id), len(train_labels))
            if _ == 20:
                for x in test_label_id:
                    train_mat1.append(self.causal_model.B[int(x)].detach().cpu().numpy())
                    label_mat1.append(train_labels[int(x)])
                train_mat1 = np.array(train_mat1)
                label_mat1 = np.array(label_mat1)
                print(train_mat1.shape, label_mat1.shape)
                np.save("train_mat_20.npy", train_mat1)
                np.save("train_label_20.npy", label_mat1)
            elif _ == 40:
                for x in test_label_id:
                    train_mat2.append(self.causal_model.B[int(x)].detach().cpu().numpy())
                    label_mat2.append(train_labels[int(x)])
                train_mat2 = np.array(train_mat2)
                label_mat2 = np.array(label_mat2)
                print(train_mat2.shape, label_mat2.shape)
                np.save("train_mat_40.npy", train_mat2)
                np.save("train_label_40.npy", label_mat2)            
            # exit()
            if _ % 20 == 0 :
                plt.subplot(2, 3, 1 + _//20)
                ax = sns.heatmap(self.causal_model.B[296].detach().cpu().numpy(), linewidth=0.04)
                plt.subplot(2, 3, 4 + _//20)
                ax = sns.heatmap(self.causal_model.B[122].detach().cpu().numpy(), linewidth=0.04)
        plt.savefig(f"results/causal_{self.config['train']['lambda_init']}_{self.lambda_1}_{self.lambda_2}_{self.lambda_3}.pdf")
        
        return self.causal_model.B.detach().cpu().numpy()


    def gnn_model_finetunes(self, device, save_dir= '',  dag = None):
        optimizer = torch.optim.Adam(
            self.gnn_model.parameters(), lr=self.config['train']['lr'] * 2,
            weight_decay=self.config['train']['weight_decay'])
        # scheduler = StepLR(optimizer, step_size = 50, gamma=0.5)
        best_result = -1
        self.gnn_model.train()
        self._logger.info("--- GNN Pre-training ---")
        # gen_epoch
        use_best_gnn = int(self.config['train']['load_prev_model'])
        f = open(save_dir + f"finetune_contra_result_{self.config['data']['dataset']}_{self.config['train']['seed']}_{self.config['train']['gen_epoch']}_{self.config['train']['ratio']}_{use_best_gnn}.txt", 'w')
        results =  []
        aucs = []
        for rounds in trange(300):
            for i, (data_in, pearson, label, normalize_pearson, idx) in tqdm(enumerate(self.train_dataloader)):
                label = label.long()
                idx = idx.long()
                gen_dag = self.causal_model.B[idx].detach()
                gen_dag_normalized = copy.deepcopy(gen_dag)
                # gen_dag_normalized[gen_dag_normalized < 0.0] = 0
                
                data_in, pearson, label, normalize_pearson, gen_dag_normalized = data_in.to(
                    device), pearson.to(device), label.to(device), normalize_pearson.to(device), gen_dag_normalized.to(device)

                mixed_data, mixed_pearson, mixed_dag, targets_a, targets_b, lam = mixup_data(
                    data_in, pearson, gen_dag_normalized, label, 1, device)
 
                output = self.gnn_model(t = data_in, label=label, \
                                                nodes_adj = normalize_pearson, pearson = pearson, \
                                                dag_adj = gen_dag_normalized, adj_matrix = True)

                loss = 2 * mixup_criterion(
                    self.loss_finetune_fn, output, targets_a, targets_b, lam)

                self.pretrain_loss.update_with_weight(loss.item(), label.shape[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                top1 = accuracy(output, label)[0]
                self.train_accuracy.update_with_weight(top1, label.shape[0])
                if i  in [len(self.train_dataloader)//3, 2*len(self.train_dataloader)//3]:
                    auc = self.test_per_epoch(mode = 'dev', device = device, use_dag=True)
                    if auc[0] > best_result:
                        best_result = auc[0]
                        best_model = copy.deepcopy(self.gnn_model)
                    aucs.append(float(auc[1]))
                    results.append(float(auc[0]))
                    f.write(f"Round: {rounds}, AUC: {auc[0]}, Acc: {auc[1]}\n")

            auc = self.test_per_epoch(mode = 'dev', device = device, use_dag=True)
            aucs.append(float(auc[1]))
            results.append(float(auc[0]))
            f.write(f"Round: {rounds}, AUC: {auc[0]}, Acc: {auc[1]}\n")
            
            if auc[0] > best_result:
                best_result = auc[0]
        f.close()
        print(f" auc: {best_result}")
        json.dump(results, open(save_dir + f"acc_{self.config['data']['dataset']}_{self.config['train']['seed']}_{self.config['train']['gen_epoch']}_{use_best_gnn}.json", 'w'), indent = 2)
        json.dump(aucs, open(save_dir + f"auc_{self.config['data']['dataset']}_{self.config['train']['seed']}_{self.config['train']['gen_epoch']}_{use_best_gnn}.json", 'w'), indent = 2)
            
        return best_model

    def gnn_model_pretrain(self, save_dir, device):
        optimizer = torch.optim.Adam(
            self.gnn_model.parameters(), lr=self.config['train']['lr'] * 2,
            weight_decay=self.config['train']['weight_decay'])
        # scheduler = StepLR(optimizer, step_size = 50, gamma=0.5)
        
        self.gnn_model.train()
        self._logger.info("--- GNN Pre-training ---")
        f = open(save_dir + f"pretrain_contra_result_{self.config['data']['dataset']}_{self.config['train']['seed']}.txt", 'w')
        results = []
        auc = []
        best_result = -1
        best_acc = 0
        best_auc = 0
        for rounds in trange(80):
            for i, (data_in, pearson, label, normalize_pearson, _) in tqdm(enumerate(self.train_dataloader)):
                label = label.long()
                # idx = idx.long()

                data_in, pearson, label, normalize_pearson = data_in.to(
                    device), pearson.to(device), label.to(device), normalize_pearson.to(device)

                # print(pearson, normalize_pearson)
                # pearson: with < 0 entities
                # Normalized Pearson: without < 0 entities.
                mixed_data, mixed_pearson, mixed_dag, targets_a, targets_b, lam = mixup_data(
                    data_in, pearson, normalize_pearson, label, 1, device)
                
                # inputs: Time Series
                # nodes: Pearson
                output = self.gnn_model(t = data_in, label=label, \
                                                nodes_adj = normalize_pearson, pearson = pearson, \
                                                dag_adj = None, adj_matrix = True)
                # print(nodes.shape, learnable_matrix.shape)

                loss = 2 * mixup_criterion(
                    self.loss_finetune_fn, output, targets_a, targets_b, lam)

                self.pretrain_loss.update_with_weight(loss.item(), label.shape[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                top1 = accuracy(output, label)[0]
                self.train_accuracy.update_with_weight(top1, label.shape[0])
                if i  in [len(self.train_dataloader)//2, len(self.train_dataloader)//4]:
                    result = self.test_per_epoch(mode = 'test', device = device)
                    f.write(f"Round: {rounds}, AUC: {result[0]}, Acc: {result[1]}\n")
                    results.append(float(result[0]))
                    auc.append(float(result[1]))
                    if float(result[1]) + float(result[0]) * 0.5 > best_result:
                        best_result = float(result[1]) + float(result[0]) * 0.5 
                        best_model = copy.deepcopy(self.gnn_model)
                        best_acc = float(result[1])
                        best_auc = float(result[0])
            result= self.test_per_epoch(mode = 'test', device = device)
            results.append(float(result[0]))
            auc.append(float(result[1]))
            f.write(f"Round: {rounds}, AUC: {result[0]}, Acc: {result[1]}\n")
        f.close()
        print(f" auc: {best_auc}, acc: {best_acc}")
        json.dump(results, open(save_dir + f"acc_{self.config['data']['dataset']}_{self.config['train']['seed']}.json", 'w'), indent = 2)
        json.dump(auc, open(save_dir + f"auc_{self.config['data']['dataset']}_{self.config['train']['seed']}.json", 'w'), indent = 2)
        return best_model


    def test_per_epoch(self, device, mode = 'valid', use_dag = False): # loss_meter, acc_meter, 
        labels = []
        result = []
        matrix = []
        self.gnn_model.eval()
        if mode == 'valid':
            dataloader = self.val_dataloader
        else:
            dataloader = self.test_dataloader
        acc_cnt = 0
        acc_num = 0
        for data_in, pearson, label, normalize_pearson, idx in dataloader:
            label = label.long()
            idx = idx.long()

            data_in, pearson, label, normalize_pearson, idx = data_in.to(
                device), pearson.to(device), label.to(device), normalize_pearson.to(device), idx.to(device) #, dag.to(device)
            if use_dag:
                gen_dag = self.causal_model.B[idx].detach()
                gen_dag_normalized = copy.deepcopy(gen_dag)
                # gen_dag_normalized[gen_dag_normalized < 0.0] = 0
                output = self.gnn_model(data_in, label, normalize_pearson, pearson, dag_adj = gen_dag_normalized, adj_matrix = True, mix = False)
                # dag = self.[idx]
            else:
                output = self.gnn_model(data_in, label, normalize_pearson, pearson, dag_adj = None, adj_matrix = True, mix = False)

            loss = self.loss_finetune_fn(output, label)
            # loss_meter.update_with_weight(
                # loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_num += top1 * label.shape[0]
            acc_cnt += label.shape[0]
            # acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()
            # print(top1)
            # print(m.detach().cpu().reshape(-1))
            # print(m.shape)
            # matrix = list(m.detach().cpu().reshape(-1)[::10])
            # import matplotlib
            # matplotlib.use("Agg")
            # import matplotlib.pyplot as plt 
            # plt.figure(figsize = [12, 8])
            # plt.hist(matrix, 200)
            # plt.savefig("test_dist.pdf")
        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        # print(labels, result)
        metric = precision_recall_fscore_support(
            labels, result, average='macro')
        f1 = f1_score(labels, result, average='macro')
        print("-----------------------")
        print("AUC:", auc, "ACC:", acc_num/acc_cnt, "F1:", f1)
        # print("-----------------------")
        return [auc, acc_num/acc_cnt] + list(metric)

    def forward(self, idx, labels = None):
        X = self.X[idx] # (bsz, b, d)
        B = self.B[idx] # (bsz, d, d)
        likelihood = self._compute_likelihood(X, B)
        l1_penalty = self._compute_L1_penalty(B)
        h = self._compute_h(B)
        score = likelihood + self.lambda_1 * l1_penalty + self.lambda_2 * h
        if labels is not None:
            self.contrast_loss_label(B, labels, margin = 1)
        else:
            pass 
        return score


class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.inner_dim = inner_dim
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Dropout(0.1),
            nn.Linear(inner_dim, inner_dim),
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Dropout(0.1),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear( 8 * roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2),
        )


    def forward(self, m, node_feature, mix_layer= 0):
        # m : adj matrix
        # node_feature: node attribute
        bz = m.shape[0]
        
        mix_layer = min(0, mix_layer)
        
        # if mix_layer == 0:
        #     mixed_m = mix_lam * m + (1 - mix_lam) * m[mix_index, :]
        #     mixed_node_feature =  mix_lam * node_feature + (1 - mix_lam) * node_feature[mix_index, :]
        #     m = mixed_m
        #     node_feature = mixed_node_feature
        # print('m, feat', m.shape, node_feature.shape)
        x = torch.einsum('ijk,ijp->ijp', m, node_feature)
        # print(m.shape, x.shape, node_feature.shape)
        # print(x,m * node_feature, torch.bmm(m, node_feature), torch.einsum('ijk,ikp->ijp', m, node_feature)  )
        # print(x.shape, m.shape, node_feature.shape, )
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


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)

        m = torch.unsqueeze(m, -1)

        return m


class FBNETGEN(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        print(f"Feature dim {node_feature_dim}, node number {roi_num}, time series dim {time_series}")
        self.graph_generation = model_config['graph_generation']
        # if model_config['extractor_type'] == 'cnn':
        #     self.extract = ConvKRegion(
        #         out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
        #         time_series=time_series)
        # elif model_config['extractor_type'] == 'gru':
        #     self.extract = GruKRegion(
        #         out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
        #         layers=model_config['num_gru_layers'])
        # if self.graph_generation == "linear":
        #     self.emb2graph = Embed2GraphByLinear(
        #         model_config['embedding_size'], roi_num=roi_num)
        # elif self.graph_generation == "product":
        # self.emb2graph = Embed2GraphByProduct(
        #     model_config['embedding_size'], roi_num=roi_num)

        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)

    def save_gnn(self, save_path):
        # save model parameters
        os.makedirs(f"{save_path}", exist_ok = True)
        torch.save(self.gnn_model.state_dict(), f"{save_path}/gnn_pretrain.pt")
        # torch.save(self.causal_model.state_dict(), f"{save_path}/causal_pretrain.pt")

    def forward(self, t, label, nodes_adj, pearson = None, dag_adj = None, adj_matrix = True, mix = False):
        # inputs: Time Series
        # nodes: Pearson
        # label: classification Label
        # nodes: Node 
        # normalize_pearson: PEARSON 
        if mix == False:
            mix_layer = -1
        # x = self.extract(t)
        # feat = x
        # x = F.softmax(x, dim=-1)
        # m = self.emb2graph(x)
        # if self.graph_generation == "linear":
        #     m = gumbel_softmax(m, temperature=0.1, hard=True)

        # m = m[:, :, :, 0]

        # bz, _, _ = m.shape
        # print(torch.max(m), torch.min(m), torch.max(nodes), torch.min(nodes))
        # nodes = 0.04 * (nodes - torch.min(nodes))/(torch.max(nodes)-torch.min(nodes)) + 0.1
        # print(torch.max(m), torch.min(m), torch.max(nodes), torch.min(nodes))
        # edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))
        if pearson is not None:
            if dag_adj is None:
                return self.predictor(nodes_adj, pearson,  mix_layer= mix_layer) #, m, edge_variance
            else:
                return self.predictor(dag_adj, pearson,  mix_layer = mix_layer) #, m, edge_variance
        # elif adj_matrix:
        #     return self.predictor(m, nodes_adj, mix_layer= mix_layer) # , m, edge_variance
        else:
            if dag_adj is None:
                return self.predictor(nodes_adj, nodes_adj, mix_layer= mix_layer) # , m, edge_variance
            else:
                return self.predictor(dag_adj, nodes_adj, mix_layer= mix_layer) # , m, edge_variance

 

class NoBearsTF():
    def __init__(self, beta1 = 0.05, beta2 = 0.001, alpha_init = 0.01, rho_init = 1.0, poly_degree = 3, l1_ratio = .5):
        self.alpha = alpha_init
        self.rho = rho_init
        self.beta1 = beta1
        self.beta2 = beta2
        self.poly_degree = poly_degree 
        self.l1_ratio = l1_ratio
    
    def roar(self):
        print('Our time is short!')
        
    def model_init_train(self, sess, init_iter = 200, init_global_step = 1e-2, noise_level = 0.1):
        for t1 in range(init_iter):
            feed_dict = {self.graph_nodes['alpha']: self.alpha, 
                         self.graph_nodes['rho']: self.rho, 
                         self.graph_nodes['opt_step'] : init_global_step / np.sqrt(1.+ 0.1 * t1),
                         self.graph_nodes['beta1']: self.beta1,
                         self.graph_nodes['beta2']: self.beta2,
                         self.graph_nodes['noise_level']: noise_level}
            sess.run(self.graph_nodes['init_train_op'], feed_dict = feed_dict)
        return
        
    def model_train(self, sess, outer_iter = 200, inner_iter = 100, init_global_step = 1e-2, noise_level = 0.1, eval_fun = None):
        ave_loss_reg = -1
        ave_loss_min = np.inf
        
        sess.run(self.graph_nodes['data_iterator_init'])
        for t in trange(outer_iter):
            sess.run(self.graph_nodes['reset_opt'])
            for t1 in range(inner_iter):
                feed_dict = {self.graph_nodes['alpha']: self.alpha, 
                             self.graph_nodes['rho']: self.rho, 
                             self.graph_nodes['opt_step'] : init_global_step / np.sqrt(1.+ t1),
                             self.graph_nodes['beta1']: self.beta1,
                             self.graph_nodes['beta2']: self.beta2,
                            self.graph_nodes['noise_level']: noise_level}
                
                
                _, v0, h_val = sess.run([self.graph_nodes['train_op'], 
                                         self.graph_nodes['loss_regress'], 
                                         self.graph_nodes['loss_penalty']], 
                                        feed_dict = feed_dict)
            
            sess.run(self.graph_nodes['moving_average_op'])
            
            if ave_loss_reg < 0:
                ave_loss_reg = v0
            else:
                ave_loss_reg = 0.1 * ave_loss_reg + 0.9 * v0
            if ave_loss_min > ave_loss_reg:
                ave_loss_min = ave_loss_reg
            elif ave_loss_reg > (1.5 * ave_loss_min) and h_val < 0.005:
                return
            
            self.alpha += self.rho * h_val
            self.rho = min(1e8, 1.1 * self.rho)
        
    def construct_graph(self, X, W_init = None, buffer_size = 100, batch_size = None):
        n, d = X.shape
        if batch_size is None:
            batch_size = n // 2
            
        tf.reset_default_graph()
        
        X_datatf = tf.constant(X.astype('float32'), name='sem_data') ## n x d    
        X_dataset = tf.data.Dataset.from_tensor_slices(X_datatf)
        X_iterator = X_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).repeat().make_initializable_iterator()
        X_tf = X_iterator.get_next()
        X_tf = tf.constant(X.astype('float32'), name='sem_data') ## n x d
        
        batch_size_tf = tf.cast(tf.shape(X_tf)[0], tf.float32)
    
        if W_init is None:
            W_tf = tf.get_variable("W", [d, d], initializer=tf.random_uniform_initializer())
        else:
            W_tf = tf.get_variable("W", initializer=W_init)
            
        W_tf = tf.linalg.set_diag(W_tf, tf.zeros(d))
        
        ## ema
        ema = tf.train.ExponentialMovingAverage(decay=0.975)
        ema_op = ema.apply([W_tf])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)
        W_ema = ema.average(W_tf)
    
        noise_level_tf = tf.placeholder(tf.float32, name = 'noise_level')

        X0_tf = X_tf + noise_level_tf * tf.random.normal(tf.shape(X_tf), stddev=1.0)
        
        ## setup objective function
        
        bk0_tf = tf.get_variable("bk_poly0", shape = (1, d), initializer=tf.random_uniform_initializer())
        X1_tf = bk0_tf
        
        regression_vars = [bk0_tf]
        for k in range(1, self.poly_degree + 1):
            sig_tf = tf.get_variable("sig_poly%d" % k, shape = (1, d), initializer=tf.random_uniform_initializer())
            regression_vars.append(sig_tf)
            
            if k == 1:
                X1_tf = X1_tf + (X0_tf) * sig_tf
            else:
                X1_tf = X1_tf + (X0_tf ** k) * sig_tf
            
            
        X2_tf = tf.matmul(X1_tf, W_tf)
        
        loss_xw = tf.reduce_sum((X2_tf - X_tf) ** 2 ) / (2. * batch_size_tf)  
        
        W2 = W_tf * W_tf
        W2p = W2 + 1e-6

        vr_var, vl_var, vr, vl, h_tf = power_iteration_tf(W2p, power_iteration_rounds = 5)

        loss_penalty =  tf.reduce_sum(h_tf) 
        
        rho_tf = tf.placeholder(tf.float32, name = 'rho')
        beta1_tf = tf.placeholder(tf.float32, name = 'beta1')
        beta2_tf = tf.placeholder(tf.float32, name = 'beta2')
        d_lrd = tf.placeholder(tf.float32, name = 'opt_step')
        alpha_tf = tf.placeholder(tf.float32, name = 'alpha')        
        
        l1_ratio = self.l1_ratio
        W_abs = tf.abs(W_tf)
        reg0 = tf.reduce_sum(W_abs)
        reg1 = tf.reduce_sum(W2) * 0.5        
        
        if l1_ratio < 1e-5:
            loss_reg = beta1_tf *reg0
        elif l1_ratio > 0.99999:
            loss_reg = beta1_tf *reg2
        else:
            loss_reg = beta1_tf * ( (1. - l1_ratio) * reg1 + l1_ratio * reg0)
        
        for w in regression_vars:
            loss_reg = loss_reg + beta2_tf * tf.reduce_sum(w ** 2)
        
        loss_init = loss_xw  + loss_reg
        loss_obj = loss_init + (rho_tf / 2.0) * (loss_penalty ** 2) + alpha_tf * loss_penalty
        
        # optimizers
        optim = tf.train.AdamOptimizer(d_lrd, beta1=0.9)  
        sn_ops = [op for op in tf.get_default_graph().get_operations() if 'update_u' in op.name or 'update_v' in op.name]

        with tf.control_dependencies(sn_ops):
            train_op = optim.minimize(loss_obj)

        optim_init = tf.train.AdamOptimizer(d_lrd, beta1=0.9)  
        with tf.control_dependencies(sn_ops):
            init_train_op = optim_init.minimize(loss_init)
            
        reset_optimizer_op = tf.variables_initializer(optim.variables())
        
        model_init = tf.global_variables_initializer()

        self.graph_nodes = {'sem_data': X_tf,
                 'weight': W_tf,
                 'weight_ema': W_ema,
                 'rho': rho_tf,
                 'loss_penalty': loss_penalty,
                 'beta1': beta1_tf,
                 'beta2': beta2_tf,
                 'opt_step': d_lrd,
                 'alpha': alpha_tf,
                 'noise_level': noise_level_tf,
                 'loss_regress': loss_xw,
                 'loss_obj': loss_obj,
                 'train_op': train_op,
                 'reset_opt': reset_optimizer_op,
                  'update_uv': sn_ops,
                 'init_vars': model_init,
                 'data_iterator_init': X_iterator.initializer, 
                 'moving_average_op': ema_op,
                 'init_train_op': init_train_op,
                }

        return self.graph_nodes
    



def mixup_data(x, nodes, dag, y, alpha=1, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_dag = lam * dag + (1-lam) * dag[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, mixed_nodes, mixed_dag, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class TotalMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1

    def update_with_weight(self, val: float, count: int):
        self.sum += val*count
        self.count += count

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count
