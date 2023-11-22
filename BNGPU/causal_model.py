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
from torch.linalg import matrix_norm
from .fbnetgen import GruKRegion, ConvKRegion

from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support

class CausalModel(nn.Module):
    _logger = logging.getLogger(__name__)
    
    def __init__(self, n, d, t, config, equal_variances=True, seed=1, B_init = None, T = 2):
        super(CausalModel, self).__init__() 
        self.n = n # number of brains
        self.d = d # number of nodes
        self.t = t # length of time series
        self.config = config 
        self.equal_variances = equal_variances
        self.B_init = B_init
        self.T = T 
        self._build()
    
    def _build(self):
        """Build tensorflow graph."""
        # Placeholders and variables
        # self.lr = tf.compat.v1.placeholder(tf.float32)
        # self.X = tf.compat.v1.placeholder(tf.float32, shape=[self.n, self.d])
        
        if self.B_init is not None:
            self.B = torch.FloatTensor(self.B_init)
        else:
            self.B = torch.randn([self.n, self.d, self.d]) * 0.01
        self.A = torch.randn([self.n, self.T * self.d, self.d]) * 0.01
        self.positive = torch.randn([self.d, self.d]) * 0.01
        self.negative = torch.randn([self.d, self.d]) * 0.01
        self._preprocess()
        #
        # Likelihood, penalty terms and score
        # self.likelihood = self._compute_likelihood()
        # self.L1_penalty = self._compute_L1_penalty()
        # self.h = self._compute_h()
        # self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h

        # Optimizer
        # self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.score)
        self._logger.debug("Finished building graph.")

    def _preprocess(self):
        """Set the diagonals of B to zero.
        Args:
            B (tf.Tensor): [d, d] weighted matrix.
        Returns:
            tf.Tensor: [d, d] weighted matrix.
        """
        for i in range(self.n):
            self.B[i].fill_diagonal_(0)
        self.positive.fill_diagonal_(0)
        self.negative.fill_diagonal_(0)

        self.B = nn.Parameter(self.B)
        self.A = nn.Parameter(self.A, requires_grad= True)
        self.positive = nn.Parameter(self.positive)
        self.negative = nn.Parameter(self.negative)
        # return tf.linalg.set_diag(B, tf.zeros(B.shape[0], dtype=tf.float32))


    def _compute_L1_penalty(self, B):
        """Compute L1 penalty.
        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.linalg.matrix_norm(B, ord=1)

    def _compute_h(self, B):
        """Compute DAG penalty.
        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        # return torch.trace(torch.matrix_exp(B * B)) - self.d
        h = 0
        for i in range(B.shape[0]):
            h += torch.trace(torch.matrix_exp(B[i] * B[i])) - self.d
        return h/B.shape[0]
    
    def _compute_h_single(self, B):
        """Compute DAG penalty.
        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        # return torch.trace(torch.matrix_exp(B * B)) - self.d
        h = torch.trace(torch.matrix_exp(B * B)) - self.d
       
        return h
        
        # return tf.linalg.trace(tf.linalg.expm(self.B * self.B)) - self.d

    def _compute_likelihood(self, X, B, A):
        """Compute (negative log) likelihood in the linear Gaussian case.
        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        # X: (bsz, b, d), B: (bsz, d, d)

        if self.equal_variances:    # Assuming equal noise variances
            # print( 0.5 * self.d * torch.log(
            #     torch.sum(
            #             torch.square(X - X @ B), dim = (1, 2)
            #     )))
            # print(
            #     0.5 * self.d * torch.log(
            #     torch.square(
            #             matrix_norm(X - X @ B)
            #             # torch.square(X - X @ B), dim = (1, 2)
            #         )
            #     )
            # )
            # # print(torch.square(X - X @ B).shape)
            # print( 0.5 * torch.sum(
            #     torch.log(
            #         torch.sum(
            #             torch.square(X - X @ B), axis = 1
            #         )
            #     ), axis = 1)
            # )
            X_t = X[:, self.T:, :]# (bsz, b-T, d)

            X_p = torch.cat([X[:, (self.T-i):(-i), :] for i in range(1, 1+self.T)], dim = -1).to(X.device)

            return 0.5 * self.d * torch.log(
                torch.sum(
                        # matrix_norm(X - X @ B)
                        matrix_norm(X_t - X_t @ B - X_p @ A)
                        # torch.square(X - X @ B), dim = (1, 2)
                    )
                ) - torch.linalg.slogdet(torch.eye(self.d).to(B.device) - B)[1]
            # return 0.5 * self.d * tf.math.log(
            #     tf.square(
            #         tf.linalg.norm(X - X @ B)
            #     )
            # ) - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]
        else:    # Assuming non-equal noise variances
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - X @ B), axis = 1
                    )
                ), axis = 1) - torch.linalg.slogdet(torch.eye(self.d).to(B.device) - B)[1]
    
    def forward(self, X, idx, labels = None):
        B = self.B[idx]
        A = self.A[idx]
        # Time series; 
        # print(X.requires_grad, B.requires_grads)
        loss = self._compute_likelihood(X, B, A)
        penalty1 = self._compute_L1_penalty(B) + self._compute_L1_penalty(A)
        penalty2 = self._compute_h(B)
        # print(A, B, loss, penalty1, penalty2)

        return loss.mean(), penalty1.mean(), penalty2.mean()


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