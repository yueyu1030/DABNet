import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from tqdm import trange
# a = np.random.random((16, 16))
# plt.imshow(a, cmap='hot', interpolation='nearest')
# plt.show()
X = np.load("../../../brain_network_data/PNC_data/514_pearson.npy",  allow_pickle=True).item()
X = X['data']


# print(X.shape)
# exit()
for j in trange(0, 514, 10):
	w = np.load('../../DAG-GNN/src/mat/MSE_graph_DAGGNN_%d.npy' %(j))
	plt.figure(figsize= [15, 14])
	ax = sns.heatmap(w, linewidth=0.2, xticklabels=False, yticklabels=False)
	# plt.colormap()
	plt.tight_layout()
	plt.savefig(f'figs/heatmap_{j}_DAGGNN.pdf')
exit()
for j in trange(0, 514, 10):
	w = np.load('results/NOBEARS_%d.npz' %(j))
	# , W_est=W_est, W_est_init = W_est_init  
	print(w['W_est'], w["W_est"].shape)
	plt.figure(figsize= [15, 14])
	ax = sns.heatmap(w['W_est'], linewidth=0.2, xticklabels=False, yticklabels=False)
	# plt.colormap()
	plt.tight_layout()
	plt.savefig(f'figs/heatmap_{j}_pred.pdf')

	plt.figure(figsize= [15, 14])
	ax = sns.heatmap(X[j], linewidth=0.2, xticklabels=False, yticklabels=False)
	# plt.colormap()
	plt.tight_layout()
	plt.savefig(f'figs/heatmap_{j}_original.pdf')

