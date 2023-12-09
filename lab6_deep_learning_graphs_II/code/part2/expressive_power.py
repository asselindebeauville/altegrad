"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import itertools
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
Gs = [nx.cycle_graph(i) for i in range(10, 20)]


############## Task 5
        
adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
x = np.ones((adj.shape[0], 1))
idx = []
for i, G in enumerate(Gs):
    idx.extend([i] * G.number_of_nodes())

adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
x = torch.FloatTensor(x).to(device)
idx = torch.LongTensor(idx).to(device)


############## Task 8
        
for neighbor_aggr, readout in itertools.product(['sum', 'mean'], ['sum', 'mean']):
    model = GNN(1, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
    print(f'Task 8: {neighbor_aggr=}, {readout=}')
    print(model(x, adj, idx))
    print('-'*79)


############## Task 9
        
G1 = nx.union(nx.cycle_graph(3), nx.cycle_graph(3), rename=('A', 'B'))
G2 = nx.cycle_graph(6)


############## Task 10
        
Gs = [G1, G2]

adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
x = np.ones((adj.shape[0], 1))
idx = []
for i, G in enumerate(Gs):
    idx.extend([i] * G.number_of_nodes())

adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
x = torch.FloatTensor(x).to(device)
idx = torch.LongTensor(idx).to(device)


############## Task 11
        
neighbor_aggr, readout = 'sum', 'sum'
model = GNN(1, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
print(f'Task 11: {neighbor_aggr=}, {readout=}')
print(model(x, adj, idx))
