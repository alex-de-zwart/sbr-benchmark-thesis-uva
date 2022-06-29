#for hyper param optimization
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

# Import Python built-in libraries
import os
import copy
import pickle
import random
import time
import ipdb

import math

# Import pip libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

# Import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch_geometric.utils import add_self_loops, degree

# Import PyG packages
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_geometric.utils
from torch_geometric.typing import Adj, OptTensor
import torch_sparse

import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

# from dataio.sessionloader.latency_writter import *
# from dataio.predictions import PredictionsWriter
# from algorithms.evaluation.utils import SimpleStopwatch
# from algorithms.evaluation.metrics.coverage import Coverage
# from algorithms.evaluation.metrics.popularity import Popularity
# from algorithms.evaluation.metrics.accuracy import MRR, HitRate
# from algorithms.evaluation.metrics.accuracy_multiple import Precision, Recall, MAP, NDCG


class GraphDataset(pyg_data.InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        self.file_name = file_name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.txt']

    @property
    def processed_file_names(self):
        return [f'{self.file_name}.pt']

    def download(self):
        pass

    def process(self):
        raw_data_file = f'{self.raw_dir}/{self.raw_file_names[0]}'
        with open(raw_data_file, 'rb') as f:
            sessions = pickle.load(f)
        data_list = []

        for session in range(len(sessions[0])):
            session, y = sessions[0][session], sessions[1][session]
            codes, uniques = pd.factorize(session)
            senders, receivers = codes[:-1], codes[1:]

            # Build Data instance
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
            y_next = y
            y = torch.tensor([y[0]], dtype=torch.long)
            data_list.append(pyg_data.Data(x=x, edge_index=edge_index, y=y, y_next=y_next))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
class GatedSessionGraphConv(pyg.nn.conv.MessagePassing):
    def __init__(self, out_channels, aggr: str = 'add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels

        self.gru = torch.nn.GRUCell(out_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        m = self.propagate(edge_index, x=x, size=None)
        x = self.gru(m, x)
        return x

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

class SRGNN(nn.Module):
    def __init__(self, hidden_size, n_items):
        super(SRGNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = n_items

        self.embedding = nn.Embedding(self.n_items, self.hidden_size)
        self.gated = GatedSessionGraphConv(self.hidden_size)

        self.q = nn.Linear(self.hidden_size, 1)
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch_map = data.x, data.edge_index, data.batch

        # (0)
#         embedding = self.embedding(x).squeeze()
        
        if len(x) == 1:
          embedding = self.embedding(x).squeeze().unsqueeze(0) 
        else:
          embedding = self.embedding(x).squeeze()

        # (1)-(5)
        v_i = self.gated(embedding, edge_index)

        # Divide nodes by session
        # For the detailed explanation of what is happening below, please refer
        # to the Medium blog post.
        sections = list(torch.bincount(batch_map).cpu())
        v_i_split = torch.split(v_i, sections)

        v_n, v_n_repeat = [], []
        for session in v_i_split:
            v_n.append(session[-1])
            v_n_repeat.append(
                session[-1].view(1, -1).repeat(session.shape[0], 1))
        v_n, v_n_repeat = torch.stack(v_n), torch.cat(v_n_repeat, dim=0)

        q1 = self.W_1(v_n_repeat)
        q2 = self.W_2(v_i)

        # (6)
        alpha = self.q(F.sigmoid(q1 + q2))
        s_g_split = torch.split(alpha * v_i, sections)

        s_g = []
        for session in s_g_split:
            s_g_session = torch.sum(session, dim=0)
            s_g.append(s_g_session)
        s_g = torch.stack(s_g)

        # (7)
        s_l = v_n
        s_h = self.W_3(torch.cat([s_l, s_g], dim=-1))

        # (8)
        z = torch.mm(self.embedding.weight, s_h.T).T
        return z

def test(loader, test_model, is_validation=False):
    test_model.eval()

    # Define K for Hit@K metrics.
    k = 20
    correct = 0
    top_k_correct = []
    top_k_mrr = []

    for _, data in enumerate(tqdm(loader)):
        data.to('cpu')
        with torch.no_grad():
            score = test_model(data)
            pred = score.max(dim=1)[1]
            label = data.y

        correct += pred.eq(label).sum().item()


        sub_scores = score.topk(20)[1]
        sub_scores = sub_scores.cpu().detach().numpy()
        #test to calculate MRR

        for ele in range(sub_scores.shape[0]):
          top_k_pred = sub_scores[ele]

          if label[ele].item() in top_k_pred:
            top_k_correct.append(1)
            position = np.where(top_k_pred == label[ele].item())[0][0] + 1
            top_k_mrr.append(1 / position)
          else:
            top_k_mrr.append(0) 
            top_k_correct.append(0) 

    mrr = np.mean(top_k_mrr)
    hit_rate = np.mean(top_k_correct)

    print(f'The MRR@20 : {mrr} HitRate@20 : {hit_rate}')  
  
    if not is_validation:
        return correct / len(loader), hit_rate ,  mrr
    else:
        return correct / len(loader), hit_rate , mrr