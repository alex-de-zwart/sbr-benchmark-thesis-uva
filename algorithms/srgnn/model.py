#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""
import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

import ipdb    
import os
from time import time
import datetime
import math
import numpy as np
import torch
from torch import nn
from itertools import chain
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pickle
from functools import partial
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from itertools import chain
from algorithms.evaluation.metrics.coverage import Coverage
from algorithms.evaluation.metrics.popularity import Popularity
from algorithms.evaluation.metrics.accuracy import MRR, HitRate
from algorithms.evaluation.metrics.accuracy_multiple import Precision, Recall, MAP, NDCG
from utils import Data, build_graph 
from algorithms.evaluation.utils import SimpleStopwatch
from dataio.sessionloader.latency_writter import *
from dataio.predictions import PredictionsWriter
import pandas as pd


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, config, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = config['hidden_size']
        self.n_node = n_node
        self.batch_size = config['batch_size']
        self.nonhybrid = 'store_true'
        self.epoch = config['epoch']
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=config['step'])
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], weight_decay=config['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=config['lr_dc'])
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = np.array([np.array(xi) for xi in A])
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func

def load_data(data_dir, data_slice, train):
    if train == True:
        train = f'/item_views_train_tr.{data_slice}.txt'
        val = f'/item_views_train_valid.{data_slice}.txt'
        print(f'The data is trained on train: {train} \n and validation: {val} \n')
    else:
        train = f'/item_views_train.{data_slice}.txt'
        val = f'/item_views_test.{data_slice}.txt'
        print(f'The data is trained on train: {train} \n and test: {val} \n')

    train_data_original = pickle.load(open(data_dir + train, 'rb'))
    test_data_original = pickle.load(open(data_dir + val, 'rb'))

    n_nodes = max(set(chain(*train_data_original[0]))) + 1
    print(f'The number of nodes in the train set: {n_nodes}')

    train_data = Data(train_data_original, shuffle=True)
    test_data = Data(test_data_original, shuffle=False)

    return train_data, test_data, test_data_original, n_nodes

@timer_func
def train_hyper(config, checkpoint_dir=None, data_dir=None):

    train_data, test_data, n_node = load_data(data_dir, data_slice=1, train=True)
    model = trans_to_cuda(SessionGraph(config, n_node))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)

    for epoch in range(config['epoch']):
        print(f"[INFO]: Epoch {epoch + 1} of {10}")
        print('start training: ', datetime.datetime.now())
        model.train()
        total_loss = 0.0
        slices = train_data.generate_batch(model.batch_size)
        for i, j in zip(slices, np.arange(len(slices))):

            # zero the parameter gradients
            model.optimizer.zero_grad()
            targets, scores = forward(model, i, train_data)
            targets = trans_to_cuda(torch.Tensor(targets).long())
            loss = model.loss_function(scores, targets - 1)
            loss.backward()
            model.optimizer.step()  # method that updates the parameters.
            total_loss += loss
            if j % int(len(slices) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
        print('\tLoss:\t%.3f' % total_loss)
        
        print('start predicting: ', datetime.datetime.now())
        with torch.no_grad():
            model.scheduler.step()
            model.eval()
            hit, mrr = [], []
            slices = test_data.generate_batch(model.batch_size)
            for i in slices:
                targets, scores = forward(model, i, test_data)
                sub_scores = scores.topk(20)[1]
                sub_scores = trans_to_cpu(sub_scores).detach().numpy()
                for score, target, mask in zip(sub_scores, targets, test_data.mask):
                    hit.append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
            hit = np.mean(hit)
            mrr = np.mean(mrr)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), model.optimizer.state_dict()), path)

        tune.report(loss=total_loss, hit=hit, mrr=mrr)

    print("Finished Training")

@timer_func
def train(config, train_data, n_node, data_slice, checkpoint_dir=None, data_dir=None):
    model = trans_to_cuda(SessionGraph(config, n_node))

    for epoch in range(10):
        print(f"[INFO]: Epoch {epoch + 1} of {10}")
        print('start training: ', datetime.datetime.now())
        model.train()
        total_loss = 0.0
        slices = train_data.generate_batch(model.batch_size)
        for i, j in zip(slices, np.arange(len(slices))):

            # zero the parameter gradients
            model.optimizer.zero_grad()
            targets, scores = forward(model, i, train_data)
            targets = trans_to_cuda(torch.Tensor(targets).long())
            loss = model.loss_function(scores, targets - 1)
            loss.backward()
            model.optimizer.step()  # method that updates the parameters.
            total_loss += loss
            if j % int(len(slices) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
        print('\tLoss:\t%.3f' % total_loss)

    checkpoint_dir = '/home/ec2-user/SageMaker/sb-rec-system/algorithms/srgnn/trained_models'
    path = os.path.join(checkpoint_dir, f"model_{data_slice}")
    torch.save(model.state_dict(), path)

    print("Finished Training")

def main(num_samples=1, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..\..', 'data\prepared\srgnn'))
    checkpoint_dir = '/home/ec2-user/SageMaker/sb-rec-system/algorithms/srgnn/'
    config = {
        "l2": 1e-5,
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 100, 128, 256]),
        "hidden_size": tune.choice([50, 100, 125, 150, 200]),
        "step": tune.choice([1, 2, 3]),
        "lr_dc": 0.1
    }
    scheduler = ASHAScheduler(
        metric="mrr",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        max_progress_rows=10,
        metric="mrr",
        mode="max",
        metric_columns=["loss", "hit", "mrr"])
    result = tune.run(
        partial(train, data_dir=data_dir, checkpoint_dir=checkpoint_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("mrr", "max", "last")
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss : {best_trial.last_result["loss"]}')
    print(f'Best trial final validation MRR@20: {best_trial.last_result["mrr"]}')
          
          
          
def test_score(train_data, model, test_data, test_data_original, latencies_out_file, pred_out_file):
    with torch.no_grad():
        model.eval()

        metrics = [NDCG(), MAP(), Precision(), Recall(), HitRate(), MRR(), Coverage(training_df=train_data),
                   Popularity(training_df=train_data)]

        predictions_writer = PredictionsWriter(outputfilename=pred_out_file, evaluation_n=20)
        latency_writer = LatencyWriter(latencies_out_file)

        slices = test_data.generate_batch(model.batch_size)

        prediction_sw = SimpleStopwatch()

        score_exp = []
        for index, element in enumerate(slices):
            next_items = test_data_original[1][index]
            next_items = [target - 1 for target in next_items]
            prediction_sw.start()
            targets, scores = forward(model, element, test_data)
            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            recommendations = pd.Series(0.0, sub_scores[0])
            prediction_sw.stop(len(next_items))

            predictions_writer.appendline(recommendations, next_items)

            for metric in metrics:
                metric.add(recommendations, np.array(next_items))

        scores = []
        for metric in metrics:
            metric_name, score = metric.result()
            scores.append("%.4f" % score)
            print(metric_name, "%.4f" % score)
          
        score_exp.append(scores)
        predictions_writer.close()
        for (position, latency) in prediction_sw.get_prediction_latencies_in_micros():
            latency_writer.append_line(position, latency)
        latency_writer.close()
          
        with open('results/srgnn_model_performance_over_all_slices.csv', 'a+') as f:
            f.write(",".join(str(item) for item in scores))
            f.write('\n')
          
        
            
#         score_exp = pd.DataFrame(score_exp)
#         cols = ['NDCG@20', 'MAP@20', 'Precision@20', 'Recall@20', 'HitRate@20', 'MRR@20', 'Coverage@20', 'Popularity@20']
           
#         score_exp.to_csv('results/srgnn_model_performance_over_all_slices.csv', sep=';', decimal=",", header=cols, index=0)