# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix

# =================================================
# Utility functions
# =================================================

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def kappa_weights(n):
    s = np.linspace(1, n, n)
    w = np.power(np.subtract.outer(s, s), 2.0) / np.power(np.subtract.outer(n, np.ones((n,))), 2)
    return w

def quadratic_weighted_kappa(a, b, n = 20):
    a = np.clip(a.astype('int32') - 1, 0, n - 1)
    b = np.clip(b.astype('int32') - 1, 0, n - 1)

    sa = coo_matrix((np.ones(len(a)), (np.arange(len(a)), a)), shape=(len(a), n))
    sb = coo_matrix((np.ones(len(b)), (np.arange(len(b)), b)), shape=(len(a), n))

    O = (sa.T.dot(sb)).toarray()
    E = np.outer(sa.sum(axis=0), sb.sum(axis=0))
    E = np.divide(E, np.sum(E)) * O.sum()
    W = kappa_weights(n)

    return 1.0 - np.multiply(O, W).sum() / np.multiply(E, W).sum()

def get_metrics(y, y_):
    rmse_row = rmse(y, y_)
    r_row, p_value = pearsonr(y, y_)
    s_row, p_value = spearmanr(y, y_)
    k_row = quadratic_weighted_kappa(np.round(y), np.round(y_), len(y_))
    return rmse_row, r_row, s_row, k_row

# =================================================
# Train function
# =================================================

def train(model, train_data, batch_size, noise=False):
    model.train()
    total_loss = 0

    mse = nn.MSELoss()
    optim = Adam(model.parameters())
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for batch, i in enumerate(dataloader):
        start_time = time.time()
        data, targets = next(iter(dataloader))
        targets = Variable(targets.float().cuda(), requires_grad=False)
        model.zero_grad()
        output = model(data)

        y_ = output.view(-1, 1)
        if noise:
            targets_noise = Variable(targets.data + torch.normal(means=torch.zeros(20), std=0.35).cuda(),
                                     requires_grad=False)
            loss = mse(y_, targets_noise)
        else:
            loss = mse(y_, targets)
        total_loss += loss.data

        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch % batch_size == 0 and batch > 0:
            cur_loss = total_loss[0] / batch_size
            elapsed = time.time() - start_time
            print('| {:5d} batches | ms/batch {:5.2f} | loss {:5.2f}'.format(
                batch, elapsed * 1000 / 10, cur_loss))
            total_loss = 0
            start_time = time.time()

# =================================================
# Evaluate function
# =================================================

def evaluate(model, test_data, batch_size):
    model.eval()
    total_loss = 0

    mse = nn.MSELoss()
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for batch, i in enumerate(dataloader):
        data, targets = next(iter(dataloader))
        targets = Variable(targets.float().cuda(), requires_grad=False)
        model.zero_grad()
        output = model(data)

        y_ = output.view(-1, 1)
        loss = mse(y_, targets)
        if batch % 20 == 0 and batch > 0:  #
            met_rmse, met_pearsonr, spearmanr, kappa = get_metrics(y_.data.float().cpu().numpy().flatten(),
                                                                    targets.data.float().cpu().numpy().flatten())
            print('| BATCH {} | RMSE : {:3.3f} | PEARSON R : {:3.3f} | SPEARMAN R : {:3.3f} | KAPPA : {:3.3f} |'.format(
                batch, met_rmse, met_pearsonr, spearmanr, kappa))

        total_loss += len(data) * loss.data
    return total_loss[0] / len(test_data)