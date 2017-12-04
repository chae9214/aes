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

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_metrics(y, y_):
    rmse_row = rmse(y, y_)
    r_row, p_value = pearsonr(y, y_)
    s_row, p_value = spearmanr(y, y_)
    return rmse_row, r_row, s_row



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
            met_rmse, met_pearsonr, spearmanr = get_metrics(y_.data.float().cpu().numpy().flatten(),
                                                            targets.data.float().cpu().numpy().flatten())
            print('BATCH {}|RMSE : {:3.3f} |PEARSON R : {:3.3f}|SPEARMAN R : {:3.3f}'.format(batch, met_rmse, met_pearsonr,
                                                                                          spearmanr))

        total_loss += len(data) * loss.data
    return total_loss[0] / len(test_data)