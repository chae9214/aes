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

def confusion_matrix(rater_a, rater_b,
                     min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None: min_rating = reduce(min, ratings)
    if max_rating is None: max_rating = reduce(max, ratings)
    num_ratings = max_rating - min_rating + 1
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b,
                             min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    scoreQuadraticWeightedKappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    scoreQuadraticWeightedKappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    score_quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

def get_metrics(y, y_):
    rmse_row = rmse(y, y_)
    r_row, p_value = pearsonr(y, y_)
    s_row, p_value = spearmanr(y, y_)
    c_row = quadratic_weighted_kappa(np.round(y), np.round(y_))
    return rmse_row, r_row, s_row, c_row



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
    y_cum = np.array([])
    targets_cum = np.array([])
    for batch, i in enumerate(dataloader):
        data, targets = next(iter(dataloader))
        targets = Variable(targets.float().cuda(), requires_grad=False)
        model.zero_grad()
        output = model(data)
        y_ = output.view(-1, 1)
        loss = mse(y_, targets)
#        if batch % 20 == 0 and batch > 0:  #
#            met_rmse, met_pearsonr, spearmanr, cohens = get_metrics(y_.data.float().cpu().numpy().flatten(),
#                                                            targets.data.float().cpu().numpy().flatten())
#            print('| BATCH {} | RMSE : {:3.3f} | PEARSON R : {:3.3f} | SPEARMAN R : {:3.3f} | COHEN KAPPA : {:3.3f} |'.format(
#                batch, met_rmse,met_pearsonr, spearmanr, cohens))
        y_cum = np.concatenate((y_cum, y_.data.float().cpu().numpy().flatten()))
        targets_cum = np.concatenate((targets_cum, targets.data.float().cpu().numpy().flatten()))
        total_loss += len(data) * loss.data
    met_rmse, met_pearsonr, spearmanr, cohens = get_metrics(y_cum, targets_cum)
    print('| RMSE : {:3.3f} | PEARSON R : {:3.3f} | SPEARMAN R : {:3.3f} | COHEN KAPPA : {:3.3f} |'.format(met_rmse,met_pearsonr, spearmanr, cohens))
    return total_loss[0] / len(test_data)