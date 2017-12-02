# -*- coding: utf-8 -*-

import argparse
import time
import math
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from preprocess import Preprocess
from model import LSTMModel

# =================================================
# Parse arguments
# =================================================

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='./data/training_set_rel3.tsv',
                    help='path to load data')
parser.add_argument('-s', '--save', type=str,  default='./save/model.pt',
                    help='path to save model')
parser.add_argument('-m', '--model', type=str, default='LSTM',
                    help="model to use (LSTM)")
parser.add_argument('-c', '--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

EMBED_SIZE = 300
BATCH_SIZE = 20
NUM_HID = 200
NUM_LAY = 2
INIT_LR = 20
EPOCHS = 5
BPTT = 35

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

# =================================================
# Load preprocessed data
# =================================================

print('=' * 89)
print('preprocessing data...')
print('=' * 89)

glove_path = "./data/glove.840B.300d.txt"
data_path = "./data/"

corpus = Preprocess(glove_path, data_path) # TODO: variable names can be misleading

class Corpus(Dataset):

    def __init__(self, filename):
        _data = pickle.load(open(filename, 'rb'))
        self.data = [_data[key] for key in _data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train = Corpus('data/train.dat')
valid = Corpus('data/valid.dat')
test = Corpus('data/test.dat')

# =================================================
# Build model
# =================================================

print('=' * 89)
print('building model...')
print('=' * 89)

n = len(corpus.vocab)

# model = RNNModel(args.model, n, EMBED_SIZE, NUM_HID, NUM_LAY)
model = LSTMModel(n, EMBED_SIZE, NUM_HID, glove_path, corpus.word2idx)
if args.cuda:
    model.cuda()

# criterion = nn.CrossEntropyLoss()

# =================================================
# Utility functions
# =================================================

# *** do not use ***
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
# train_data = batchify(corpus.train, BATCH_SIZE)
# valid_data = batchify(corpus.valid, eval_batch_size)
# test_data = batchify(corpus.test, eval_batch_size)

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# *** do not use ***
def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

# =================================================
# Train model
# =================================================

print('=' * 89)
print('training model...')
print('=' * 89)

def evaluate(data_source):
    model.eval()
    total_loss = 0
    n = len(corpus.vocab)
    # hidden = model.init_hidden(eval_batch_size)
    mse = nn.MSELoss()
    data

    for i in range(0, len(data_source) - 1, args.bptt): # TODO: fix loop (bptt vs batch)
        # data, targets = get_batch(data_source, i, evaluation=True)
        dataloader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)
        data, targets = next(iter(dataloader))

        output = model(data)

        y_ = output.view(-1, n)
        loss = mse(y_, targets)

        total_loss += len(data) * loss.data
        # hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def train():
    model.train()
    total_loss = 0
    start_time = time.time()
    # hidden = model.init_hidden(args.batch_size)
    optim = Adam(model.parameters())
    mse = nn.MSELoss()
    dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    for batch, i in enumerate(range(0, len(train) - 1, args.bptt)): # TODO: fix loop (bptt vs batch)
        # data, targets = get_batch(train_data, i)

        # x, y = next(iter(dataloader))
        # type(x) == <class 'list'>
        # type(x[0]) == <class 'tuple'> of size batch_size
        # type(y) == <class 'torch.LongTensor'> of size batch_size
        data, targets = next(iter(dataloader))

        # hidden = repackage_hidden(hidden)
        model.zero_grad()
        output = model(data)

        y_ = output.view(-1, n)
        loss = mse(y_, targets)

        # loss = criterion(output.view(-1, n), targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

lr = INIT_LR
best_val_loss = None

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(valid)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        lr /= 4.0

# =================================================
# Run on test data
# =================================================

# with open(args.save, 'rb') as f:
#     model = torch.load(f)

# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)