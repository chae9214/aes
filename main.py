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

train_data = Corpus('data/train.dat')
valid_data = Corpus('data/valid.dat')
test_data = Corpus('data/test.dat')

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

eval_batch_size = 10
# train_data = batchify(corpus.train, BATCH_SIZE)
# valid_data = batchify(corpus.valid, eval_batch_size)
# test_data = batchify(corpus.test, eval_batch_size)

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


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
    dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    for batch, i in enumerate(dataloader):  # TODO: fix loop (bptt vs batch)
        # data, targets = get_batch(data_source, i, evaluation=True)
        data, targets = next(iter(dataloader))

        targets = Variable(targets.float(), requires_grad=False)
        targets = targets.cuda()
        # hidden = repackage_hidden(hidden)
        model.zero_grad()
        output = model(data)
        y_ = output.view(-1, 1)
        loss = mse(y_, targets)
        total_loss += len(data) * loss.data
        # hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def train():
    model.train()
    total_loss = 0
    # hidden = model.init_hidden(args.batch_size)
    optim = Adam(model.parameters())
    mse = nn.MSELoss()
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    for batch, i in enumerate(dataloader):  # TODO: fix loop (bptt vs batch)
        start_time = time.time()
        #  data, targets = get_batch(train_data, i)
        # x, y = next(iter(dataloader))
        # type(x) == <class 'list'>
        # type(x[0]) == <class 'tuple'> of size batch_size
        # type(y) == <class 'torch.LongTensor'> of size batch_size
        data, targets = next(iter(dataloader))
        targets = Variable(targets.float().cuda(), requires_grad=False)
        # hidden = repackage_hidden(hidden)
        model.zero_grad()
        output = model(data)
        y_ = output.view(-1, 1)
        targets_noise = Variable(targets.data + torch.normal(means=torch.zeros(20), std=0.35).cuda(), requires_grad=False)
        loss = mse(y_, targets_noise)

        # loss = criterion(output.view(-1, n), targets)
        optim.zero_grad()
        loss.backward()
        optim.step()


        # loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % 10 == 0 and batch > 0: # @FIXME : Temporary value
            cur_loss = total_loss[0] / 10  # @FIXME : Temporary value
            elapsed = time.time() - start_time
            print('| {:5d} batches | ms/batch {:5.2f} | loss {:5.2f}'.format(
                     batch, elapsed * 1000 / 10, cur_loss)) # @FIXME : Temporary value
            total_loss = 0
            start_time = time.time()

lr = INIT_LR
best_val_loss = None

for epoch in range(1, 100): # @FIXME : Temporary value
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(valid_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    if not best_val_loss or val_loss < best_val_loss:
        with open("./model/model.pt", 'wb') as f: # @FIXME : Temporary value
            torch.save(model, f)
        best_val_loss = val_loss

print('-' * 89)
print('best_val_loss: '  + best_val_loss)
print('-' * 89)
# =================================================
# Run on test data
# =================================================

test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))


# with open(args.save, 'rb') as f:
#     model = torch.load(f)

# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)
