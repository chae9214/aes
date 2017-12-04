# -*- coding: utf-8 -*-

import os
import argparse
import time
import math
import pickle
import torch
from torch.utils.data import Dataset

from preprocess import Preprocessor
from model import CNNModel, LSTMModel
from train import train, evaluate

# =================================================
# Utility functions
# =================================================

def parse_args():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--data', type=str, default='./data/',
                        help='path to load data')
    parser.add_argument('--save', type=str, default='./save/',
                        help='path to save model')
    # set/model/embedding choice
    parser.add_argument('--set', type=str, default='1',
                        help="essay set to use (1-8)")
    parser.add_argument('--model', type=str, default='LSTM',
                        help="model to use (CNN, LSTM, Bi-LSTM)")
    parser.add_argument('--embed', type=str, default='glove',
                        help="embedding to use (None, glove)")
    # parameters to tune
    parser.add_argument('--e_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--h_dim', type=int, default=200,
                        help='hidden dimension')
    parser.add_argument('--batch', type=int, default=20,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers')
    # boolean arguments
    parser.add_argument('-n', '--noise', action='store_true',
                        help='use noise')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='use CUDA')
    args = parser.parse_args()
    return args

def pprint(filepath, s):
    print(s)
    open(filepath, 'a').write(s)

# =================================================
# Corpus class
# =================================================

class Corpus(Dataset):

    def __init__(self, filename):
        _data = pickle.load(open(filename, 'rb'))
        self.data = [_data[key] for key in _data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =================================================
# Main
# =================================================

if __name__=='__main__':
    args = parse_args()
    datapath = os.path.join(args.data, "essayset_{}".format(args.set))
    savefile = "model_{}_{}--set{}_emb{}_hid{}_bat{}_epc{}.pt".format(args.model, args.embed, datapath.strip('/')[-1],
                                                                      args.e_dim, args.h_dim, args.batch, args.epochs)
    logfile = "log_{}_{}--set{}_emb{}_hid{}_bat{}_epc{}.pt".format(args.model, args.embed, datapath.strip('/')[-1],
                                                                   args.e_dim, args.h_dim, args.batch, args.epochs)
    savefile_path = os.path.join(args.save, savefile)
    logfile_path = os.path.join(args.save, logfile)

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    ### Load preprocessed data
    pprint(logfile_path, '=' * 89)
    pprint(logfile_path, 'preprocessing data...')
    pprint(logfile_path, '=' * 89)

    glove_path = "./data/glove.840B.300d.txt"

    preprocessor = Preprocessor(datapath)
    train_data = Corpus(os.path.join(datapath, 'train.dat'))
    valid_data = Corpus(os.path.join(datapath, 'valid.dat'))
    test_data = Corpus(os.path.join(datapath, 'test.dat'))

    ### Build model
    pprint(logfile_path, '=' * 89)
    pprint(logfile_path, 'building model...')
    pprint(logfile_path, '=' * 89)

    n = len(preprocessor.vocab)

    if args.model == 'LSTM':
        model = LSTMModel(n, args.e_dim, args.h_dim)
    elif args.model == 'CNN':
        model = CNNModel()

    if args.embed == 'glove':
        model.init_weights_glove(glove_path, preprocessor.word2idx)

    if args.cuda:
        model.cuda()

    ### Train model
    pprint(logfile_path, '=' * 89)
    pprint(logfile_path, 'training model...')
    pprint(logfile_path, '=' * 89)

    lr = args.lr
    best_val_loss = None

    for epoch in range(1, args.epochs):
        epoch_start_time = time.time()
        train(model, train_data, args.batch, args.noise)
        val_loss = evaluate(model, valid_data, args.batch)
        pprint(logfile_path, '-' * 89)
        pprint(logfile_path, '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        pprint(logfile_path, '-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(savefile_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss

    pprint(logfile_path, '-' * 89)
    pprint(logfile_path, 'best_val_loss: ' + best_val_loss)
    pprint(logfile_path, '-' * 89)

    with open(savefile_path, 'rb') as f:
        model = torch.load(f)

    ### Run on test data
    test_loss = evaluate(model, test_data, args.batch)
    pprint(logfile_path, '=' * 89)
    pprint(logfile_path, '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    pprint(logfile_path, '=' * 89)

