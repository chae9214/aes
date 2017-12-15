# -*- coding: utf-8 -*-

import numpy as np
import codecs
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor as LT
import torch.nn.functional as F

# =================================================
# GloVe Word Vectors
# =================================================

def load_glove(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:     # on Linux
    # with open(path) as f:                                   # on Mac
        glove = {}
        for line in f.readlines():
            values = line.split()
            word = ''.join(values[0:-300])
            vector = np.array(values[-300:], dtype='float32')
            glove[word] = vector
        return glove

def load_embeddings(glove, word2idx, e_dim=300):
    embeddings = np.zeros((len(word2idx), e_dim))
    for word in glove.keys():
        index = word2idx.get(word)
        if index:
            vector = np.array(glove[word][1], dtype='float32')
            embeddings[index] = vector
    return torch.from_numpy(embeddings).float()

# =================================================
# CNN Model
# =================================================

class CNNModel(nn.Module):
    def __init__(self, n, e_dim, h_dim, dropout):
        super(CNNModel, self).__init__()
        self.e_dim = e_dim
        self.h_dim = h_dim

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(n, self.e_dim)
        self.init_weights_uniform()

        Ci = 1 # Number of channels in the input image
        Co = 100 # Number of channels produced by the convolution
        Ks = [3, 4, 5]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, self.e_dim)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.fc1 = nn.Linear(len(Ks)*Co, 1)

    def init_weights_uniform(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def init_weights_glove(self, glove_path, word2idx):
        glove = load_glove(glove_path)
        self.embedding.weight.data.copy_(load_embeddings(glove, word2idx))

    def forward(self, x):
        # x = [[11, 21, 31], [12, 22, 32]]
        # x.size() == [seqs, batch_size]
        x = self.embedding(Variable(LT([list(x_) for x_ in x]).cuda(), requires_grad=False))
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        out = self.fc1(x)  # (N,C)
        return out

# =================================================
# LSTM Model
# =================================================

class LSTMModel(nn.Module):

    def __init__(self, n, e_dim, h_dim, dropout, biLSTM = False):
        super(LSTMModel, self).__init__()
        self.e_dim = e_dim
        self.h_dim = h_dim

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(n, self.e_dim)
        self.biLSTM = biLSTM
        self.init_weights_uniform()

        self.encoder = nn.LSTM(self.e_dim, self.h_dim, bidirectional = biLSTM)

        self.fc1 = nn.Linear(self.h_dim, 2048)

        if(biLSTM == True):
            self.fc2 = nn.Linear(4096, 1)
            self.bn = nn.BatchNorm1d(4096)
        else:
            self.fc2 = nn.Linear(2048, 1)
            self.bn = nn.BatchNorm1d(2048)

    def init_weights_uniform(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def init_weights_glove(self, glove_path, word2idx):
        glove = load_glove(glove_path)
        self.embedding.weight.data.copy_(load_embeddings(glove, word2idx))

    def init_hidden(self, batch_size):
        if self.biLSTM == True:
            return Variable(torch.zeros((2, batch_size, self.h_dim)), requires_grad=False)
        else:
            return Variable(torch.zeros((1, batch_size, self.h_dim)), requires_grad=False)

    def forward(self, x):
        # x = [[11, 21, 31], [12, 22, 32]]
        # x.size() == [seq_len, batch_size]
        x = self.embedding(Variable(LT([list(x_) for x_ in x]).cuda(), requires_grad=False))
        # x.size() == [seq_len, batch_size, e_dim]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        c = self.init_hidden(batch_size)
        h = h.cuda()
        c = c.cuda()
        x = x.contiguous()
        out, (h, c) = self.encoder(x, (h, c))
        # out.size() == [seq_len, batch_size, h_dim]
        h = h.squeeze()
        # h.size() == [batch_size, h_dim]
        out = self.fc1(h)           # h_dim -> 2048

        if(self.biLSTM == True):
            out = torch.cat((out[0], out[1]), 1)

        out = self.bn(out)          # 2048 -> 2048
        out = F.leaky_relu(out)     # activation
        out = self.dropout(out)
        out = self.fc2(out)         # 2048 -> 1
        # out.size() == [batch_size, 1]
        return out