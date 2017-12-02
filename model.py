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

def load_embeddings(glove, word2idx, embedding_dim=300):
    embeddings = np.zeros((len(word2idx), embedding_dim))
    for word in glove.keys():
        index = word2idx.get(word)
        if index:
            vector = np.array(glove[word][1], dtype='float32')
            embeddings[index] = vector
    return torch.from_numpy(embeddings).float()

# =================================================
# LSTM Model
# =================================================

class LSTMModel(nn.Module):

    def __init__(self, n, d, h_dim, glove_path, word2idx):
        super(LSTMModel, self).__init__()
        # d = embedding_dimension
        self.d = d

        self.glove = load_glove(glove_path)
        self.word2idx = word2idx
        self.embedding = nn.Embedding(n, d)
        self.embedding.weight.data.copy_(load_embeddings(self.glove, self.word2idx))

        self.h_dim = h_dim
        self.encoder = nn.LSTM(self.d, self.h_dim)
        self.fc1 = nn.Linear(self.h_dim, 2048)
        self.fc2 = nn.Linear(2048, 1)
        self.bn = nn.BatchNorm1d(2048)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros((1, batch_size, self.h_dim)), requires_grad=False)

    def forward(self, x):
        # x = [[11, 21, 31], [12, 22, 32]]
        # x.size() == [seqs, batch_size]
        x = self.embedding(Variable(LT([list(x_) for x_ in x]).cuda(), requires_grad=False))

#        x = self.embedding(Variable(LT(x), requires_grad=False))
        # x.size() == [seqs, batch_size, embedding_dimension]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        c = self.init_hidden(batch_size)
        h = h.cuda()
        c = c.cuda()
        out, (h, c) = self.encoder(x, (h, c))
        # out.size() == [seqs, batch_size, h_dim]
        h = h.squeeze()
        # h.size() == [batch_size, h_dim]
        out = self.fc1(h)  # 1024 -> 2048
        out = self.bn(out)  # 2048 -> 2048
        out = F.leaky_relu(out)  # activation
        out = self.fc2(out)  # 2048 -> 1
        # out.size() == [batch_size, 1]
        return out

# =================================================
# RNN Model
# =================================================

class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(Model, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
