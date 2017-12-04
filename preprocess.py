# -*- coding: utf-8 -*-

import os
import codecs
import pickle
from nltk.tokenize import TweetTokenizer

# =================================================
# Functions for read/write
# =================================================

def pickle_write(filename, obj):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def pickle_read(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

# =================================================
# Utility functions
# =================================================

def equalize(data, length=500):  # data : list type
    if (len(data) > length):
        return data[:length]
    else:
        return data + ['<eos>'] * (500 - len(data))

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# =================================================
# Preprocessor class
# =================================================

class Preprocessor():

    def __init__(self, data_path):
        self.vocab, self.word2idx, self.idx2word = self.build_vocab(data_path)

        # self.glove = load_glove(glove_path)
        # self.embeddings = load_embeddings(self.glove, self.word2idx)

        train_filename = os.path.join(data_path, 'train.dat')
        valid_filename = os.path.join(data_path, 'valid.dat')
        test_filename = os.path.join(data_path, 'test.dat')

        if not os.path.exists(train_filename):
            self.train = self.tokenize(os.path.join(data_path, 'train.csv'))
            self.valid = self.tokenize(os.path.join(data_path, 'valid.csv'))
            self.test = self.tokenize(os.path.join(data_path, 'test.csv'))
            pickle_write(train_filename, self.train)
            pickle_write(valid_filename, self.train)
            pickle_write(test_filename, self.train)
        else:
            self.train = pickle_read(train_filename)
            self.valid = pickle_read(valid_filename)
            self.test = pickle_read(test_filename)

    def build_vocab(self, path):
        tknzr = TweetTokenizer()
        vocab = set()
        file_list = ['train.csv', 'valid.csv', 'test.csv']
        for file_name in file_list:
            with codecs.open(os.path.join(path, file_name), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    temp = line.replace('\ufeff', '').replace('\n', '').split(',')
                    data = tknzr.tokenize(''.join(temp[1:-1]))
                    vocab |= set(data)

        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = len(word2idx.keys()) * [0]
        for word in word2idx.keys():
            idx2word[word2idx[word]] = word
        return vocab, word2idx, idx2word

    def tokenize(self, path):
        tknzr = TweetTokenizer()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            data = {}
            # essay_id, essay, domain1_score
            for line in f.readlines():
                temp = line.replace('\ufeff', '').replace('\n', '').split(',')
                if(isNumber(temp[-1])):
                    data[temp[0]] = [[self.word2idx[x] for x in tknzr.tokenize(''.join(temp[1:-1]))], int(temp[-1])]
            return data

# =================================================
# Main
# =================================================

if __name__=='__main__':
    glove_path= '/home/hpc/PycharmProjects/AES_project/data/glove/glove.840B.300d.txt'
    data_path = '/home/hpc/PycharmProjects/AES_project/data/testset/'

    with open('/home/hpc/PycharmProjects/AES_project/data/result.txt', 'w') as f:
        print('initializing... Start')
        p = Preprocess(glove_path, data_path)
        print('initializing... End')

        print('Vocab: ', file=f)
        print(p.vocab, file=f)
        print('word2idx: ', file=f)
        print(p.word2idx, file=f)
        print('idx2word: ', file=f)
        print(p.idx2word, file=f)
        print('train: ', file=f)
        print(p.train, file=f)
        print('valid: ', file=f)
        print(p.valid, file=f)
        print('test: ', file=f)
        print(p.test, file=f)
