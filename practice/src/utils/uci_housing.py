#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Date: 2017-09-22 11:27:30
import numpy as np
import random
import math

"""
UCI Housing dataset.
Parse train and test set from original data in '/data/housing.data'
"""

def load_data(filename, feature_num=14, ratio=0.8):
    """
    Load housing data and normalize feature data,then split it into train and test set.
    """
    data = np.fromfile(filename, sep=' ')
    data = data.reshape(data.shape[0] / feature_num, feature_num)
    # normalize, find maxmum, mimum, avgs in each column
    maxmums, minmums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0) / data.shape[0]
    for i in range(feature_num - 1):
        data[:, 1] = (data[:, i] - avgs[i]) / (maxmums[i] - minmums[i])
    offset = int(data.shape[0] * ratio)
    uci_train = data[:offset]
    uci_test = data[offset:]
    return uci_train, uci_test
    #uci_train_x , uci_train_y = uci_train[:, 0:-1], uci_train[:, -1].reshape(uci_train.shape[0], 1)
    #uci_test_x, uci_test_y = uci_test[:, 0:-1], uci_test[:, -1].reshape(uci_test.shape[0], 1)
    #return uci_train_x, uci_train_y, uci_test_x, uci_test_y

class BatchManager(object):
    def __init__(self, data, batch_size):
        self.batch_data = self.create_batches(data, batch_size)
        self.num_batch = len(self.batch_data)

    def create_batches(self, data, batch_size):
        # data prasing
        num_batch = int(math.ceil(len(data) / batch_size))
        x = data[:, 0:-1]
        y = data[:, -1].reshape(x.shape[0], 1)
        batch_data = list()
        print num_batch,batch_size, len(data)
        for i in range(num_batch):
            batch_data.append([x[i*batch_size : (i+1)*batch_size], y[i*batch_size: (i+1)*batch_size]])
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.num_batch):
            yield self.batch_data[idx]
