#!/usr/bin/env python
# Author  : Jerry.Shi
# File    : classify_main.py
# PythonVersion: python3.5
# Date    : 2017/4/7 14:23
# Software: PyCharm Community Edition

import sys
# Note: add module path to sys to tell sys where find your module
sys.path.append('D:\\github\\DL-Course\\cs231n\\assignment1\\classifier')

from load_data  import  *
from k_nearest_neighbor import *
import numpy as np
import  time
import matplotlib.pyplot as plt

def KNN(data_path):
    Xtr, Ytr, Xte, Yte = load_CIFAR10(data_path)

    # flatten all image data to one-dimension
    Xtr_row = Xtr.reshape(Xtr.shape[0], 32*32*3) # become 50000*3072
    Xte_row = Xte.reshape(Xte.shape[0], 32*32*3) # become 10000*3072

    # create nn classifier
    nn = NearestNeighbor()

    # train, which means to record all the words
    nn.train(Xtr_row, Ytr)

    # predict
    Yte_pre = nn.predict(Xte_row)

    print('accuracy: %f' % (np.mean(Yte_pre == Yte)))

def CVKnn(data_path):
    """
    Cross Validation to find the best hyperparameter K for neighbors
    :param data_path: data set path
    :return: plot the k and accuracy curve
    """

    # load the full data from dataset
    Xtr, Ytr, Xte, Yte = load_CIFAR10(data_path)
    # flatten all image data to one-dimension
    Xtr_row = Xtr.reshape(Xtr.shape[0], 32*32*3) # become 50000*3072
    Xte_row = Xte.reshape(Xte.shape[0], 32*32*3) # become 10000*3072

    # choose 1000 data as validation , left as training data
    XtrVal_row = Xte_row[:1000, :]
    YtrVal_row = Ytr[:1000]


    # left 49,000 as train samples
    Xtr_row = Xte_row[1000:, :]
    Ytr = Ytr[1000:]

    validation_acc = []
    nn = NearestNeighbor()
    for k in [1, 3, 5, 10, 20, 50, 100]:
        nn.train(Xtr_row, Ytr)
        pred = nn.predict(XtrVal_row, k = k)
        acc = np.mean(YtrVal_row == pred)
        print('accuracy: %f' % (acc))
        validation_acc.append((k, acc))

    npArr = np.array(validation_acc)
    plt.plot(npArr[:, 0], npArr[:, 1],  'b-', lw=1.5)
    plt.show()


if __name__ == "__main__":
    dataSet = "D:\\data\\corpus\\cifar-10-batches-py"
    start_time = time.time()
    # KNN(dataSet)
    CVKnn(dataSet)
    end_time = time.time()
    print("Finshed Knn classification time cost: %0.3fs" % (end_time-start_time))