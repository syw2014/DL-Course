#!/usr/bin/env python
# Author  : Jerry.Shi
# File    : classify_main.py
# PythonVerion: python3.5
# Date    : 2017/4/7 14:23
# Software: PyCharm Community Edition

import sys
# Note: add module path to sys to tell sys where find your module
sys.path.append('D:\\github\\DL-Course\\cs231n\\assignment1\\classifier')

from load_data  import  *
from k_nearest_neighbor import *
import numpy as np
import  time

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


if __name__ == "__main__":
    dataSet = "D:\\data\\corpus\\cifar-10-batches-py"
    start_time = time()
    KNN(dataSet)
    end_time = time()
    print("Finshed Knn classification time cost: %0.3fs", (end_time-start_time))