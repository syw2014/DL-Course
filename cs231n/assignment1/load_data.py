#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : load_data.py
# Date    : 2017/4/6 17:43
# Version : python3.5
# Software: PyCharm Community Edition

"""Descriptions:
        A module to load data from CIFAR image dataset. From the interface provided load train sample and test sample.
        The variable names are just like symbols as in course.As we know CIFAR10 there were six training sample zip,
        and a test sample, detail descriptions about this dataset can be found https://www.cs.toronto.edu/~kriz/cifar.html
        , and we use <CIFAR-10 python version>. This part was reference <https://github.com/MyHumbleSelf/cs231n.git>"""


import pickle
import os
import numpy as np

def load_CIFAR_batch(filename):
    """
    Load dataset from given file in batch way
    :param filename:
    :return: train data as array, label array
    """
    with open(filename, 'r') as ifs:
        dataDict = pickle.load(ifs)

        # extract samples and it's label
        sampleX = dataDict['data']
        labelY = dataDict['label']

        # convertion
        sampleX = sampleX.reshape(10000,3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        labelY = np.array(labelY)
    return sampleX, labelY


def load_CIFAR10(path):
    """
    Given data path load data
    :param path: data directory
    :return: training data, training label, test data, test label
    """
    sampleList = []
    labelList = []
    # load all the data, as there only five training samples name as data_batch_id
    for i in range(1, 6):
        # get full filename
        filename = os.path.join(path, 'data_batch_%d' % (i, ))
        x, y = load_CIFAR_batch(filename)

        sampleList.append(x)
        labelList.append(y)

        # combine elements as one array
        Xtr = np.concatenate(sampleList)
        Ytr = np.concatenate(labelList)
    del x, y
    print("Training data loaded, total size : %d", len(Xtr))

    # load test data
    Xte, Yte = load_CIFAR_batch(os.path.join(path, 'test_bach'))


    return Xtr, Ytr, Xte, Yte