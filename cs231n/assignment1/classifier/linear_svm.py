#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : linear_svm.py
# PythonVersion: python3.5
# Date    : 2017/4/13 9:08
# Software: PyCharm Community Edition

import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    A simple implementation of Structured SVM loss function
    :param W: C X D array of weights, C is the number of class, D is the number of sample dimension
    :param X: D X N array of data, data 
    :param y: 1-dimension array of class labels, the size of array is N number of class labels from 0,...,k-1
    :param reg: regularization , type float
    :return:  a tuple of loss and dW, the shape is as the same of W
    """

    # step 1, weight initialize the gradient value to 0
    dW = np.zeros(W.shape)

    # step 2, get class number and sample number
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0

    # step 3, compute loss and the gradient
    # traverse train samples
    for i in range(num_train):
        scores = W.dot(X[:, i]) # compute similarity of sample and all classes
        correct_class_score = scores[y[i]]
        # compute margin and gradient
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
            # compute gradient
            dW[y[i], :] -= X[:, i].T   ## if j != y_i
            dW[j, :] += X[:, i].T   # sum each contribution of X_i

    # step 4, process the loss
    # Right now loss is the sum over all training  sampels, but we want it to be an average instead so we divided by
    # num_train
    loss /= num_train

    # step 5
    # add regularization
    loss += 0.5 * reg * np.sum(W * W)

    # step 6, process gradient
    dW /= num_train

    # step 7, Gradient regularization that carries through per https://piazza.com/class/i37qi08h43qfv?cid=118
    dW += reg * W

    return loss, dW