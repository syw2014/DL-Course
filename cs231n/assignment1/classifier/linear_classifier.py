#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : linear_classifier.py
# PythonVersion: python3.5
# Date    : 2017/4/8 11:17
# Software: PyCharm Community Edition
# Description: Use linear svm loss to train classification module

import sys
# Note: add module path to sys to tell sys where find your module
sys.path.append('D:\\github\\DL-Course\\cs231n\\assignment1\\classifier')

# import linear svm loss function
from linear_svm import *

import numpy as np


class LinearClassifier:

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        use batch samples to train classifier
        :param X: Input training sample , D X N, D is the dimension of data, each clomn is a point, N is the sample size
        :param y: training sample label, 1 X N dimension array, each value is one of 0 ... k -1
        :param learning_rate: learning rate for optimization
        :param reg: regularization strength
        :param num_iters: number of training iterators
        :param batch_size: input training samples in each iterator
        :param verbose: whether to print optimizated infos
        :return: an array of training loss calculated in each iterator
        """

        # step 1, get number of training samples and sample dimension, class numbers
        dim, num_train = X.shape
        # assume class id starts from 0,
        num_classs = np.max(y) + 1

        # step 2, initialize W matrix by a random value
        if self.W is None:
            self.W = np.random.rand(num_classs, dim) *  0.001

        # step 3, use stochastic gradient desent to optimize W
        loss_history = []

        for iters in range(num_iters):
            # batch samples and label for model train
            X_batch = None
            y_batch = None

            # sampling batch size samples and labels in training data
            # Hint: Use np.random.choice to generate indices. Sampling with
            # replacement is faster than sampling without replacement.
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[:, idx]
            y_batch = y[:, idx]

            # compute loss and gradient
            loss , grad = self.loss(X_batch, y_batch, reg)

            # store loss at current iterator
            loss_history.append(loss)

            # update weight by SGD(Stochastic Gradient Desent)
            self.W -= learning_rate * grad

            # whether to print infos for current iterator training
            if verbose and iters % 100 == 0:
                print("iteration %d / %d , loss is %f") % (iters, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Predict sample label use trained W
        :param X: input predict samples, size is D X N, D is the dimension of each data, N is the number of predict data
        :return: predict label, size is 1 X N, N is the number of predict data
        """
        # initialize predicted label array
        y_pred = np.zeros(X.shape[1])

        y_pred = np.argmax(np.dot(self.W, X), axis=0)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function , and it's der
        :param X_batch: trainging sample, size is N X batch_size
        :param y_batch: training label , size is 1 X batch_size
        :param reg: regularization strength
        :return: loss,training loss of this current iterator, grad as the sampe shape of self.W
        """
        pass

class LinearSVM(LinearClassifier):
    """
    A subclass that uses the Multiclass SVM loss function
    """
    def loss(self, X_batch, y_batch, reg):
        return  svm_loss_naive(self.W, X_batch, y_batch, reg)

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


