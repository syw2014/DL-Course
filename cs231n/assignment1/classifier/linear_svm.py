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
            # Note: not full understand !!!!
            dW[y[i], :] -= X[:, i].T   # compute the correct class gradient
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


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss, vectorized implementation
    :param W: C X D array of weights, C is the number of class, D is the number of sample dimension
    :param X: D X N array of data, data
    :param y: 1-dimension array of class labels, the size of array is N number of class labels from 0,...,k-1
    :param reg: regularization , type float
    :return:  a tuple of loss and dW, the shape is as the same of W
    """
    # step 1, create store variable
    # store the training loss in variable loss
    loss = 0.0
    dW = np.zeors(W.shape)   # gradient matrix

    # step 2, get use info for model
    D = X.shape[0]    # sample dimension
    num_classes = W.shape[0]  # number of class labels
    num_train = X.shape[1]    # number of train sample

    # step 3, compute score of weight * trainning data
    scores = W.dot(X)

    # step 4, Construct correct_scores vecto
    # Construct correct_scores vector that is D x 1-dimension (or 1xD) so we can subtract out
    # where we append the "true" scores: [W*X]_{y_1, 1}, [W*X]_{y_2, 2}, ..., [W*X]_{y_D, D}
    # Using advanced indexing into scores: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    # Slow, sadly:
    # correct_scores = np.diag(scores[y,:])
    # Fast (index in both directions):
    correct_scores = scores[y, np.arange(num_train)]  # using the fact that all elements in y are < C == num_classes

    # step 5, compute margin
    mat = scores - correct_scores + 1 # delta =  1
    mat[y, np.arange(num_train)] = 0  # accounting for the j=y_i term we shouldn't count (subtracting 1 makes up for it since w_j = w_{y_j} in this case)

    # step 6, compute maximum
    thresh = np.maximum(np.zeros(num_classes, num_train), mat)

    # step 7, compute training loss
    loss = np.sum(thresh)
    loss /= num_train

    # step 8, add regularization
    loss += 0.5 * reg * np.sum(W * W)

    # step 9, compute gradient over vectorized
    # Binarize into integers
    binary = thresh
    binary[thresh > 0] = 1
    # Perform the two operations simultaneously
    # (1) for all j: dW[j,:] = sum_{i, j produces positive margin with i} X[:,i].T
    # (2) for all i: dW[y[i],:] = sum_{j != y_i, j produces positive margin with i} -X[:,i].T
    col_sum = np.sum(binary, axis=0)
    binary[y, range(num_train)] = -col_sum[range(num_train)]
    dW = np.dot(binary, X.T)

    # Divide
    dW /= num_train

    # Regularize
    dW += reg * W

    return loss, dW
