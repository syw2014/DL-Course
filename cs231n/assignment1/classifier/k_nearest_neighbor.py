#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : k_nearest_neighbor.py
# PythonVerion: python3.5
# Date    : 2017/4/7 10:11
# Software: PyCharm Community Edition

import numpy as np
from collections import Counter

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """X is N x D where each row is an example. Y is 1-Dimension with size N"""
        # the nearest neighbor classifier simply to remember all the training data
        self.X = X
        self.y = y

    def predict(self, X, k=1, loop_num=0):

        if loop_num == 0:
            dists = self.compute_distacne_no_loop(X)
        elif loop_num == 1:
            dists = self.compute_distance_one_loop(X)
        elif loop_num == 2:
            dists = self.compute_distance_two_loop(X)
        else:
            raise ValueError('Invalide value %d for loop number' % loop_num)

        return self.predict_labels(dists, k=k)


    # different predict label calculation methods
    # in this part we only use L2 distance (Euclidean distance), and there are three implementation

    def compute_distance_two_loop(self, X):
        """
        compute the distance between every test data with every training data point,so there are two loop
        :param X: test data, each row is a single sample
        :return: dists[i, j], the distance between i-th test data and j-th training data
        """
        # get test data size
        num_test = X.shape[0]
        # get train data size
        num_train = self.X.shape[0]

        # final result array
        dists = np.zeros((num_test, num_train))

        # calculate distance i-th test data and j-th training data
        for i in range(num_test):
            for j in range(num_train):

                # l1 distance
                dists[i, j] = np.linalg.norm(self.X[j, :] - X[i, :])
                # l2 distance, slow
                # dists[i, j] = np.sqrt(np.sum(np.square(self.X[j, :] - X[i, :])))
        return dists

    def compute_distance_one_loop(self, X):
        """
        compute the distance between each test data point between trainding data with one loop over test data
        :param X: test data, each row is a single sample
        :return: dists[i, j], the distance between i-th test data and j-th training data
        """
        # get test data size
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            # l1 distance
            dists[i, :] = np.linalg.norm(self.X - X[i, :], axis=1)

            # l2 distance
            # dists[i, :] = np.sum(np.abs(self.X - X[i,:])**2,axis=-1)**(1./2)
        return dists

    def compute_distacne_no_loop(self, X):
        """
        compute distance between test data and training data without explicit loop
        :param X: test data, each row is a single sample
        :return: dists[i, j], the distance between i-th test data and j-th training data
        """
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))

        # as we know , test data and traing data are all vectors, so we can use matrix multiplication to complete
        # l2 distance calculation
        # Dist = (X_train_matrix - X_test_matrix) ** 2
        # = X_train_matrix^T * X_train_matrix - 2 * X_train_matrix * X_test_matrix + X_test_matrix^T * X_test_matrix

        M = 2 * np.dot(X, self.X.T)
        test = np.square(X).sum(axis=1)
        train = np.square(self.X).sum(axis=1)

        dists = np.sqrt(train  - M + np.matrix(test).T)
        return dists

    def predict_labels(self, dists, k=1):
        """
        get the top k nearest distance then return is't related label as the predict result
        :param dists: an array num_test * num_train size, each element indicates the distance i-th test datd point between
                        j-th train data point
        :param k: number to select top nearest distance
        :return: predict label
        """
        num_test = dists.shape[0]
        predict = np.zeros(num_test)

        for i in range(num_test):
            # closest predict y
            closest_y = []
            # use np.argsort to get the sorted index of every distance
            labels = self.y[np.argsort(dists[i, : ])].flatten()
            if k > len(labels):
                k = len(labels)

            closest_y = labels[:k]

            # Counter automatically breaks ties the right way (by choosing the smaller label):
            # >>> Counter([3, 2, 1, 3, 3, 3, 4, 1, 1, 1]).most_common(1)
            # [(1, 4)]
            # >>> Counter([1, 2, 3, 1, 1, 1, 4, 3, 3, 3]).most_common(1)
            # [(1, 4)]
            # print closest_y.shape
            c = Counter(closest_y)
            predict[i] = c.most_common(1)[0][0]

        return predict