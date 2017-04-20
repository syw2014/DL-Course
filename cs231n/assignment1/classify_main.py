#!/usr/bin/env python
# Author  : Jerry.Shi
# File    : classify_main.py
# PythonVersion: python3.5
# Date    : 2017/4/7 14:23
# Software: PyCharm Community Edition

import sys
# Note: add module path to sys to tell sys where find your module
sys.path.append('D:\\github\\DL-Course\\cs231n\\assignment1\\classifier')


import numpy as np
import  time
import matplotlib.pyplot as plt
from load_data  import  *
from k_nearest_neighbor import *
from linear_svm import *
from gradient_check import *


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

def svm_classifier(dataset):
    """
    Classsify images to their categories by linear svm model
    :param dataset: training and test data path
    :return: Nope
    """

    # step 1, load dataset
    if len(dataSet) == 0:
        print("Require no-empty dataset but found %s") % (dataSet)
        sys.exit(-1)
    # load CIFAR10 is a util tool function in load_data.py
    X_train, y_train, X_test, y_test = load_CIFAR10(dataSet)

    # print the shape of every data
    print("Training data shape: ", X_train.shape)
    print("Training labels shape: ", y_train.shape)
    print("Test data shape: ", X_test.shape)
    print("Test labels shape: ", y_test.shape)


    # step 2, print training data
    # visualize images for every label, we only show 7 imgs for every label
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        # find the index of samples label == y in training samples
        idxs = np.flatnonzero(y_train == y)
        # random choose samples
        idxs = np.random.choice(idxs, samples_per_class, replace=False)

        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

    # step 3,preprocessing, subsampling
    num_training = 49000
    num_validation = 1000
    num_test = 1000

    # validation dataset
    mask = range(num_training, num_training + num_validation)
    # validataion data
    X_val = X_train[mask]
    y_val = y_train[mask]

    # train set
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # test set
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # print size information for every set
    print("Training data shape: ", X_train.shape)
    print("Cross validation triaining data shape: ", X_val.shape)
    print("Cross validation training labels shape: ", y_val.shape)
    print("Test data shape: ", X_test.shape)
    print("Test labels shaple: ", y_test.shape)

    # step 4, preprocessing, reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    print("Training data shape: ", X_train.shape)
    print("Test data shape: ", X_test.shape)
    print("Cross validation data shape: ", X_val.shape)

    # step 5.1, preprocessing, substract the mean img
    mean_img = np.mean(X_train, axis=0)
    print(mean_img[:10])

    # image show
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_img.reshape((32,32,3)).astype('uint8'))

    # step 5.2, subtract the mean image from train and test data
    X_train -= mean_img
    X_test -= mean_img
    X_val -= mean_img

    # step 5.3,  append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    # Also, lets transform both data matrices so that each image is a column.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T

    print("After appending bias, new shape are: ", X_train.shape, X_test.shape, X_val.shape)

    # step 6, classify
    # generate a random svm weight of small numbers
    W = np.random.randn(num_classes, X_train.shape[0]) * 0.0001
    loss, grad = svm_loss_naive(W, X_train, y_train, 0.00001)

    print("SVM naive training completed, loss: ", loss)

    # gradient check
    f = lambda w: svm_loss_naive(w, X_train, y_train, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)
    # print(grad_numerical)


if __name__ == "__main__":
    dataSet = "D:\\data\\corpus\\cifar-10-batches-py"
    # start_time = time.time()
    # # KNN(dataSet)
    # CVKnn(dataSet)
    # end_time = time.time()
    # print("Finshed Knn classification time cost: %0.3fs" % (end_time-start_time))

    # linear svm
    start_time = time.time()
    svm_classifier(dataSet)
    end_time = time.time()
    # print("Finshed Knn classification time cost: %0.3fs" % (end_time-start_time))