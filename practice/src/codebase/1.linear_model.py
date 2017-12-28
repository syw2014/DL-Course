#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-09-22 10:00:30

import tensorflow as tf
import sys
sys.path.append('../')
from utils.uci_housing import *

# step1, load data and process
housing_data = '../../../data/housing.data'
uci_train, uci_test = load_data(housing_data)

batch_size = 30
feature_dim = 13
train_data = BatchManager(uci_train, batch_size)
test_data = BatchManager(uci_test, 150)

print train_data.num_batch
#train_x, train_y, test_x, test_y = load_data(housing_data)
# step2, create graph
x = tf.placeholder(shape=[None, feature_dim],dtype=tf.float32, name='X')
y = tf.placeholder(shape=[None, 1],dtype=tf.float32, name='y')
W = tf.Variable(tf.zeros([feature_dim, 1]), dtype=tf.float32)
b = tf.Variable(tf.zeros([feature_dim]), dtype=tf.float32)

num_sample = x.shape[0]

pred = tf.matmul(x, W) + b
loss = tf.reduce_sum(tf.square(pred - y)) / (2*batch_size)
train_op = tf.train.AdamOptimizer(0.005).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    mim_loss = 1000.0
    for i in range(50000):
        for batch in train_data.iter_batch(shuffle=True):
            xx, yy = batch
            _, lo , ww= sess.run([train_op, loss, W], feed_dict={x: np.asarray(xx), y: np.asarray(yy)})
            print("epoch:{} batch loss: {:.3f}".format(i, lo))
            if lo < mim_loss:
                mim_loss = lo
        # evaluate
        #    if i % 5 == 0:
        #        print("weith:{}".format(ww))
    print("minimum loss: {:.3f}".format(mim_loss))
# step3, train model
# step4, evaluate
