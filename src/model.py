from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

num_nodes = 5
days = 30
training = True

def generate_data(csv_file):
    data = np.genfromtxt(csv_file, delimiter=',')[:, 1:]
    data = data[1:] / data[:-1] - 1
    return data

graph = tf.Graph()
with graph.as_default():
    # Variables
    # Input gate
    x = tf.placeholder([num_nodes, days])
    ix = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([num_nodes]))
    # Forget gate
    fx = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.ones([num_nodes]))
    # Memory gate
    cx = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([num_nodes]))
    # Output gate
    ox = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([num_nodes]))
    # Classifier weights and biases
    w = tf.Variable(tf.random_uniform([num_nodes, num_nodes]))
    b = tf.Variable(tf.zeros([num_nodes]))

    def lstm_cell(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(ix, i) + tf.matmul(im, o) + ib)
        forget_gate = tf.sigmoid(tf.matmul(fx, i) + tf.matmul(fm, o) + fb)
        update = tf.matmul(cx, i) + tf.matmul(cm, o) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(ox, i) + tf.matmul(om, o) + ob)
        return output_gate * tf.tanh(state), state

    y = list()
    output = tf.zeros([num_nodes])
    state = tf.zeros([num_nodes])
    unrolling_less = 0
    if training: unrolling_less = 1
    for i in range(days-unrolling_less):
        output, state = lstm_cell(x[i, :], output, state)
    x = x[1:]

    logits = tf.matmul(w, tf.concat(0, y)) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, x)))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    prediction = tf.matmul(w, y[-1]) + b