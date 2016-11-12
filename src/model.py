from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
import cStringIO
from six.moves.urllib.request import urlretrieve

num_nodes = 5
days = 30
stock = 'AAPL'

def generate_data(csv_file):
    with open(csv_file+'.csv', "r") as myfile:
        data = myfile.read().replace('"', '')
    data = np.genfromtxt(cStringIO.StringIO(data), delimiter=',')[1:, 1:]
    data = data.astype(float)
    data = data[1:] / data[:-1] - 1
    data = np.fliplr(data.transpose())
    train_data = data[:2240][:]
    valid_data = data[2240:][:]
    return train_data, valid_data

train_data, valid_data = generate_data('../res/'+stock)

graph = tf.Graph()
with graph.as_default():
    # Variables
    # Input gate
    x = tf.placeholder(tf.float32, shape=[num_nodes, days])
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
    b = tf.Variable(tf.zeros([num_nodes, 1]))

    def lstm_cell(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(ix, i) + tf.matmul(im, o) + ib)
        forget_gate = tf.sigmoid(tf.matmul(fx, i) + tf.matmul(fm, o) + fb)
        update = tf.matmul(cx, i) + tf.matmul(cm, o) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(ox, i) + tf.matmul(om, o) + ob)
        return output_gate * tf.tanh(state), state

    y = list()
    output = tf.zeros([num_nodes, 1])
    state = tf.zeros([num_nodes, 1])
    i = 0
    while i < (days-1):
        output, state = lstm_cell(tf.reshape(x[:, i], [5, 1]), output, state)
        y.append(output)
        i += 1
    x = x[:, 1:]

    logits = tf.matmul(w, tf.reshape(tf.reduce_sum(tf.concat(0, y), 0), [5, 1])) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.reshape(tf.reduce_sum(tf.concat(0, x), 1), [5, 1])))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    yT, _ = lstm_cell(tf.reshape(x[:, i], [5, 1]), output, state)
    prediction = tf.matmul(w, yT) + b

num_steps = 2201
summary_frequency = 100
with tf.Session(graph = graph) as sess:
    tf.initialize_all_variables().run()
    print('Initialized')