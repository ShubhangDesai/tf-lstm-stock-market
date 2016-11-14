from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
import cStringIO

# stock = 'AAPL'
stock = 'GOOGL'
# stock = 'MSFT'

num_nodes = 5
days = 30
valid_unrolls = 2
valid_amount = 100

def generate_data(csv_file):
    with open(csv_file+'.csv', "r") as myfile:
        data = myfile.read().replace('"', '')
    data = np.genfromtxt(cStringIO.StringIO(data), delimiter=',')[1:, 1:]
    data = data.astype(float)
    data = data[1:] / data[:-1] - 1
    data = np.fliplr(data.transpose())
    train_data = data[:, :2030]
    valid_data = data[:, 2030:2260]
    return train_data, valid_data

train_data, valid_data = generate_data('../res/'+stock)

class BatchGenerator(object):
  cursor = 0
  def __init__(self, data, batch_size, num_unrollings):
    self._data = data
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings

  def next(self):
    batch = np.zeros([self._batch_size, self._num_unrollings])
    for i in range(self._num_unrollings):
        batch[:, i] = self._data[:, i+self.cursor]
    self.cursor += 1
    return batch

train_batches = BatchGenerator(train_data, num_nodes, days)
valid_batches = BatchGenerator(valid_data, num_nodes, valid_unrolls)

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
    MSE = 0
    while i < (days-1):
        output, state = lstm_cell(tf.reshape(x[:, i], [5, 1]), output, state)
        y.append(output)
        MSE += tf.reduce_mean(tf.square(output - tf.reshape(x[:, i+1], [5, 1])))
        i += 1

    loss = MSE/(days-1)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    yT, _ = lstm_cell(tf.reshape(x[:, i], [5, 1]), output, state)

num_steps = 2000
summary_frequency = 100
with tf.Session(graph = graph) as sess:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batch = train_batches.next()
        batch = batch.astype(np.float32)
        _, l = sess.run([optimizer, loss], feed_dict={x: batch})
        mean_loss += l
        if (step+1) % summary_frequency == 0:
            mean_loss = mean_loss/summary_frequency
            print('Average MSE at step %s: %s' % (step, mean_loss))
            mean_loss = 0
            valid_batch = valid_batches.next()
            valid_batch = batch.astype(np.float32)

            v_l, yT = sess.run([loss, yT], feed_dict={x: valid_batch})
            yT = yT[:, 0]
            print('Validation predictions: %s' % yT)

            # v_l = sess.run(loss, feed_dict={x: valid_batch})
            print('Validation MSE: %f' % v_l)


    save_path = tf.train.Saver().save(sess, '../models/' + stock + '_model.ckpt')