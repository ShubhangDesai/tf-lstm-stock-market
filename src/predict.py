import tensorflow as tf

num_nodes = 5
days = 30

with tf.Graph().as_default():
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
    x = x[:, 1:]

    loss = MSE/(days-1)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    yT, _ = lstm_cell(tf.reshape(x[:, i], [5, 1]), output, state)

def predict(stock, data):
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, '../models' + stock + '_model.ckpt')
        prediction = sess.run([yT], feed_dict={x: data})
        return prediction