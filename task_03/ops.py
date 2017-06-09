import tensorflow as tf
import numpy as np

def weight_variable(shape, stddev=0.1, name='weight'):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.get_variable(name, initializer=initializer)


def bias_variable(shape, bias=0., name='bias'):
    initializer = tf.constant(bias, shape=shape)
    return tf.get_variable(name, initializer=initializer)


def rnn(signal, steps_n, hidden_n, input_n, name='rnn'):
    with tf.variable_scope(name):
        W = weight_variable(shape=(input_n + hidden_n, hidden_n))
        bias = bias_variable(shape=[hidden_n])
        h_0 = bias_variable(shape=(1, hidden_n), name='h_0')

    h = {}
    h[-1] = tf.tile(h_0, [tf.shape(signal)[0], 1])

    for t in range(steps_n):
        signal_t = signal[:, t, :]
        input = tf.concat([signal_t, h[t-1]], axis=1)
        h[t] = tf.tanh(tf.matmul(input, W) + bias)

    return h[steps_n-1]


def lstm(signal, steps_n, hidden_n, input_n, forget_bias=1.0, name='lstm'):
    with tf.variable_scope(name):
        W = weight_variable(shape=(input_n + hidden_n, 4 * hidden_n), name='W')

        bias_value = np.zeros((4 * hidden_n), dtype='float32')
        bias_value[1 * hidden_n: 2 * hidden_n] += forget_bias

        bias = tf.get_variable('b', initializer=bias_value)

        c_0 = bias_variable(shape=(1, hidden_n), name='c_0')
        h_0 = bias_variable(shape=(1, hidden_n), name='h_0')

        c, h = {}, {}
        c[-1] = tf.tile(c_0, [tf.shape(signal)[0], 1])
        h[-1] = tf.tile(h_0, [tf.shape(signal)[0], 1])

    for t in range(steps_n):
        signal_t = signal[:, t, :]
        input = tf.concat([signal_t, h[t-1]], axis=1)

        z = tf.matmul(input, W) + bias
        i = tf.sigmoid(z[:, 0 * hidden_n: 1 * hidden_n])
        f = tf.sigmoid(z[:, 1 * hidden_n: 2 * hidden_n])
        o = tf.sigmoid(z[:, 2 * hidden_n: 3 * hidden_n])
        g =    tf.tanh(z[:, 3 * hidden_n: 4 * hidden_n])

        c[t] = f * c[t-1] + i * g
        h[t] = o * tf.tanh(c[t])

    return h[steps_n-1]



def fully_connected(signal, out_size, name='fc'):
    in_size = signal.get_shape().as_list()[-1]
    stddev = 2 / in_size

    with tf.variable_scope(name):
        W_fc = weight_variable([in_size, out_size], stddev)
        b_fc = bias_variable([out_size], 0.0)

    return tf.matmul(signal, W_fc) + b_fc


def reshape(signal, shape):
    return tf.reshape(signal, shape)


def loss_function(signal, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=labels)
    return tf.reduce_mean(cross_entropy)


# TODO: refactor it to make it look nicer
def accuracy(signal, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(signal, axis=1)), tf.float32))