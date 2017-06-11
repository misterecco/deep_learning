import tensorflow as tf
import numpy as np


def weight_variable(shape, stddev=0.1, name='weight'):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.get_variable(name, initializer=initializer)


def bias_variable(shape, bias=0., name='bias'):
    initializer = tf.constant(bias, shape=shape)
    return tf.get_variable(name, initializer=initializer)


def random_crop(signal):
    height = tf.random_uniform(shape=(), minval=24, maxval=28, dtype=tf.int32)
    width = tf.random_uniform(shape=(), minval=24, maxval=28, dtype=tf.int32)

    cropped = tf.slice(signal, [0, 0, 0], [-1, tf.to_int32(height), tf.to_int32(width)])

    return cropped


def flatten(signal):
    shape = tf.shape(signal)
    size = shape[1] * shape[2]

    return tf.reshape(signal, shape=[-1, tf.to_int32(size)])


def pad(signal):
    flat = flatten(signal)

    length = tf.shape(flat)[-1]
    padded_length = tf.cast(tf.ceil(length / 28), tf.int32) * 28
    padding = padded_length - length

    height = padded_length // 28
    width = 28

    padded = tf.pad(flat, [[0,0], [0, padding]])

    return tf.reshape(padded, shape=[-1, tf.to_int32(height), width])


def augment(signal):
    return pad(random_crop(signal))


def lstm(signal, hidden_n, input_n, forget_bias=1.0, name='lstm'):
    shape = tf.shape(signal)
    steps_n = tf.to_int32(shape[-2])
    length = input_n * steps_n

    with tf.variable_scope(name):
        W = weight_variable(shape=(input_n + hidden_n, 4 * hidden_n), name='W')

        bias_value = np.zeros((4 * hidden_n), dtype='float32')
        bias_value[1 * hidden_n: 2 * hidden_n] += forget_bias

        bias = tf.get_variable('b', initializer=bias_value)

        c_0 = bias_variable(shape=(1, hidden_n), name='c_0')
        h_0 = bias_variable(shape=(1, hidden_n), name='h_0')

        test = bias_variable(shape=(1, hidden_n, 0), name='test')

    prev_c = tf.tile(c_0, [tf.shape(signal)[0], 1])
    prev_h = tf.tile(h_0, [tf.shape(signal)[0], 1])

    tmp = tf.expand_dims(prev_c, 1)

    c = tf.tile(tmp, [1, 0, 1])
    h = tf.tile(tmp, [1, 0, 1])

    i = tf.constant(0)
    loop_var = [i, prev_c, prev_h, c, h]

    while_cond = lambda t, _pc, _ph, _c, _h: tf.less(t, steps_n)

    def body(t, pc, ph, c, h):
        signal_t = signal[:, tf.to_int32(t), :]
        input = tf.concat([signal_t, ph], axis=1)

        z = tf.matmul(input, W) + bias
        i = tf.sigmoid(z[:, 0 * hidden_n: 1 * hidden_n])
        f = tf.sigmoid(z[:, 1 * hidden_n: 2 * hidden_n])
        o = tf.sigmoid(z[:, 2 * hidden_n: 3 * hidden_n])
        g =    tf.tanh(z[:, 3 * hidden_n: 4 * hidden_n])

        next_c = f * pc + i * g
        next_h = o * tf.tanh(next_c)

        nc = tf.concat([c, tf.expand_dims(next_c, 1)], 1)
        nh = tf.concat([h, tf.expand_dims(next_h, 1)], 1)

        return (tf.add(t, 1), next_c, next_h, nc, nh)

    r = tf.while_loop(while_cond, body, loop_var,
                       shape_invariants=[i.get_shape(),
                       prev_c.get_shape(), prev_h.get_shape(),
                       tf.TensorShape([None, None, 28]), tf.TensorShape([None, None, 28])])

    return r[-1]


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


def get_last_row(signal):
    shape = tf.shape(signal)
    signal = tf.slice(signal, [0, tf.to_int32(shape[1]-1), 0], [-1, 1, -1])
    return tf.squeeze(signal, axis=[1])


def accuracy(signal, labels):
    bool_tensor = tf.equal(tf.argmax(labels, axis=1), tf.argmax(signal, axis=1))
    return tf.reduce_mean(tf.cast(bool_tensor, tf.float32))