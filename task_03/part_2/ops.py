import tensorflow as tf
import numpy as np


def weight_variable(shape, stddev=0.1, name='weight'):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.get_variable(name, initializer=initializer)


def bias_variable(shape, bias=0., name='bias'):
    initializer = tf.constant(bias, shape=shape)
    return tf.get_variable(name, initializer=initializer)


def augment(signal, input_n):
    shape = tf.shape(signal)
    mb_size = tf.to_int32(shape[0])

    dims = tf.random_uniform(shape=(mb_size, 2), minval=24, maxval=28, dtype=tf.int32)

    lengths = tf.squeeze(tf.reduce_prod(dims, axis=1, keep_dims=True), axis=1)
    max_len = tf.reduce_max(lengths)

    padded_len = tf.cast(tf.ceil(max_len / input_n), tf.int32) * input_n
    paddings = padded_len - lengths

    steps_n = tf.ceil(lengths / input_n)

    i = tf.constant(0)
    ns_0 = tf.zeros([0, tf.to_int32(padded_len) // input_n, input_n])

    while_cond = lambda i, _s: tf.less(i, mb_size)
    loop_var = [i, ns_0]

    def body(i, ns):
        idx = tf.to_int32(i)
        img = signal[idx, :, :]
        pad = paddings[idx]

        d = dims[idx, :]
        height = tf.to_int32(d[0])
        width = tf.to_int32(d[1])

        img = tf.slice(img, [0,0], [height, width])
        img = tf.reshape(img, [-1])
        img = tf.pad(img, [[0, pad]])
        img = tf.reshape(img, shape=[-1, input_n])

        next_ns = tf.concat([ns, tf.expand_dims(img, 0)], 0)

        return (tf.add(i, 1), next_ns)

    r = tf.while_loop(while_cond, body, loop_var,
                      shape_invariants=[i.get_shape(), tf.TensorShape([None, None, input_n])])

    new_signal = r[-1]

    return (new_signal, steps_n)


def flatten(signal):
    shape = tf.shape(signal)
    size = shape[1] * shape[2]

    return tf.reshape(signal, shape=[-1, tf.to_int32(size)])


def bidirect_lstm(input, hidden_n, input_n, forget_bias=1.0, name='lstm'):
    hidden_n //= 2
    (signal, all_steps_n) = input

    rev_signal = tf.reverse(signal, axis=[1])
    rev_name = name + '_rev'

    (signal, all_steps_n) = lstm((signal, all_steps_n),
                                  hidden_n, input_n, forget_bias, name)

    (rev_signal, all_steps_n) = lstm((rev_signal, all_steps_n),
                                     hidden_n, input_n, forget_bias, rev_name)

    rev_signal = tf.reverse(rev_signal, axis=[1])

    return (tf.concat([signal, rev_signal], 2), all_steps_n)


def lstm(input, hidden_n, input_n, forget_bias=1.0, name='lstm'):
    (signal, all_steps_n) = input

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
                        tf.TensorShape([None, None, hidden_n]),
                        tf.TensorShape([None, None, hidden_n])])

    return (r[-1], all_steps_n)


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


def get_last_row(input, input_n):
    (signal, steps_n) = input

    shape = tf.shape(signal)
    mb_size = tf.to_int32(shape[0])

    i = tf.constant(0)
    ns_0 = tf.zeros([0, input_n])

    while_cond = lambda i, _s: tf.less(i, mb_size)
    loop_var = [i, ns_0]

    def body(i, ns):
        idx = tf.to_int32(i)
        l = tf.to_int32(steps_n[idx])

        last_row = tf.expand_dims(signal[idx, l-1, :], 0)
        next_ns = tf.concat([ns, last_row], 0)

        return (tf.add(i, 1), next_ns)

    r = tf.while_loop(while_cond, body, loop_var,
                      shape_invariants=[i.get_shape(), tf.TensorShape([None, input_n])])

    return r[-1]


def accuracy(signal, labels):
    bool_tensor = tf.equal(tf.argmax(labels, axis=1), tf.argmax(signal, axis=1))
    return tf.reduce_mean(tf.cast(bool_tensor, tf.float32))