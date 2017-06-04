import tensorflow as tf
import numpy as np


def weight_variable(shape, stddev):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.get_variable('weight', initializer=initializer)


def bias_variable(shape, bias):
    initializer = tf.constant(bias, shape=shape)
    return tf.get_variable('bias', initializer=initializer)


def relu(signal):
    return tf.nn.relu(signal)


def batch_norm(signal, training):
    shape = signal.get_shape()
    size = shape[-1]
    axis = [0, 1, 2]
    eps = 0.0001
    momentum = 0.99

    beta = tf.get_variable('beta', initializer=tf.zeros([size]), dtype=tf.float32)
    gamma = tf.get_variable('gamma', initializer=tf.ones([size]), dtype=tf.float32)

    moving_mean = tf.get_variable('moving_mean', initializer=tf.zeros(shape[1:]), dtype=tf.float32)
    moving_variance = tf.get_variable('moving_variance', initializer=tf.ones(shape[1:]), dtype=tf.float32)

    if training:
        mean = tf.reduce_mean(signal, axis=axis)
        devs_squared = tf.square(signal - mean)
        variance = tf.reduce_mean(devs_squared)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
            moving_mean.assign(moving_mean * momentum + (1. - momentum) * mean))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
            moving_variance.assign(moving_variance * momentum + (1. - momentum) * variance))

        return tf.nn.batch_normalization(signal, mean, variance, beta, gamma, eps)

    return tf.nn.batch_normalization(signal, moving_mean, moving_variance, beta, gamma, eps)


def conv_2d(signal, size, stride, out_channels):
    shape = signal.get_shape()
    in_channels = int(shape[3])
    ft = weight_variable([size, size, in_channels, out_channels], stddev=0.5)

    return tf.nn.conv2d(signal, filter=ft, strides=[1, stride, stride, 1],
                        padding="SAME", data_format="NHWC")


def deconv_2d(signal, size, stride, out_channels):
    shape = signal.get_shape()
    in_channels = int(shape[3])
    ft = weight_variable([size, size, in_channels, out_channels], stddev=0.5)
    out_shape = tf.stack([shape[0], shape[1]*2, shape[2]*2, out_channels])

    return tf.nn.conv2d_transpose(signal, filter=ft, output_shape=out_shape,
                                  strides=[1, stride, stride, 1],
                                  padding="SAME", data_format="NHWC")


def max_pool_2d(signal, size, stride):
    return tf.nn.max_pool(signal, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],
                            padding='SAME', data_format="NHWC")


def concat(s1, s2):
    return tf.concat([s1, s2], 3)


def pixel_wise_softmax(signal):
    exp_map = tf.exp(signal)
    sum_exp = tf.reduce_sum(exp_map, 3, keep_dims=True)
    return tf.div(exp_map, sum_exp)


def cross_entropy(signal, ground_truth):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=ground_truth)


def pixel_wise_cross_entropy(signal, ground_truth):
    num_class = signal.get_shape().as_list()[-1]
    signal_reshaped = tf.reshape(signal, shape=[-1, num_class])
    gt_reshaped = tf.reshape(ground_truth, shape=[-1, num_class])
    cross_ent = cross_entropy(signal_reshaped, gt_reshaped)
    return tf.reduce_mean(cross_ent)


def loss_function(signal, ground_truth):
    return pixel_wise_cross_entropy(signal, ground_truth)


def cond_horizontal_flip(signal, cond):
    return tf.cond(cond,
                   lambda: tf.reverse(signal, axis=[0]),
                   lambda: tf.identity(signal))


def cond_vertical_flip(signal, cond):
    return tf.cond(cond,
                   lambda: tf.reverse(signal, axis=[1]),
                   lambda: tf.identity(signal))


def random_boolean():
    return tf.less(tf.random_uniform([], 0, 1.0), 0.5)


def randomly_flip_files(files):
    h_flip = random_boolean()
    tmp = [cond_horizontal_flip(file, h_flip) for file in files]

    v_flip = random_boolean()
    return [cond_vertical_flip(file, v_flip) for file in tmp]


def horizontal_flip(signal):
    return tf.reverse(signal, axis=[1])


def vertical_flip(signal):
    return tf.reverse(signal, axis=[2])


def transpose(signal):
    return tf.reverse(signal, axis=[1,2])


def average(signals):
    stack = tf.stack(signals)
    return tf.reduce_mean(stack, axis=0, keep_dims=True)