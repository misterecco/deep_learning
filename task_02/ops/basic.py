import tensorflow as tf
import numpy as np


def weight_variable(shape, stddev):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initializer, name='weight')


def bias_variable(shape, bias):
    initializer = tf.constant(bias, shape=shape)
    return tf.Variable(initializer, name='bias')


def relu(signal):
    return tf.nn.relu(signal)


def batch_norm(signal):
    shape = signal.get_shape()
    size = shape[3] if len(shape) >= 4 else 1
    axis = [0, 1, 2] if len(shape) >= 4 else [0, 1]

    beta = tf.Variable(tf.ones([size]), dtype=tf.float32)
    gamma = tf.Variable(tf.zeros([size]), dtype=tf.float32)

    eps = 0.00001
    mean = tf.reduce_mean(signal, axis=axis)
    devs_squared = tf.square(signal - mean)
    var = tf.reduce_mean(devs_squared)
    std = tf.sqrt(var + eps)
    x_norm = (signal - mean) / std

    return x_norm * beta + gamma


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

    return tf.nn.conv2d_transpose(signal, filter=ft, strides=[1, stride, stride, 1],
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


def loss_function(signal, ground_truth):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=ground_truth))