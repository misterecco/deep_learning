import tensorflow as tf
from .basic import conv_2d, deconv_2d, max_pool_2d, relu, batch_norm


def conv(signal, out_channels, variable_scope="conv"):
    with tf.variable_scope(variable_scope):
        return conv_2d(signal, 3, 1, out_channels)


def upconv(signal, out_channels, variable_scope="upconv"):
    with tf.variable_scope(variable_scope):
        return deconv_2d(signal, 3, 2, out_channels)


def convout(signal, variable_scope="convout"):
    with tf.variable_scope(variable_scope):
        return conv_2d(signal, 1, 1, 3)


def max_pool(signal, variable_scope="max_pool"):
    with tf.variable_scope(variable_scope):
        return max_pool_2d(signal, 2, 2)


def bn_conv_relu(signal, out_channels, training, variable_scope="bn_conv_relu"):
    with tf.variable_scope(variable_scope):
        return relu(conv(batch_norm(signal, training), out_channels))


def bn_upconv_relu(signal, out_channels, training, variable_scope="bn_upconv_relu"):
    with tf.variable_scope(variable_scope):
        return relu(upconv(batch_norm(signal, training), out_channels))
