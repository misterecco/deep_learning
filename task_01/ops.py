import tensorflow as tf
import numpy as np


def weight_variable(shape, stddev):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initializer, name='weight')


def bias_variable(shape, bias):
    initializer = tf.constant(bias, shape=shape)
    return tf.Variable(initializer, name='bias')


def dropout(x, keep_prob):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(x.get_shape(), dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    ret = x / keep_prob * binary_tensor
    ret.set_shape(x.get_shape())

    return ret


class InputLayer():
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.kwargs = kwargs

    def __str__(self):
        return "Input layer: {}".format(self.shape)

    def __call__(self):
        return tf.placeholder(tf.float32, self.shape, *self.kwargs)


class ReluActivation():
    def __str__(self):
        return "Relu activation"

    def __call__(self, signal):
        return tf.nn.relu(signal)


class BatchNormalization():
    def __str__(self):
        return "Batch normalization"

    def __call__(self, signal):
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


class Dropout():
    def __init__(self,  keep_prob):
        self.keep_prob = keep_prob

    def __str__(self,):
        return "Dropout"

    def __call__(self, signal):
        random_tensor = self.keep_prob
        random_tensor += tf.random_uniform(tf.shape(signal), dtype=signal.dtype)
        binary_tensor = tf.floor(random_tensor)
        ret = signal / self.keep_prob * binary_tensor
        ret.set_shape(signal.get_shape())

        return ret


class FullyConnectedLayer():
    def __init__(self, new_num_neurons):
        self.new_num_neurons = new_num_neurons

    def __str__(self):
        return "Fully connected layer: {}".format(self.new_num_neurons)

    def __call__(self, signal):
        cur_num_neurons = int(signal.get_shape()[1])
        stddev = 2 / self.new_num_neurons

        W_fc = weight_variable([cur_num_neurons, self.new_num_neurons], stddev)
        b_fc = bias_variable([self.new_num_neurons], 0.0)

        return tf.matmul(signal, W_fc) + b_fc


class ConvolutionalLayer2D():
    def __init__(self, size, filters, stride):
        self.size = size
        self.stride = stride
        self.filters = filters

    def __str__(self):
        return "Convolutional layer: size: {}x{}, stride: {}, filters: {}".format(
            self.size, self.size, self.stride, self.filters)

    def __call__(self, signal):
        shape = signal.get_shape()
        depth = int(shape[3])
        ft = weight_variable([self.size, self.size, depth, depth * self.filters], stddev=0.5)

        return tf.nn.conv2d(signal, filter=ft, strides=[1, self.stride, self.stride, 1],
                            padding="SAME", data_format="NHWC")


class MaxPool():
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def __str__(self):
        return "Pooling layer: size: {}, stride: {}".format(self.size, self.stride)

    def __call__(self, signal):
        return tf.nn.max_pool(signal, ksize=[1, self.size, self.size, 1], strides=[1, self.stride, self.stride, 1],
                              padding='SAME', data_format="NHWC")


class Reshape():
    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return "Reshape: {}".format(self.shape)

    def  __call__(self, signal):
        return tf.reshape(signal, shape=self.shape)


class Flatten():
    def __str__(self):
        return "Flatten"

    def  __call__(self, signal):
        shape = np.array(signal.get_shape())
        size = np.product(shape[1:])

        return tf.reshape(signal, shape=[-1, int(size)])