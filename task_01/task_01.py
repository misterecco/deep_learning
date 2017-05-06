import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data
import os


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


def reshape(x, new_shape):
    return tf.reshape(x, shape=new_shape)


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
        beta = tf.Variable(1.0, dtype=tf.float32)
        gamma = tf.Variable(0.0, dtype=tf.float32)

        eps = 0.00001
        mean = tf.reduce_mean(signal)
        devs_squared = tf.square(signal - mean)
        var = tf.reduce_mean(devs_squared)
        std = tf.sqrt(var + eps)
        x_norm = (signal - mean) / std

        return x_norm * beta + gamma


class Dropout():
    def __str__(self):
        return "Dropout"

    def __call__(self, signal, keep_prob):
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(signal.get_shape(), dtype=signal.dtype)
        binary_tensor = tf.floor(random_tensor)
        ret = signal / keep_prob * binary_tensor
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


class ConvolutionalLayer():
    def __init__(self):
        pass

    def __str__(self):
        pass

    def __call__(self, signal):
        return signal


class MnistTrainer(object):
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys, self.keep_prob: 0.5})
        return results[1:]


    def create_model(self):
        self.x = InputLayer([None, 784], name='x')()
        self.y_target = InputLayer([None, 10])()
        self.keep_prob = tf.placeholder(tf.float32)

        layersList = [
            FullyConnectedLayer(64),
            BatchNormalization(),
            ReluActivation(),
            FullyConnectedLayer(64),
            BatchNormalization(),
            ReluActivation(),
            FullyConnectedLayer(64),
            BatchNormalization(),
            ReluActivation(),
            FullyConnectedLayer(64),
            BatchNormalization(),
            ReluActivation(),
            FullyConnectedLayer(64),
            BatchNormalization(),
            ReluActivation(),
            FullyConnectedLayer(10),
        ]

        signal = self.x

        print('Signal shape: {}'.format(signal.get_shape()))
        for idx, layer in enumerate(layersList):
            print(layer)
            signal = layer(signal)

        print('Signal shape: {}'.format(signal.get_shape()))

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y_target))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        self.train_step = tf.train.MomentumOptimizer(0.05, momentum=0.9).minimize(self.loss)
        # print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):
 
        self.create_model()
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

 
        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables
            batches_n = 100000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)
 
                    vloss = self.train_on_batch(batch_xs, batch_ys)
 
                    losses.append(vloss)
 
                    if batch_idx % 100 == 0:
                        print('Batch {batch_idx}: mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, mean_loss=np.mean(losses[-200:], axis=0))
                        )
                        print('Test results', self.sess.run([self.loss, self.accuracy],
                                                            feed_dict={self.x: mnist.test.images,
                                                                       self.y_target: mnist.test.labels,
                                                                       self.keep_prob: 1.0}))

 
            except KeyboardInterrupt:
                print('Stopping training!')
                pass
 
            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                                                self.y_target: mnist.test.labels, self.keep_prob: 1.0}))
 
 
if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()


