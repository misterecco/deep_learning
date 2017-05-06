import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from ops import (InputLayer, ReluActivation, FullyConnectedLayer, BatchNormalization,  Reshape, ConvolutionalLayer2D,
                MaxPool, Dropout, Flatten)


class MnistTrainer(object):
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys, self.keep_prob: 0.5})
        return results[1:]


    def create_model(self, x):
        self.x = x
        self.y_target = InputLayer([None, 10])()
        self.keep_prob = tf.placeholder(tf.float32)

        layersList = [
            Reshape([-1, 28, 28, 1]),
            ConvolutionalLayer2D(size=3, filters=5, stride=1),
            BatchNormalization(),
            ReluActivation(),
            MaxPool(size=2, stride=2),
            ConvolutionalLayer2D(size=3, filters=5, stride=1),
            BatchNormalization(),
            ReluActivation(),
            MaxPool(size=2, stride=2),
            Flatten(),
            BatchNormalization(),
            Dropout(self.keep_prob),
            ReluActivation(),
            FullyConnectedLayer(1024),
            ReluActivation(),
            Dropout(self.keep_prob),
            FullyConnectedLayer(10),
        ]

        signal = self.x

        print('Signal shape: {}'.format(signal.get_shape()))
        for idx, layer in enumerate(layersList):
            print(layer)
            signal = layer(signal)
            print('\tSignal shape: {}'.format(signal.get_shape()))

        self.out = signal

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y_target))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        self.train_step = tf.train.MomentumOptimizer(0.05, momentum=0.9).minimize(self.loss)
        # print('list of variables', list(map(lambda x: x.name, tf.global_variables())))


    def train(self):
 
        self.create_model(InputLayer([None, 784], name='x')())
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

 
        with tf.Session() as self.sess:
            self.saver = tf.train.Saver()

            print(tf.trainable_variables())

            tf.global_variables_initializer().run()  # initialize variables
            batches_n = 20000
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

                    if batch_idx % 1000 == 0:
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

            self.saver.save(self.sess, 'checkpoints/best.ckpt')


    def create_viz_model(self, number):
        perfect_answer = [1. if i == number else 0. for i in range(0, 10)]
        print(perfect_answer)

        self.answers = tf.nn.softmax(self.out)
        zeros = tf.zeros_like(self.x, dtype=tf.float32)
        ones = tf.zeros_like(self.x, dtype=tf.float32)

        self.ls = 100 * tf.reduce_sum(tf.square(self.answers - perfect_answer)) \
                  + tf.reduce_sum(tf.square(tf.minimum(self.x, zeros))) \
                  + tf.reduce_sum(tf.square(ones - tf.maximum(self.x, ones)))

        self.opt = tf.train.MomentumOptimizer(0.05, momentum=0.9).minimize(self.ls, var_list=[self.x])


    def visualize_numbers(self):
        self.create_model(tf.Variable(tf.zeros(shape=[1, 28*28]), dtype=tf.float32, name='x'))

        self.create_viz_model(5)

        # self.create_viz_model(5)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        with tf.Session() as self.sess:
            print(tf.trainable_variables())

            self.saver = tf.train.Saver(var_list=tf.trainable_variables()[1:])

            tf.global_variables_initializer().run()  # initialize variables

            self.saver.restore(self.sess, 'checkpoints/best.ckpt', )

            for step in range(1001):
                loss, x, out, _ = self.sess.run([self.ls, self.x, self.answers, self.opt], feed_dict={self.y_target: mnist.test.labels, self.keep_prob: 1.0})
                if step % 100 == 0:
                    print("Step {}".format(step), loss, out)
                if step == 1000:
                    # print(x)
                    xnp = np.array(x).reshape([28, 28])
                    plt.imshow(xnp, cmap=plt.cm.binary, interpolation='nearest')
                    plt.show()





 
 
if __name__ == '__main__':
    trainer = MnistTrainer()
    # trainer.train()

    trainer.visualize_numbers()

