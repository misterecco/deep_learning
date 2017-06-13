import datetime

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from ops import rnn, lstm, reshape, loss_function, accuracy, fully_connected


class MnistTrainer(object):
    def create_model(self):
        input_n = 28
        steps_n = 784 // input_n

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        labels = self.y_target

        signal = self.x
        signal = tf.reshape(signal, [-1, steps_n, input_n])
        signal = lstm(signal, steps_n, input_n, input_n)

        signal = fully_connected(signal, 10)

        self.loss = loss_function(signal, labels)
        self.accuracy = accuracy(signal, labels)
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)


    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.loss, self.accuracy, self.train_step],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})

        return results[:2]


    def print_start(self):
        date_string = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M")
        print(date_string + " start training")


    def run_test(self):
        date_string = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M")
        print(date_string + ' test results', self.sess.run([self.loss, self.accuracy],
                                            feed_dict={self.x: self.mnist.test.images,
                                                       self.y_target: self.mnist.test.labels}))


    def train(self):
        self.create_model()
        self.mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
        self.print_start()

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()
            batches_n = 25000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n + 1):
                    batch_xs, batch_ys = self.mnist.train.next_batch(mb_size)

                    loss, accuracy = self.train_on_batch(batch_xs, batch_ys)


                    losses.append(loss)

                    if batch_idx % 100 == 0:
                        print('Batch {}: mean_loss {}'.format(
                            batch_idx, np.mean(losses[-200:], axis=0)))

                    if batch_idx % 1000 == 0:
                        self.run_test()


            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            self.run_test()


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()

