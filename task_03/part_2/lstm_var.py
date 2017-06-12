import os
import math
import os
import sys
import datetime
import logging
import pathlib

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from ops import (lstm, reshape, loss_function, accuracy, bidirect_lstm,
                 fully_connected, augment, get_last_row)


INPUT_SIZE = 28
DATASET_PATH = "../MNIST_data/"
MB_SIZE = 64

date_string = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M")
if len(sys.argv) > 1:
    date_string = sys.argv[1]
log_filename = "log/{}.log".format(date_string)
ckpt_filename = "checkpoints/{}.ckpt".format(date_string)
summary_train_path = "out/{}/training".format(date_string)
summary_val_path = "out/{}/validation".format(date_string)

pathlib.Path(log_filename).touch()
logger = logging.getLogger('u-net')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.warning("Nodename: {}".format(os.uname()[1]))
if len(sys.argv) > 1:
    logger.warning("Starting checkpoint file: {}".format(sys.argv[1]))
logger.warning("Checkpoint file: {}".format(ckpt_filename))


class MnistTrainer(object):
    def load_checkpoint(self):
        if len(sys.argv) > 1:
            self.saver.restore(self.sess, ckpt_filename)


    def create_model(self):
        layers = [28, 56, 56]

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SIZE * INPUT_SIZE])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        labels = self.y_target
        signal = self.x

        signal = tf.reshape(signal, [-1, INPUT_SIZE, INPUT_SIZE])
        signal = augment(signal, layers[0])

        for i in range(1, len(layers)):
            hidden_n = layers[i]
            input_n = layers[i-1]
            name = "lstm_{}".format(i)
            signal = bidirect_lstm(signal, hidden_n, input_n, name=name)

        signal = get_last_row(signal, layers[-1])
        signal = fully_connected(signal, 10)

        self.global_step = tf.get_variable('global_step', initializer=0)
        update_global_step = tf.assign(self.global_step, self.global_step + 1)

        self.loss = loss_function(signal, labels)
        self.accuracy = accuracy(signal, labels)

        with tf.control_dependencies([update_global_step]):
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        loss_sum = tf.summary.scalar('loss', self.loss)
        acc_sum = tf.summary.scalar('accuracy', self.accuracy)
        self.all_summaries = tf.summary.merge([loss_sum, acc_sum])


    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.loss, self.accuracy, self.global_step,
                                self.all_summaries, self.train_step],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})

        return results[:4]


    def run_training_step(self, losses):
        batch_xs, batch_ys = self.mnist.train.next_batch(MB_SIZE)

        loss, accuracy, global_step, summary = self.train_on_batch(batch_xs, batch_ys)

        losses.append(loss)

        if global_step % 100 == 0:
            logger.debug('Batch {}: mean_loss {}'.format(
                global_step, np.mean(losses[-200:], axis=0)))
            self.train_writer.add_summary(summary, global_step)
            self.train_writer.flush()

        if global_step % 1000 == 0:
            self.run_test()


    def run_test(self):
        loss, accuracy, global_step, summary = self.sess.run([self.loss, self.accuracy,
                                         self.global_step, self.all_summaries],
                                         feed_dict={self.x: self.mnist.test.images,
                                                    self.y_target: self.mnist.test.labels})
        logger.debug('Test results (after batch {}): loss: {}, acc: {}'.format(
                                                  global_step, loss, accuracy))
        self.val_writer.add_summary(summary, global_step)
        self.val_writer.flush()


    def train(self):
        self.create_model()
        self.mnist = input_data.read_data_sets(DATASET_PATH, one_hot=True)

        logger.info("Start training")

        with tf.Session() as self.sess:
            self.saver = tf.train.Saver()
            self.train_writer = tf.summary.FileWriter(summary_train_path, self.sess.graph)
            self.val_writer = tf.summary.FileWriter(summary_val_path, self.sess.graph)

            tf.global_variables_initializer().run()

            self.load_checkpoint()

            batches_n = 50000

            if '--skip-training' in sys.argv:
                batches_n = 0

            losses = []
            try:
                for batch_idx in range(batches_n + 1):
                    self.run_training_step(losses)

            except KeyboardInterrupt:
                logger.info('Stopping training!')
                pass

            self.run_test()

            if not '--skip-training' in sys.argv:
                self.saver.save(self.sess, ckpt_filename)


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()

