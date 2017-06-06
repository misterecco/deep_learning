import datetime
import logging
import tensorflow as tf
import numpy as np
import pathlib
import os
import sys
import math
from ops.queues import create_batch_queue, IMAGE_SIZE
from ops.basic import (loss_function, concat,
                       relu, randomly_flip_files, horizontal_flip,
                       vertical_flip, double_flip, average)
from ops.complex import conv, max_pool, convout, bn_conv_relu, bn_upconv_relu


DATASET_PATH = '/data/spacenet2'
TRAINING_SET = 'training_set.txt'
VALIDATION_SET = 'validation_set.txt'
TRAINING_SET_SIZE = 10081
VALIDATION_SET_SIZE = 512

BATCH_SIZE = 8
EPOCHS_N = 12
NN_IMAGE_SIZE = 512
BASE_CHANNELS = 32

date_string = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M")
log_filename = "log/{}.log".format(date_string)
ckpt_filename = "checkpoints/{}.ckpt".format(date_string)

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
logger.warning("Parameters: BATCH_SIZE: {}, NN_IMAGE_SIZE: {}, BASE_CHANNELS: {}".format(
    BATCH_SIZE, NN_IMAGE_SIZE, BASE_CHANNELS))
logger.warning("Augmentation: horizontal and vertical flips")
if len(sys.argv) > 1:
    logger.warning("Starting checkpoint file: {}".format(sys.argv[1]))
logger.warning("Checkpoint file: {}".format(ckpt_filename))


def prepare_file_list(file):
    f = open(file, 'r')
    return [l.strip() for l in f]

def prepare_img_list(file_list):
    img_dir = DATASET_PATH + "/images/"
    return [img_dir + f for f in file_list]

def prepare_ht_list(file_list):
    ht_dir = DATASET_PATH + "/heatmaps/"
    return [ht_dir + f for f in file_list]

def prepare_file_paths(file):
    file_list = prepare_file_list(file)
    return (prepare_img_list(file_list), prepare_ht_list(file_list))


class Trainer():
    def prepare_queues(self):
        train_paths = prepare_file_paths(TRAINING_SET)
        val_paths = prepare_file_paths(VALIDATION_SET)

        self.train_image_batches = create_batch_queue(train_paths,
                                   batch_size=BATCH_SIZE, augment=randomly_flip_files)
        self.val_image_batches = create_batch_queue(val_paths,
                                 batch_size=BATCH_SIZE)


    def create_nn(self, signal, training=True):
        signal = tf.image.resize_images(signal, [NN_IMAGE_SIZE, NN_IMAGE_SIZE])

        with tf.variable_scope("in"):
            signal = conv(signal, BASE_CHANNELS)
            signal = relu(signal)

        with tf.variable_scope("down-1"): # in: 512, out: 256
            skip_1 = signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = max_pool(signal)

        with tf.variable_scope("down-2"): # in: 256, out: 128
            skip_2 = signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = max_pool(signal)

        with tf.variable_scope("down-3"): # in: 128, out: 64
            skip_3 = signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = max_pool(signal)

        with tf.variable_scope("down-4"): # in: 64, out: 32
            skip_4 = signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = max_pool(signal)

        with tf.variable_scope("down-5"): # in: 32, out: 16
            skip_5 = signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = max_pool(signal)

        with tf.variable_scope("down-6"): # in: 16, out: 8
            skip_6 = signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = max_pool(signal)

        with tf.variable_scope("up-0"): # in: 8, out: 16
            signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = bn_upconv_relu(signal, BASE_CHANNELS, training)

        with tf.variable_scope("up-6"): # in: 16, out: 32
            signal = concat(signal, skip_6)
            signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = bn_upconv_relu(signal, BASE_CHANNELS, training)

        with tf.variable_scope("up-5"): # in: 32, out: 64
            signal = concat(signal, skip_5)
            signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = bn_upconv_relu(signal, BASE_CHANNELS, training)

        with tf.variable_scope("up-4"): # in: 64, out: 128
            signal = concat(signal, skip_4)
            signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = bn_upconv_relu(signal, BASE_CHANNELS, training)

        with tf.variable_scope("up-3"): # in: 128, out: 256
            signal = concat(signal, skip_3)
            signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = bn_upconv_relu(signal, BASE_CHANNELS, training)

        with tf.variable_scope("up-2"): # in: 256, out: 512
            signal = concat(signal, skip_2)
            signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = bn_upconv_relu(signal, BASE_CHANNELS, training)

        with tf.variable_scope("up-1"): # in: 512, out: 512
            signal = concat(signal, skip_1)
            signal = bn_conv_relu(signal, BASE_CHANNELS, training)
            signal = convout(signal)

        signal = tf.image.resize_images(signal, [IMAGE_SIZE, IMAGE_SIZE])

        return signal


    def create_model(self):
        with tf.variable_scope('model') as self.vs:
            signal = self.train_image_batches[0]
            ground_truth = self.train_image_batches[1]

            self.u_net = tf.make_template('u_net', self.create_nn, training=True)

            signal = self.u_net(signal)

            self.loss = loss_function(signal, ground_truth)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer().minimize(self.loss)


    def create_validation_model(self):
        with tf.variable_scope(self.vs, reuse=True):
            signal = self.val_image_batches[0]
            ground_truth = self.val_image_batches[1]

            self.u_net_train = tf.make_template('u_net', self.create_nn, training=False)

            s1 = horizontal_flip(signal)
            s2 = vertical_flip(signal)
            s3 = double_flip(signal)

            signal = self.u_net_train(signal)
            s1 = self.u_net_train(s1)
            s2 = self.u_net_train(s2)
            s3 = self.u_net_train(s3)

            s1 = horizontal_flip(s1)
            s2 = vertical_flip(s2)
            s3 = double_flip(s3)

            signal = average([signal, s1, s2, s3])

            self.loss_val = loss_function(signal, ground_truth)


    def train_on_batch(self):
        results = self.sess.run([self.loss, self.train_step])

        return results[0]


    def predict_batch(self):
        results = self.sess.run([self.loss_val])

        return results[0]


    def run_epoch(self, step_func, steps, losses):
        for step_idx in range(steps + 1):
            vloss = step_func()
            losses.append(vloss)

            if step_idx % 10 == 0:
                mean_20 = np.mean(losses[-20:], axis=0)
                mean_200 = np.mean(losses[-200:], axis=0)
                logger.debug('Step {}: mean_loss(20): {} mean_loss(200): {}'.format(step_idx, mean_20, mean_200))


    def run_train_epoch(self):
        steps = math.ceil(TRAINING_SET_SIZE / BATCH_SIZE)
        losses = []
        self.run_epoch(self.train_on_batch, steps, losses)
        logger.info("End of epoch, training set loss (whole epoch avg): {}".format(np.mean(losses, axis=0)))


    def run_validation_epoch(self):
        steps = VALIDATION_SET_SIZE // BATCH_SIZE
        losses = []
        self.run_epoch(self.predict_batch, steps, losses)
        logger.info("Validation set loss: {}".format(np.mean(losses, axis=0)))


    def load_checkpoint(self):
        if len(sys.argv) > 1:
            self.saver.restore(self.sess, sys.argv[1])


    def train(self):
        self.prepare_queues()
        self.create_model()
        self.create_validation_model()

        logger.info("Start training")

        with tf.Session() as self.sess:
            self.saver = tf.train.Saver()

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            self.load_checkpoint()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                epochs = EPOCHS_N
                if '--skip-training' in sys.argv:
                    epochs = 0
                for epoch_idx in range(epochs):
                    logger.info("====== START OF EPOCH {} ======".format(epoch_idx))
                    self.run_train_epoch()
                    self.run_validation_epoch()

            except KeyboardInterrupt:
                logger.info('Stopping training -- keyboard interrupt')

            logger.info("Final predictions")
            self.run_validation_epoch()

            coord.request_stop()
            coord.join(threads)

            if not '--skip-training' in sys.argv:
                self.saver.save(self.sess, ckpt_filename)

            self.sess.close()


if __name__ == '__main__':
    trainer = Trainer()

    trainer.train()
