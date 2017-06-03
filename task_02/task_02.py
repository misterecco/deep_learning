import datetime
import logging
import tensorflow as tf
import numpy as np
import pathlib
from ops.queues import create_batch_queue, IMAGE_SIZE
from ops.basic import pixel_wise_softmax, loss_function, concat, relu
from ops.complex import conv, max_pool, convout, bn_conv_relu, bn_upconv_relu


DATASET_PATH = './spacenet2'
TRAINING_SET = 'training_set.txt'
VALIDATION_SET = 'validation_set.txt'
TRAINING_SET_SIZE = 10063
VALIDATION_SET_SIZE = 530

BATCH_SIZE = 8
EPOCHS_N = 20
NN_IMAGE_SIZE = 512
BASE_CHANNELS = 8


log_filename = datetime.datetime.utcnow().strftime("log/%Y-%m-%d_%H:%M.log")
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
    # TODO: data augmentation
    def prepare_queues(self):
        train_paths = prepare_file_paths(TRAINING_SET)
        val_paths = prepare_file_paths(VALIDATION_SET)

        self.train_image_batches = create_batch_queue(train_paths, batch_size=BATCH_SIZE)
        self.val_image_batches = create_batch_queue(val_paths, batch_size=BATCH_SIZE)


    def create_nn(self, signal):
        signal = tf.image.resize_images(signal, [NN_IMAGE_SIZE, NN_IMAGE_SIZE])
        
        with tf.variable_scope("in"):
            signal = conv(signal, BASE_CHANNELS)
            signal = relu(signal)

        with tf.variable_scope("down-1"): # in: 512, out: 256
            skip_1 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.variable_scope("down-2"): # in: 256, out: 128
            skip_2 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.variable_scope("down-3"): # in: 128, out: 64
            skip_3 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.variable_scope("down-4"): # in: 64, out: 32
            skip_4 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.variable_scope("down-5"): # in: 32, out: 16
            skip_5 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.variable_scope("down-6"): # in: 16, out: 8
            skip_6 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)
        
        with tf.variable_scope("up-0"): # in: 8, out: 16
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.variable_scope("up-6"): # in: 16, out: 32
            signal = concat(signal, skip_6)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.variable_scope("up-5"): # in: 32, out: 64
            signal = concat(signal, skip_5)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.variable_scope("up-4"): # in: 64, out: 128
            signal = concat(signal, skip_4)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.variable_scope("up-3"): # in: 128, out: 256
            signal = concat(signal, skip_3)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.variable_scope("up-2"): # in: 256, out: 512
            signal = concat(signal, skip_2)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.variable_scope("up-1"): # in: 512, out: 512
            signal = concat(signal, skip_1)
            signal = bn_conv_relu(signal, BASE_CHANNELS // 2)
            signal = convout(signal)

        signal = tf.image.resize_images(signal, [IMAGE_SIZE, IMAGE_SIZE])        
        
        return signal


    def create_model(self):
        signal = self.train_image_batches[0]
        ground_truth = self.train_image_batches[1]

        self.u_net = tf.make_template('u_net', self.create_nn)

        signal = self.u_net(signal)

        self.loss = loss_function(signal, ground_truth)
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)


    def create_validation_model(self):
        signal = self.val_image_batches[0]
        ground_truth = self.train_image_batches[1]

        signal = self.u_net(signal)

        self.loss_val = loss_function(signal, ground_truth)
        self.out_val = pixel_wise_softmax(signal)


    def train_on_batch(self):
        results = self.sess.run([self.loss, self.train_step])

        return results[0] 


    def predict_batch(self):
        results = self.sess.run([self.loss_val, self.out_val])

        return results[0]


    def run_epoch(self, step_func, steps, losses):
        for step_idx in range(steps + 1):
            vloss = step_func()
            losses.append(vloss)

            if step_idx % 10 == 0:
                mean_20 = np.mean(losses[-20:], axis=0)
                mean_200 = np.mean(losses[-200:], axis=0)
                logger.info('Step {}: mean_loss(20): {} mean_loss(200): {}'.format(step_idx, mean_20, mean_200))
    
    def run_train_epoch(self):
        steps = TRAINING_SET_SIZE // BATCH_SIZE + 1
        losses = []
        self.run_epoch(self.train_on_batch, steps, losses)
        logger.info("End of epoch, training set loss (whole epoch avg: {}".format(np.mean(losses, axis=0)))        


    def run_validation_epoch(self):
        steps = VALIDATION_SET_SIZE // BATCH_SIZE + 1      
        losses = []  
        self.run_epoch(self.predict_batch, steps, losses)
        logger.info("Validation loss: {}".format(np.mean(losses, axis=0)))


    def train(self):
        self.prepare_queues()
        self.create_model()
        self.create_validation_model()

        logger.info("Start training")

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for epoch_idx in range(EPOCHS_N):
                    logger.info("====== START OF EPOCH {} ======".format(epoch_idx))
                    self.run_train_epoch()
                    self.run_validation_epoch()
                    
            except KeyboardInterrupt:
                logger.info('Stopping training -- keyboard interrupt')

            self.run_validation_epoch()
                
            coord.request_stop()
            coord.join(threads)

            self.sess.close()


if __name__ == '__main__':
    trainer = Trainer()

    trainer.train()
