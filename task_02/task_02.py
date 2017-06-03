import tensorflow as tf
import numpy as np
import math
from ops.queues import create_batch_queue, IMAGE_SIZE
from ops.basic import pixel_wise_softmax, loss_function, concat, relu
from ops.complex import conv, max_pool, convout, bn_conv_relu, bn_upconv_relu


DATASET_PATH = './spacenet2'
TRAINING_SET = 'training_set.txt'
VALIDATION_SET = 'validation_set.txt'

BATCH_SIZE = 4
BATCHES_N = 100000
NN_IMAGE_SIZE = 512
BASE_CHANNELS = 16


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


    def create_model(self):
        signal = self.train_image_batches[0]
        ground_truth = self.train_image_batches[1]

        signal = tf.image.resize_images(signal, [NN_IMAGE_SIZE, NN_IMAGE_SIZE])

        with tf.name_scope("in"):
            signal = conv(signal, BASE_CHANNELS)
            signal = relu(signal)

        with tf.name_scope("down-1"): # in: 512, out: 256
            skip_1 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.name_scope("down-2"): # in: 256, out: 128
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            skip_2 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.name_scope("down-3"): # in: 128, out: 64
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            skip_3 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.name_scope("down-4"): # in: 64, out: 32
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            skip_4 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.name_scope("down-5"): # in: 32, out: 16
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            skip_5 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)

        with tf.name_scope("down-6"): # in: 16, out: 8
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            skip_6 = signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = max_pool(signal)
        
        with tf.name_scope("up-6"): # in: 8, out: 16
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.name_scope("up-5"): # in: 16, out: 32
            signal = concat(signal, skip_6)
            signal = bn_conv_relu(signal, BASE_CHANNELS * 3 // 2)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.name_scope("up-4"): # in: 32, out: 64
            signal = concat(signal, skip_5)
            signal = bn_conv_relu(signal, BASE_CHANNELS * 3 // 2)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.name_scope("up-3"): # in: 64, out: 128
            signal = concat(signal, skip_4)
            signal = bn_conv_relu(signal, BASE_CHANNELS * 3 // 2)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.name_scope("up-2"): # in: 128, out: 256
            signal = concat(signal, skip_3)
            signal = bn_conv_relu(signal, BASE_CHANNELS * 3 // 2)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.name_scope("up-1"): # in: 256, out: 512
            signal = concat(signal, skip_2)
            signal = bn_conv_relu(signal, BASE_CHANNELS * 3 // 2)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = bn_upconv_relu(signal, BASE_CHANNELS)

        with tf.name_scope("out"): # in: 512, out: 512
            signal = concat(signal, skip_1)
            signal = bn_conv_relu(signal, BASE_CHANNELS * 3 // 2)
            signal = bn_conv_relu(signal, BASE_CHANNELS)
            signal = convout(signal)

        # TODO: maybe there is a way to merge output and loss
        # NOTE: out doesn't have to be calculated for training set
        signal = tf.image.resize_images(signal, [IMAGE_SIZE, IMAGE_SIZE])
        self.out = pixel_wise_softmax(signal)
        self.loss = loss_function(signal, ground_truth)
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)


    def train_on_batch(self):
        results =  self.sess.run([self.loss, self.train_step])
        # print('OUT:')
        # print(results[2])
        
        return results[0] 


    def train(self):
        self.prepare_queues()
        self.create_model()

        losses = []

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for batch_idx in range(BATCHES_N + 1):

                    vloss = self.train_on_batch()
                    losses.append(vloss)

                    val = float(vloss)

                    if batch_idx % 10 == 0:
                        mean_20 = np.mean(losses[-20:], axis=0)
                        mean_200 = np.mean(losses[-200:], axis=0)
                        print('Batch {}: mean_loss(20): {} mean_loss(200): {}'.format(batch_idx, mean_20, mean_200))
                    
                    if math.isnan(val):
                        break                    
            
            except KeyboardInterrupt:
                print('Stopping training -- keyboard interrupt')
                
            coord.request_stop()
            coord.join(threads)

            self.sess.close()


if __name__ == '__main__':
    trainer = Trainer()

    trainer.train()
