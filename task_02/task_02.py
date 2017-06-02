import tensorflow as tf
import numpy as np
from ops.queues import create_batch_queue, IMAGE_SIZE
from ops.basic import pixel_wise_softmax, loss_function, concat, relu
from ops.complex import conv, max_pool, convout, bn_conv_relu, bn_upconv_relu


DATASET_PATH = './spacenet2'
TRAINING_SET = 'training_set.txt'
VALIDATION_SET = 'validation_set.txt'

BATCH_SIZE = 16
BATCHES_N = 100
NN_IMAGE_SIZE = 256


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

        signal = pixel_wise_softmax(signal)

        signal = tf.image.resize_images(signal, [IMAGE_SIZE, IMAGE_SIZE])

        self.out = signal


        self.loss = loss_function(signal, ground_truth)
        # self.train_step = tf.train.MomentumOptimizer(0.05, momentum=0.9).minimize(self.loss)


    def train_on_batch(self):
        results =  self.sess.run([self.loss, self.train_step])
        results =  self.sess.run([self.loss, self.out])
        print(results[1])
        
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
                for batch_idx in range(BATCHES_N):

                    vloss = self.train_on_batch()
                    losses.append(vloss)

                    print('Batch {}: mean_loss {}'.format(
                        batch_idx, np.mean(losses[-200:], axis=0)))
            
            except KeyboardInterrupt:
                print('Stopping training -- keyboard interrupt')
                
            coord.request_stop()
            coord.join(threads)

            self.sess.close()


if __name__ == '__main__':
    trainer = Trainer()

    trainer.train()
