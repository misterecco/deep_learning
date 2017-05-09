import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
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

        # signal = tf.clip_by_value(self.x, 0.0, 1.0)
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


    def create_viz_model(self):
        self.opt = tf.train.MomentumOptimizer(0.05, momentum=0.9).minimize(self.loss, var_list=[self.x])


    def save_images(self, data):
        for i in range(data.shape[0]):
            img_data = data[i].reshape((28, 28))
            img = Image.fromarray(img_data, 'L')
            img.save('numbers/num_{}.png'.format(i))


    def plot_digits(self, data):
        fig, axes = plt.subplots(2, 5, figsize=(10, 10),
                                 subplot_kw={'xticks': [], 'yticks': []},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i, ax in enumerate(axes.flat):
            ax.imshow(data[i].reshape(28, 28),
                      cmap='gist_yarg', interpolation='nearest',
                      clim=(0, 16))

        plt.show()


    def visualize_numbers(self):
        self.create_model(tf.Variable(tf.truncated_normal(shape=[10, 28*28], stddev=1), dtype=tf.float32, name='x'))
        y = [[1. if i == number else 0. for i in range(0, 10)] for number in range(0, 10)]

        self.create_viz_model()

        with tf.Session() as self.sess:
            self.saver = tf.train.Saver(var_list=tf.trainable_variables()[1:])

            tf.global_variables_initializer().run()

            self.saver.restore(self.sess, 'checkpoints/best.ckpt')

            steps = 10000

            for step in range(steps+1):
                loss, x, acc, _ = self.sess.run([self.loss, self.x, self.accuracy, self.opt], feed_dict={self.y_target: y, self.keep_prob: 1.0})
                if step % 100 == 0:
                    print("Step {}".format(step), loss, acc)
                if step == steps:
                    self.save_images(x)
                    self.plot_digits(x)



if __name__ == '__main__':
    trainer = MnistTrainer()

    # Train and visualize modes are mutally exclusive. Leave one uncommented at a time
    trainer.train()
    # trainer.visualize_numbers()
1
