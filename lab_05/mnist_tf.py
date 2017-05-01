import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

LOG_DIR_TRAIN = 'out/mnist_tf/train'
LOG_DIR_TEST = 'out/mnist_tf/test'
SESSION_PATH = 'tmp/model.ckpt'

''''
Tasks:
1. Train a simple linear model(no hidden layers) on the mnist dataset.
   Use softmax layer and cross entropy loss i.e.
   if P is the vector of predictions, and y is one-hot encoded correct label
   the loss = \sum{i=0..n} y_i * -log(P_i)
   Train it using some variant of stochastic gradient descent.
   What performance(loss, accuracy) do you get?

2. Then change this solution to implement a Multi Layer Perceptron.
   Choose the activation function and initialization of weights.
   Make sure it is easy to change number of hidden layers, and sizes of each layer.
   What performance(loss, accuracy) do you get?

3. If you used built in tensorflow optimizer like tf.train.GradientDescentOptimizer, try
   implementing it on your own. (hint: tf.gradients method)

4. Add summaries and make sure you can see the the progress of learning in TensorBoard.
   (You can read more here https://www.tensorflow.org/get_started/summaries_and_tensorboard,
   example there is a little involved, you can look at summary_example.py for a shorter one)

5. Add periodic evaluation on the validation set (mnist.validation).
   Can you make the model overfit?

   Make sure the statistics from training and validation end up in TensorBoard
   and you can see them both on one plot.

6. Enable saving the weights of the trained model.
   Make it also possible to read the trained model.
   (You can read about saving and restoring variable in https://www.tensorflow.org/programmers_guide/variables)

Extra task:
* Show the images from the test set that the model gets wrong. Do it using TensorBoard.
* Try running your model on CIFAR-10 dataset. See what results you can get. In the future we will try
to solve this dataset with Convolutional Neural Network.
'''
class MnistTrainer():
    @staticmethod
    def softmax(y):
        y_exp = tf.exp(y)
        return y_exp / tf.expand_dims(tf.reduce_sum(y_exp, -1), -1)


    @staticmethod
    def activation(v):
        return tf.nn.relu(v)


    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    @staticmethod
    def layer(x, in_size, out_size):
        W = MnistTrainer.weight_variable([in_size, out_size])
        b = MnistTrainer.bias_variable([out_size])

        return MnistTrainer.activation(tf.matmul(x, W) + b)


    def update_variables(self):
        learning_rate = 0.01
        var_list = tf.all_variables()
        grads = tf.gradients(self.loss, var_list)

        var_updates = []

        for grad, var in zip(grads, var_list):
            var_updates.append(var.assign_sub(learning_rate * grad))

        return tf.group(*var_updates)


    def train_on_batch(self, batch_xs, batch_ys):
        feed_dict = {
            self.x: batch_xs,
            self.y_target: batch_ys
        }

        loss, acc, summ, _ = self.sess.run([self.loss, self.accuracy, self.all_summaries,  self.train_step], feed_dict=feed_dict)

        return loss, acc, summ


    def create_model(self, hidden_layers=()):
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        self.y_target = tf.placeholder(tf.float32, shape=[None, 10])

        layers_shapes = (784, *hidden_layers, 10)
        layers = [self.x]

        for i in range(1, len(layers_shapes)):
            layers.append(self.layer(layers[i-1], layers_shapes[i-1], layers_shapes[i]))

        y = layers[-1]

        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_target, logits=y))
        m_log = (-1.0) * tf.log(self.softmax(y))
        self.loss = tf.reduce_mean(tf.multiply(m_log, self.y_target))

        # self.train_step = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)
        self.train_step = self.update_variables()

        correct_prediction = tf.equal(tf.arg_max(self.y_target, 1), tf.arg_max(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.all_summaries = tf.summary.merge_all()


    def restore_session(self):
        # XXX: check if file exists
        print('Reading checkpoint: {}'.format(SESSION_PATH))
        self.saver.restore(self.sess, SESSION_PATH)


    def train(self):
        self.create_model(hidden_layers=(1000, 200))
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



        with tf.Session() as self.sess:
            self.saver = tf.train.Saver()

            train_writer = tf.summary.FileWriter(LOG_DIR_TRAIN, self.sess.graph)
            test_writer = tf.summary.FileWriter(LOG_DIR_TEST, self.sess.graph)

            # XXX: this should be tf variable (so it will get saved in checkpoint)
            step = 0

            tf.global_variables_initializer().run()  # initialize variables

            self.restore_session()

            batches_n = 10000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)

                    loss, accuracy, summary = self.train_on_batch(batch_xs, batch_ys)

                    # print(loss, accuracy)


                    losses.append(loss)

                    if batch_idx % 100 == 0:
                        step += 1
                        train_writer.add_summary(summary, step)
                        train_writer.flush()

                        print('Batch {batch_idx}: loss {loss}, mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, loss=loss, mean_loss=np.mean(losses[-200:]))
                        )

                        test_summary = self.sess.run(self.all_summaries, feed_dict={self.x: mnist.test.images, self.y_target: mnist.test.labels})
                        test_writer.add_summary(test_summary, step)
                        test_writer.flush()



            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                                                self.y_target: mnist.test.labels}))

            self.saver.save(self.sess, SESSION_PATH)


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()

