import tensorflow as tf


DATASET_PATH = './spacenet2'
TRAINING_SET = 'training_set.txt'
VALIDATION_SET = 'validation_set.txt'

BATCH_SIZE = 8
IMAGE_SIZE = 650
NUM_CHANELS = 3



def prepare_file_list(file):
    f = open(file, 'r')
    return [l.strip() for l in f]

def prepare_img_list(file_list):
    img_dir = DATASET_PATH + "/images/"
    return [img_dir + f for f in file_list]

def prepare_ht_list(file_list):
    ht_dir = DATASET_PATH + "/images/"
    return [ht_dir + f for f in file_list]

def prepare_file_paths(file):
    file_list = prepare_file_list(file)
    return (prepare_img_list(file_list), prepare_ht_list(file_list))



class Trainer():
    def prepare_queues(self):
        train_paths = prepare_file_paths(TRAINING_SET)
        val_paths = prepare_file_paths(VALIDATION_SET)

        train_paths_tensors = [tf.convert_to_tensor(path, dtype=tf.string) for path in train_paths]
        val_paths_tensors = [tf.convert_to_tensor(path, dtype=tf.string) for path in val_paths]

        train_input_queue = tf.train.slice_input_producer(
            train_paths_tensors,
            shuffle=True
        )
        val_input_queue = tf.train.slice_input_producer(
            val_paths_tensors,
            shuffle=True
        )

        train_file_contents = [tf.read_file(file) for file in train_input_queue]
        val_file_contents = [tf.read_file(file) for file in val_input_queue]

        train_images = [tf.image.decode_jpeg(img, channels=NUM_CHANELS) for img in train_file_contents]
        val_images = [tf.image.decode_jpeg(img, channels=NUM_CHANELS) for img in val_file_contents]

        for image in train_images:
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANELS])
        for image in val_images:
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANELS])

        self.train_image_batches = tf.train.batch(
            train_images,
            batch_size = BATCH_SIZE
        )
        self.val_image_batches = tf.train.batch(
            val_images,
            batch_size = BATCH_SIZE
        )

    def train(self):
        self.prepare_queues()

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            print('From the train set:')
            for _ in range(1):
                print(self.sess.run(self.train_image_batches[0]))

            print('From the val set:')
            for _ in range(1):
                print(self.sess.run(self.val_image_batches[0]))

            coord.request_stop()
            coord.join(threads)

            self.sess.close()


if __name__ == '__main__':
    trainer = Trainer()

    trainer.train()
