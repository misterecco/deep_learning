import tensorflow as tf


DATASET_PATH = './spacenet2'
TRAINING_SET_FILENAMES = 'training_set.txt'
VALIDATION_SET_FILENAMES = 'validation_set.txt'
BATCH_SIZE = 8
IMAGE_SIZE = 650
NUM_CHANELS = 3



def prepare_file_list(file):
    f = open(file, 'r')
    return [l.strip() for l in f]

def prepare_img_list(filelist):
    img_dir = DATASET_PATH + "/images/"
    return [img_dir + f for f in filelist]

def prepare_ht_list(filelist):
    ht_dir = DATASET_PATH + "/images/"
    return [ht_dir + f for f in filelist]

train_paths = prepare_file_list(TRAINING_SET_FILENAMES)
val_paths = prepare_file_list(VALIDATION_SET_FILENAMES)

train_paths_img = prepare_img_list(train_paths)
train_paths_ht = prepare_ht_list(train_paths)

val_paths_img = prepare_img_list(val_paths)
val_paths_ht = prepare_ht_list(val_paths)


train_paths_img_tensor = tf.convert_to_tensor(train_paths_img, dtype=tf.string)
train_paths_ht_tensor = tf.convert_to_tensor(train_paths_ht, dtype=tf.string)

val_paths_img_tensor = tf.convert_to_tensor(val_paths_img, dtype=tf.string)
val_paths_ht_tensor = tf.convert_to_tensor(val_paths_ht, dtype=tf.string)


train_input_queue = tf.train.slice_input_producer(
    [train_paths_img_tensor, train_paths_ht_tensor],
    shuffle=True
)

val_input_queue = tf.train.slice_input_producer(
    [val_paths_img_tensor, val_paths_ht_tensor],
    shuffle=True
)

# TODO: finish pipeline also for validation set

train_image_content = tf.read_file(train_input_queue[0])
train_heatmap_content = tf.read_file(train_input_queue[1])

train_image = tf.image.decode_jpeg(train_image_content, channels=NUM_CHANELS)
train_heatmap = tf.image.decode_jpeg(train_heatmap_content, channels=NUM_CHANELS)


train_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANELS])
train_heatmap.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANELS])


train_image_batch, train_heatmap_batch = tf.train.batch(
    [train_image, train_heatmap],
    batch_size = BATCH_SIZE
)


val_image_content = tf.read_file(val_input_queue[0])
val_heatmap_content = tf.read_file(val_input_queue[1])

val_image = tf.image.decode_jpeg(val_image_content, channels=NUM_CHANELS)
val_heatmap = tf.image.decode_jpeg(val_heatmap_content, channels=NUM_CHANELS)


val_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANELS])
val_heatmap.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANELS])


val_image_batch, val_heatmap_batch = tf.train.batch(
    [val_image, val_heatmap],
    batch_size = BATCH_SIZE
)



with tf.Session() as sess:

    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    print('From the train set:')
    for _ in range(1):
        print(sess.run(train_image_batch))

    print('From the val set:')
    for _ in range(1):
        print(sess.run(val_image_batch))

    coord.request_stop()
    coord.join(threads)

    sess.close()