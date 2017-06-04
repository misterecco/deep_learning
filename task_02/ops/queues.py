import tensorflow as tf

NUM_CHANELS = 3
IMAGE_SIZE = 650


def create_batch_queue(paths, batch_size, augment=None):
    paths_tensors = [tf.convert_to_tensor(path, dtype=tf.string) for path in paths]

    input_queue = tf.train.slice_input_producer(
        paths_tensors,
        shuffle=True
    )

    file_contents = [tf.read_file(file) for file in input_queue]

    images = [tf.image.decode_jpeg(img, channels=NUM_CHANELS) for img in file_contents]

    images[1] = images[1] / 255

    if augment:
        images = augment(images)

    for image in images:
        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANELS])

    return tf.train.batch(
        images,
        batch_size = batch_size
    )
