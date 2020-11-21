import tensorflow as tf


def augmentation(image, label):
    height = image.shape[0]
    width = image.shape[1]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, width + 8, height + 8)
    image = tf.image.random_crop(image, [height, width, image.shape[2]])
    return image, label
