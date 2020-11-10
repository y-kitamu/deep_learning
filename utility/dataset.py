import os
import glob
import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2


def crop_from_mask_image(image, output_size=(24, 24), max_thresh=150, variance_thresh=100):
    back_type = np.random.choice(["edge", "background"])
    dx, dy = int(output_size[1] / 2), int(output_size[0] / 2)
    mask_thresh = min(int(output_size[0] / 4), int(output_size[1] / 4))
    while True:
        center_x = int(np.random.rand() * image.shape[1])
        center_y = int(np.random.rand() * image.shape[0])
        roi = image[center_y - dy:center_y + dy + 1, center_x - dx:center_x + dx + 1]
        if (roi[0, :, 0] == 0).sum() > mask_thresh or (roi[-1, :, 0]) > mask_thresh or \
           (roi[:, 0, 0] == 0).sum() > mask_thresh or (roi[:, -1, 0] == 0) > mask_thresh:
            continue

        is_edge = roi.max() > max_thresh and roi.var() > variance_thresh
        if back_type == "edge":
            if is_edge:
                return roi
        elif back_type == "background":
            if not is_edge:
                return roi
        else:
            print("Type '{}' is not correct type name".format(back_type))
            sys.exist(-1)


def crop_and_augment(image,
                     label,
                     output_size=(24, 24),
                     padding=(8, 8),
                     max_thresh=150,
                     variance_thresh=100):
    """
    Args:
        image (np.array)                  : input image. 3-D Array [H, W, C] (channel last).
                                            Not 4-D array.
        label (int)                       :
        output_size (tuple of (int, int)) : output (height, width)
        padding (tuple of (int, int))     : padding of (top and bottom, left and right)
        max_thresh (int)                  :
        variance_thresh (int)             :
    """
    assert len(image.shape) == 3
    if int(label) == 0:
        image, label = crop_from_mask_image(image, output_size, max_thresh, variance_thresh)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
    else:
        image = tfa.image.rotate(image, tf.constant(2 * np.pi * np.random.rand()))
        image = tf.image.resize_with_crop_or_pad(image, output_size[0] + padding[0] * 2,
                                                 output_size[1] + padding[1] * 2)
        image = tf.image.random_crop(image, (output_size[0], output_size[1], image.shape[-1]))
    assert image.shape[:2] == output_size
    return image, label


def create_dataset(data_dir, n_classes, weights=[1.0, 1.0, 1.0, 1.0, 1.0], batch_size=120):
    datasets = []
    num_data = 0
    for i in range(n_classes):
        class_data_dir = os.path.join(n_classes, "{}".format(i))
        image_list = []
        file_list = glob.glob(class_data_dir, "*.png")
        for fname in file_list:
            image_list.append(cv2.imread(fname))
        images = np.stack(image_list, axis=0)
        labels = np.ones(len(file_list)) * i
        num_data += len(file_list)
        assert images.shape[0] == len(file_list)

        datasets.append(tf.data.Dataset.from_tensor_slices((images, labels)))

    train_ds = tf.data.experimental.sample_from_datasets(datasets, weights)
    train_ds = train_ds.map(crop_and_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.repeat().shuffle(num_data).prefetch(buffer_size=num_data).batch(batch_size)
    return train_ds
