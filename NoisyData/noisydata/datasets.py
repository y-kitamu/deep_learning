import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from .utility import Logging


def create_cifar10_data():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, y_train, x_val, y_val, y_val, x_test, y_test


def create_noised_cifar10_data(noise_ratio=0.1):
    """
    """
    n_classes = 10
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    order = np.random.permutation(np.arange(x_train.shape[0]))
    x_train = x_train[order]
    y_train = y_train[order]
    gt_y_train = np.ndarray(y_train.shape, dtype=np.uint8)
    gt_y_train[...] = y_train[...]

    # import pdb; pdb.set_trace()
    for i in range(n_classes):
        indices = np.where(gt_y_train == i)
        n_noise_data = int(indices[0].shape[0] * noise_ratio)
        noised_indices = np.random.choice(indices[0], n_noise_data, replace=False)
        noised_labels = np.random.choice([k for k in range(n_classes) if k != i], n_noise_data)
        y_train[(noised_indices, np.zeros(noised_indices.shape[0], dtype=np.int))] = noised_labels
        # Logging().logging("class index = {}, n_data = {}, n_noise_data = {}, n_diff = {}".format(
        #     i, indices[0].shape[0], n_noise_data, sum(y_train[indices] != gt_y_train[indices])))
        assert sum(y_train[indices] != gt_y_train[indices]) == n_noise_data

    x_train, x_val, y_train, y_val, gt_y_train, gt_y_val = train_test_split(
        x_train, y_train, gt_y_train, test_size=0.1)
    return x_train, y_train, gt_y_train, x_val, y_val, gt_y_val, x_test, y_test


def load_mnist_dataset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(y_trains.shape[0]).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    Logging().logging("load mnist")
    return train_ds, test_ds
