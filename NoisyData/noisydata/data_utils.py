from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np


def create_categorical(num_classes, *args):
    res = []
    for arg in args:
        res.append(keras.utils.to_categorical(arg, num_classes))
    return res


class CIFAR10Data:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def subtract_mean(self):
        mean_image = np.mean(self.x_train, axis=0)
        x_train = self.x_train - mean_image
        x_test = self.x_test - mean_image
        return x_train, x_test

    def get_data(self, subtract_mean=True, output_shape=None):
        x_train, x_test = self.subtract_mean()
        y_train, y_test = create_categorical(len(self.classes), self.y_train, self.y__test)
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.2,
                                                          stratify=y_train)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_noisy_data(self, noise_ratio=0.1, subtract_mean=True, output_shape=None):
        y_train = np.ndarray(self.y_train.shape, dtype=np.uint8)
        y_train[...] = self.y_train[...]
        gt_y_train = np.ndarray(y_train.shape, dtype=np.uint8)
        gt_y_train[...] = y_train[...]
        num_classes = len(self.classes)
        for i in range(num_classes):
            indices = np.where(gt_y_train == i)
            n_noise_data = int(indices[0].shape[0] * noise_ratio)
            noised_indices = np.random.choice(indices[0], n_noise_data, replace=False)
            noised_labels = np.random.choice([k for k in range(num_classes) if k != i], n_noise_data)
            y_train[(noised_indices, np.zeros(noised_indices.shape[0], dtype=np.int))] = noised_labels
            assert sum(y_train[indices] != gt_y_train[indices]) == n_noise_data
        y_train, y_test = create_categorical(len(self.classes), y_train, self.y_test)
        x_train, x_test = self.subtract_mean()

        x_train, x_val, y_train, y_val, gt_y_train, gt_y_val = train_test_split(x_train,
                                                                                y_train,
                                                                                gt_y_train,
                                                                                test_size=0.2,
                                                                                stratify=y_train)

        return x_train, y_train, x_val, y_val, x_test, y_test, gt_y_train, gt_y_val
