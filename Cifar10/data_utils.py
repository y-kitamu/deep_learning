from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import cv2


class CIFAR10Data:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def get_stretch_data(self, subtract_mean=True):
        num_classes = len(self.classes)
        x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype(np.float32)
        y_train = keras.utils.to_categorical(self.y_train, num_classes)
        x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype(np.float32)
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0).astype('uint8')
            x_train -= mean_image
            x_test -= mean_image
        return x_train, y_train, x_test, y_test

    def split_train_val_test(self, x_train, y_train, x_test, y_test):
        num_train = int(x_train.shape[0] * 0.8)
        num_val = x_train.shape[0] - num_train
        mask = list(range(num_train, num_train + num_val))
        x_val = x_train[mask]
        y_val = y_train[mask]

        mask = list(range(num_train))
        x_train = x_train[mask]
        y_train = y_train[mask]
        print(x_train.shape)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_data(self, subtract_mean=True, output_shape=None):
        x_train, x_test = self.subtract_mean()
        y_train, y_test = self.create_categorical()
        return self.split_train_val_test(x_train, y_train, x_test, y_test)

    def create_categorical(self, y_train=None, y_test=None):
        y_train = self.y_train if y_train is None else y_train
        y_test = self.y_test if y_test is None else y_test
        num_classes = len(self.classes)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        return y_train, y_test

    def subtract_mean(self):
        mean_image = np.mean(self.x_train, axis=0)
        x_train = self.x_train - mean_image
        x_test = self.x_test - mean_image
        return x_train, x_test

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
            # Logging().logging("class index = {}, n_data = {}, n_noise_data = {}, n_diff = {}".format(
            #     i, indices[0].shape[0], n_noise_data, sum(y_train[indices] != gt_y_train[indices])))
            assert sum(y_train[indices] != gt_y_train[indices]) == n_noise_data
        y_train, y_test = self.create_categorical(y_train)
        x_train, x_test = self.subtract_mean()
        return self.split_train_val_test(x_train, y_train, x_test, y_test)
