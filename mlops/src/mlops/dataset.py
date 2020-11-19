import os
import sys
import glob
import queue
import random
import multiprocessing as mp

import numpy as np
import tensorflow as tf
import cv2
from omegaconf import DictConfig

from mlops.constant import AugmentParameters, FlawParameters, FlawlessParameters, DatasetParameters
from mlops.utility import imread_16U, _get_index_array_filename
from mlops.preprocess import load_filelist_csv


def _create_flawless_images(num_crop, image_filename, flawless_params, augment_params, num_channel,
                            output_image_size):
    """Crop and augment 'num_crop' images from original image (image of 'filename')
    Args:
        num_crop (int) : number of images that cropped from the original(input) image.
        filename (str) : Input image filename
        flawless_params (FlawlessParameters) :
        augment_params (AugmentParameters) :
        output_image_size (int):
        num_channel (int) :
    Return (np.array, np.array)
        cropped images array and labels array.
        shape of cropped images array is [num_crop, output_image_size, output_image_size,
        num_channel] and shape of labels array is [num_crop].
    """
    num_background = int(num_crop * flawless_params.ratios[0] / sum(flawless_params.ratios))
    num_edge = num_crop - num_background

    images_array = np.zeros((num_crop, output_image_size, output_image_size, num_channel),
                            dtype=np.float32)
    labels_array = np.ones(num_crop)
    labels_array[:num_background] *= flawless_params.classes[0]
    labels_array[num_background:] *= flawless_params.classes[1]

    image = imread_16U(image_filename)
    index_array = np.load(_get_index_array_filename(image_filename, flawless_params))

    offset_x = output_image_size
    offset_y = output_image_size

    background_samples = np.random.choice(index_array["background"], size=num_background, replace=False)
    edge_samples = np.random.choice(index_array["edge"], size=num_edge, replace=False)
    for idx, (x, y) in enumerate(background_samples):
        roi = image[y - offset_y:y + offset_y, x - offset_x:x + offset_x]
        images_array[idx, ...] = _image_augmentation(roi, augment_params, output_image_size)
    for idx, (x, y) in enumerate(edge_samples):
        roi = image[y - offset_y:y + offset_y, x - offset_x:x + offset_x]
        images_array[num_background + idx, ...] = _image_augmentation(roi, augment_params,
                                                                      output_image_size)
    return images_array, labels_array


def _create_flaw_image(filename, aug_params, num_channel, output_image_size):
    """
    Args:
        filename (str):
        aug_params (AugmentParameters):
        num_channel (int) :
        ouptut_image_size (int) :
    """
    image = imread_16U(filename)
    return _image_augmentation(image, aug_params=aug_params, output_size=output_image_size)


def _image_augmentation(img, aug_params, output_size=24):
    """Random rotate, random shift-crop and random scaling.
    Args:
        img (np.ndarray) :
        aug_params (AugmentParameters) :
        output_size (int) :
    """
    assert len(img.shape) == 3
    angle = int(random.uniform(0, 2 * np.pi))
    scale = int(random.uniform(aug_params.min_scale, aug_params.max_scale))
    shift = np.random.randint(-aug_params.shift_range, aug_params.shift_range + 1, 2)
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), angle, scale)
    matrix[:, 2] += shift
    img = cv2.warpAffine(img, matrix, (width, height))
    if len(img.shape) == 2:
        img = img[..., None]
    cy, cx = int(height / 2), int(width / 2)
    delta = int(output_size / 2)
    return img[cy - delta:cy + delta, cx - delta:cx + delta]


class TrainDataGenerator:
    """
    Args:
        config (DictConfig):
        is_train (bool):
    """

    def __init__(self, config: DictConfig, is_train=True):
        self.flawless_params = FlawlessParameters(**config["flawless"])
        self.flaws_params = FlawParameters(**config["flaws"])
        self.aug_params = AugmentParameters(**config["augment"])
        self.params = DatasetParameters(**config["overall"],
                                        flawless_params=self.flawless_params,
                                        flaw_params=self.flaws_params,
                                        augment_params=self.aug_params)
        self.num_flaws_per_batch = int(self.params.batch_size /
                                       (1 + sum(self.params.flawless_params.ratios)))
        self.num_flawless_per_batch = self.params.batch_size - self.num_flaws_per_batch
        self.class_file_dict = load_filelist_csv(csv_filename=self.params["overall"]["csv_filename"])
        self.pool = mp.Pool(self.params.num_process)

    def create_dataset(self):
        train_ds = tf.data.Dataset.from_generator(
            self,
            output_types=(tf.float32, tf.float32),
            output_shapes=((self.params.batch_size, self.params.output_image_size,
                            self.params.output_image_size, self.params.num_channel),
                           (self.params.batch_size)))
        return train_ds.prefetch(1)

    def __del__(self):
        self.pool.terminate()
        self.pool = None

    def __call__(self):
        images_array = np.zeros((self.params.batch_size, self.params.output_image_size,
                                 self.params.output_image_size, self.params.num_channel))
        labels_array = np.zeros(self.params.batch_size)
        num_crops_per_image = int(self.num_flawless_per_batch /
                                  self.params.flawless_params.num_files_per_batch)
        weights = np.array(self.params.flaw_params.ratios)
        weights /= sum(weights)
        weights[-1] = 1 - sum(weights[:-1])

        while True:
            flawless = self._get_next_batch(images_array, labels_array, num_crops_per_image)
            flaws = self._get_next_batch(images_array, labels_array, weights)
            images_array[:self.num_flawless_per_batch, ...] = flawless[0]
            labels_array[:self.num_flawless_per_batch] = flawless[1]
            images_array[self.num_flawless_per_batch:, ...] = flaws[0]
            labels_array[self.num_flawless_per_batch:] = flaws[1]
            yield images_array, labels_array

    def _get_next_flawless_batch(self, images_array, labels_array, num_crops_per_image):
        queues = []
        for idx, filename in enumerate(self._get_next_files(0, self.num)):
            num_crop = min(num_crops_per_image, self.num_flaws_per_batch - idx * num_crops_per_image)
            queues.append(
                self.pool.apply_async(_create_flawless_images, (
                    num_crop,
                    filename,
                    self.params.flawless_params,
                    self.params.augment_params,
                    self.params.num_channel,
                    self.params.output_image_size,
                )))
            # images, labels = _augment_and_crop_flawless_images(num_crop, filename,
            #                                                    self.params.flawless_params,
            #                                                    self.params.augment_params,
            #                                                    self.params.num_channel,
            #                                                    self.params.output_image_size)
            # images_array[remaining - num_crop:remaining] = images
            # labels_array[remaining - num_crop:remaining] = labels

        start_idx = 0
        for que in queues:
            images, labels = que.get()
            num_element = labels.shape[0]
            images_array[start_idx:start_idx + num_element] = images
            labels_array[start_idx:start_idx + num_element] = labels
            start_idx += num_element
        return images_array, labels_array

    def _get_next_flaw_batch(self, images_array, labels_array, weights):
        ques = []
        labels_array[...] = np.random.choice(self.params.flaw_params.classes,
                                             size=self.num_flaws_per_batch,
                                             p=weights)
        for cls_idx in self.params.flaw_params.classes:
            cnt = (labels_array.astype(int) == cls_idx).sum()
            if cnt == 0:
                continue
            for fname in self.get_next_files(cls_idx, cnt):
                ques.append(
                    self.pool.apply_async(_create_flaw_image,
                                          (fname, self.params.augment_params, self.params.num_channel,
                                           self.params.output_image_size)))
        for idx, que in enumerate(ques):
            images_array[idx, ...] = que.get()
        return images_array, labels_array

    def _get_next_files(self, label, size):
        return np.random.choice(self.class_file_dict[label], size=size)
