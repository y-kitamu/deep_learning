import os
import sys
import glob
import queue
import random
import threading
import multiprocessing as mp
import dataclasses

import numpy as np
import tensorflow as tf
import cv2
from omegaconf import DictConfig


@dataclasses.dataclass
class AugmentParameters:
    min_scale: float = 1.0
    max_scale: float = 1.0
    shift_range: int = 8


@dataclasses.dataclass
class FlawlessParameters:
    classes: tuple = (0, 5)
    ratios: tuple = (0.6, 0.6)
    num_files_per_batch: int = 5
    num_process: int = 3
    # threshold parameters used in checking if a cropped image is edge or background.
    max_thresh: float = 1500
    variance_thresh: float = 7500
    # number of images created from a single cropped image by augmentation
    num_augment: int = 1


@dataclasses.dataclass
class FlawParameters:
    classes: tuple = (1, 2, 3, 4)
    ratios: tuple = (0.2, 1.0, 1.0, 1.0)
    num_process: int = 3


@dataclasses.dataclass
class DatasetParameters:
    train_data_root: str
    val_data_root: str
    batch_size: int = 100
    output_image_size: int = 24
    num_channel: int = 3
    augment_params: AugmentParameters = AugmentParameters()
    flawless_params: FlawlessParameters = FlawlessParameters()
    flaw_params: FlawParameters = FlawParameters()


def _judge_edge(roi, flawless_params):
    return roi.max() > flawless_params.max_thresh and roi.var() > flawless_params.variance_thresh


def _augment_and_crop_flawless_images(num_crop, filename, flawless_params, augment_params, num_channel,
                                      output_image_size):
    """
    Args:
        num_crop (int) :
        filename (str) :
        output_image_size (int):
        num_channel (int) :
        flawless_params (FlawlessParameters) :
    Return (np.array, np.array)
        cropped images array and labels array.
        shape of cropped images array is [num_crop, output_image_size, output_image_size,
        num_channel] and shape of labels array is [num_crop].
    """
    num_edge = int(num_crop * flawless_params.ratios[1] / sum(flawless_params.ratios))
    num_background = num_crop - num_edge

    background_idx = flawless_params.classes[0]
    edge_idx = flawless_params.classes[1]

    images_array = np.zeros((num_crop, output_image_size, output_image_size, num_channel),
                            dtype=np.float32)
    labels_array = np.zeros(num_crop)

    image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) == 2:
        assert num_channel == 1
        image = image[..., None]

    delta = output_image_size
    mask_thresh = output_image_size / 4
    for i in range(0, num_crop, flawless_params.num_augment):
        # repeat random crop until target (edge or background) image is obtained
        while True:
            # random crop from source image
            cx = int(np.random.rand() * (image.shape[1] - output_image_size)) + delta
            cy = int(np.random.rand() * (image.shape[0] - output_image_size)) + delta
            roi = image[cy - delta:cy + delta, cx - delta:cx + delta]
            # check if cropped image is valid (not masked)
            if (roi[0, :, 0] == 0).sum() > mask_thresh or \
               (roi[-1, :, 0] == 0).sum() > mask_thresh or \
               (roi[:, 0, 0] == 0).sum() > mask_thresh or (roi[:, -1, 0] == 0).sum() > mask_thresh:
                continue

            # judge edge of label
            is_edge = _judge_edge(roi, flawless_params)
            if num_edge <= 0:
                images_array[i, ...] = random_rotate_and_scale_and_shift_crop(
                    roi, augment_params, output_image_size)
                labels_array[i] = edge_idx if is_edge else background_idx
                break
            if is_edge and num_edge > 0:
                images_array[i, ...] = random_rotate_and_scale_and_shift_crop(
                    roi, augment_params, output_image_size)
                labels_array[i] = edge_idx
                num_edge -= 1
                break
            if not is_edge and num_background > 0:
                images_array[i, ...] = random_rotate_and_scale_and_shift_crop(
                    roi, augment_params, output_image_size)
                labels_array[i] = background_idx
                num_background -= 1
                break
    return images_array, labels_array


def _augment_and_crop_flaw_image(filename, aug_params, num_channel, output_image_size):
    image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) == 2:
        if num_channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = image[..., None]

    return random_rotate_and_scale_and_shift_crop(image,
                                                  aug_params=aug_params,
                                                  output_size=output_image_size)


def random_rotate_and_scale_and_shift_crop(img, aug_params, output_size=24):
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
    """

    def __init__(self, config: DictConfig, is_train=True):
        flawless_params = FlawlessParameters(**config["flawless"])
        flaws_params = FlawParameters(**config["flaws"])
        aug_params = AugmentParameters(**config["augment"])
        self.params = DatasetParameters(**config["overall"],
                                        flawless_params=flawless_params,
                                        flaw_params=flaws_params,
                                        augment_params=aug_params)
        data_root = self.params.train_data_root if is_train else self.params.val_data_root
        candidate = glob.glob(
            os.path.join(data_root, "{}*".format(self.params.flawless_params.classes[0])))
        if len(candidate) != 1:
            print("Directory hierarcy is not collect. len(candidate) must be 1, not {}".format(
                len(candidate)))
            sys.exit(-1)
        background_dirname = candidate[0]

        self.num_flaws_per_batch = int(self.params.batch_size /
                                       (1 + sum(self.params.flawless_params.ratios)))
        self.num_flawless_per_batch = self.params.batch_size - self.num_flaws_per_batch

        self.flawless_gen = TrainFlawlessDataGenerator(
            params=self.params,
            input_image_dir=background_dirname,
            num_flawless_per_batch=self.num_flawless_per_batch,
        )
        self.flaws_gen = TrainFlawDataGenerator(
            params=self.params,
            input_image_root_dir=data_root,
            num_flaws_per_batch=self.num_flaws_per_batch,
        )

    def __call__(self):
        images_array = np.zeros((self.params.batch_size, self.params.output_image_size,
                                 self.params.output_image_size, self.params.num_channel))
        labels_array = np.zeros(self.params.batch_size)

        for flawless, flaws in zip(self.flawless_gen(), self.flaws_gen()):
            images_array[:self.num_flawless_per_batch, ...] = flawless[0]
            labels_array[:self.num_flawless_per_batch] = flawless[1]
            images_array[self.num_flawless_per_batch:, ...] = flaws[0]
            labels_array[self.num_flawless_per_batch:] = flaws[1]
            yield images_array, labels_array

    def create_dataset(self):
        train_ds = tf.data.Dataset.from_generator(
            self,
            output_types=(tf.float32, tf.float32),
            output_shapes=((self.params.batch_size, self.params.output_image_size,
                            self.params.output_image_size, self.params.num_channel),
                           (self.params.batch_size)))
        return train_ds.prefetch(1)


class TrainFlawlessDataGenerator:
    """Generator class of Flawless image data.
    `__call__` method return batch of cropped flawless images :
    [num_flawless_per_batch, output_image_size, output_image_size, num_channel].

    Args:
        input_image_dir (str)        : Path to input image directory
        num_flawless_per_batch (int) : Number of flawless images per each training batch
        num_files_per_batch (int)    : Number of files used to create a single batch.
            For example, if num_files_per_batch is 1, all cropped flawless images in a single batch
            are cropped from the same image file.
        output_image_size (int)      : Output batch image size
        num_channel (int)            : Number of input and output image channel
        background_class_idx (int)   :
        edge_class_idx (int)         :
        edge_ratio (float)           : Edge image ratio of each output batch.
            `judege_edge` method judge whether the cropped image is edge or background.
        max_thresh (float)           : Parameter used in `judge_edge`
        variance_thresh (float)      : Parameter used in `judge_edge`
    """

    def __init__(self, params, input_image_dir, num_flawless_per_batch):
        self.params = params
        self.input_image_dir = input_image_dir
        self.num_flawless_per_batch = num_flawless_per_batch

        self.file_list = list(glob.glob(os.path.join(self.input_image_dir, "*.png")))
        self.file_que = queue.Queue()
        self.pool = mp.Pool(self.params.flawless_params.num_process)
        self.sampling_weights = None

        self._check_variables_assertion()

    def _check_variables_assertion(self):
        assert self.num_flawless_per_batch > 0
        assert self.params.flawless_params.num_files_per_batch > 0
        assert len(self.file_list) > 0
        assert len(self.params.flawless_params.classes) == len(self.params.flawless_params.ratios)
        for ratio in self.params.flaw_params.ratios:
            assert not ratio > 1.0
            assert not ratio < 0.0

    def __del__(self):
        self.pool.terminate()
        self.pool = None

    def __call__(self):
        """
        Return :  generator of flawless image data batch :
                  [num_flawless_per_batch, ouptu_image_size, output_image_size, num_channel]
        """
        images_array = np.zeros((self.num_flawless_per_batch, self.params.output_image_size,
                                 self.params.output_image_size, self.params.num_channel),
                                dtype=np.float32)
        labels_array = np.zeros(self.num_flawless_per_batch, dtype=np.float32)
        num_crops_per_image = int(self.num_flawless_per_batch /
                                  self.params.flawless_params.num_files_per_batch)
        while True:
            remaining = self.num_flawless_per_batch
            queues = []
            while (remaining > 0):
                num_crop = min(remaining, num_crops_per_image)
                filename = self._get_next_file()
                queues.append(
                    self.pool.apply_async(_augment_and_crop_flawless_images, (
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

                remaining -= num_crop

            start_idx = 0
            for que in queues:
                images, labels = que.get()
                num_element = labels.shape[0]
                images_array[start_idx:start_idx + num_element] = images
                labels_array[start_idx:start_idx + num_element] = labels
                start_idx += num_element
            yield images_array, labels_array

    def _get_next_file(self):
        if self.file_que.empty():
            que_list = self.file_list
            if self.sampling_weights is not None:
                que_list = sum([[fname] * self.sampling_weights[fname] for fname in self.file_list], [])
            for fname in random.sample(que_list, len(que_list)):
                self.file_que.put(fname)
        return self.file_que.get()


class TrainFlawDataGenerator:
    """
    """

    def __init__(self, params, input_image_root_dir, num_flaws_per_batch):
        self.params = params
        self.input_image_root_dir = input_image_root_dir
        self.num_flaws_per_batch = num_flaws_per_batch

        self.class_file_dict = self._search_input_image_files()
        self.pool = mp.Pool(self.params.flaw_params.num_process)

    def _search_input_image_files(self):
        class_file_dict = {}
        for cls in self.params.flaw_params.classes:
            class_file_dict[cls] = [
                fname for fname in glob.glob(
                    os.path.join(self.input_image_root_dir, "{}*".format(cls), "*.png"))
            ]
        return class_file_dict

    def __call__(self):
        images_array = np.zeros((self.num_flaws_per_batch, self.params.output_image_size,
                                 self.params.output_image_size, self.params.num_channel),
                                dtype=np.float32)
        labels_array = np.zeros(self.num_flaws_per_batch, dtype=np.float32)
        weights = np.array(self.params.flaw_params.ratios)
        weights /= sum(weights)
        weights[-1] = 1 - sum(weights[:-1])
        while True:
            ques = []
            labels_array[...] = np.random.choice(self.params.flaw_params.classes,
                                                 size=self.num_flaws_per_batch,
                                                 p=weights)
            for cls_idx in self.params.flaw_params.classes:
                cnt = (labels_array.astype(int) == cls_idx).sum()
                if cnt == 0:
                    continue
                for fname in self.get_next_file(cls_idx, cnt):
                    ques.append(
                        self.pool.apply_async(_augment_and_crop_flaw_image,
                                              (fname, self.params.augment_params,
                                               self.params.num_channel, self.params.output_image_size)))
            for idx, que in enumerate(ques):
                images_array[idx, ...] = que.get()
            yield images_array, labels_array

    def get_next_file(self, label, size):
        return np.random.choice(self.class_file_dict[label], size=size)
