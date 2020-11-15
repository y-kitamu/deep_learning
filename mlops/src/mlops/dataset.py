import os
import sys
import glob
import queue
import random
import multiprocessing as mp

import numpy as np
import tensorflow as tf
import cv2


class TrainDataGenerator:
    """
    """

    def __init__(self,
                 input_image_root_dir,
                 batch_size=120,
                 background_class_idx=0,
                 edge_class_idx=5,
                 num_files_per_batch=5,
                 flaw_classes=[1, 2, 3, 4],
                 background_ratio=0.6,
                 edge_ratio=0.6,
                 flaw_ratios=[0.1, 1.0, 1.0, 1.0],
                 output_image_size=24,
                 num_channel=1,
                 process_num=6):
        self.batch_size = batch_size
        self.num_flaws_per_batch = int(batch_size / (1 + background_ratio + edge_ratio))
        self.num_flawless_per_batch = batch_size - self.num_flaws_per_batch
        self.output_image_size = output_image_size
        self.num_channel = num_channel

        candidate = glob.glob(os.path.join(input_image_root_dir, "{}*".format(background_class_idx)))
        if len(candidate) != 1:
            print("Directory hierarcy is not collect. len(candidate) must be 1, not {}".format(
                len(candidate)))
            sys.exit(-1)
        background_dirname = candidate[0]

        self.flawless_gen = TrainFlawlessDataGenerator(
            input_image_dir=background_dirname,
            num_flawless_per_batch=self.num_flawless_per_batch,
            num_files_per_batch=num_files_per_batch,
            output_image_size=output_image_size,
            num_channel=num_channel,
            background_class_idx=background_class_idx,
            edge_class_idx=edge_class_idx,
            edge_ratio=edge_ratio / (edge_ratio + background_ratio),
            process_num=int(process_num / 2),
        )
        self.flaws_gen = TrainFlawDataGenerator(
            input_image_root_dir=input_image_root_dir,
            num_flaws_per_batch=self.num_flaws_per_batch,
            classes=flaw_classes,
            output_image_size=output_image_size,
            num_channel=num_channel,
            sampling_weights=flaw_ratios,
            process_num=int(process_num / 2),
        )

    def __call__(self):
        images_array = np.zeros(
            (self.batch_size, self.output_image_size, self.output_image_size, self.num_channel))
        labels_array = np.zeros(self.batch_size)
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
            output_shapes=((self.batch_size, self.output_image_size, self.output_image_size,
                            self.num_channel), (self.batch_size)))
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

    def __init__(self,
                 input_image_dir,
                 num_flawless_per_batch,
                 num_files_per_batch,
                 output_image_size=24,
                 num_channel=1,
                 background_class_idx=0,
                 edge_class_idx=5,
                 edge_ratio=0.5,
                 max_thresh=1500,
                 variance_thresh=7500,
                 is_divide_edge=True,
                 process_num=3):
        self.input_image_dir = input_image_dir
        self.file_list = list(glob.glob(os.path.join(self.input_image_dir, "*.png")))
        self.num_flawless_per_batch = num_flawless_per_batch
        self.num_files_per_batch = num_files_per_batch
        self.output_image_size = output_image_size
        self.num_channel = num_channel

        self.background_class_idx = background_class_idx
        self.edge_class_idx = edge_class_idx
        self.edge_ratio = edge_ratio

        self.max_thresh = max_thresh
        self.variance_thresh = variance_thresh
        self.is_divide_edge = is_divide_edge

        self.sampling_weights = None
        self.file_que = queue.Queue()

        self._check_variables_assertion()

    def _check_variables_assertion(self):
        assert self.num_flawless_per_batch > 0
        assert self.num_files_per_batch > 0
        assert len(self.file_list) > 0
        assert not self.edge_ratio > 1.0
        assert not self.edge_ratio < 0.0

    def __call__(self):
        """
        Return :  generator of flawless image data batch :
                  [num_flawless_per_batch, ouptu_image_size, output_image_size, num_channel]
        """
        images_array = np.zeros((self.num_flawless_per_batch, self.output_image_size,
                                 self.output_image_size, self.num_channel),
                                dtype=np.float32)
        labels_array = np.zeros(self.num_flawless_per_batch, dtype=np.float32)
        num_crops_per_image = int(self.num_flawless_per_batch / self.num_files_per_batch)
        while True:
            remaining = self.num_flawless_per_batch
            while (remaining > 0):
                num_crop = min(remaining, num_crops_per_image)
                filename = self._get_next_file()
                images, labels = self._crop_image(num_crop, filename)
                images_array[remaining - num_crop:remaining] = images
                labels_array[remaining - num_crop:remaining] = labels
                remaining -= num_crop
            yield images_array, labels_array

    def _crop_image(self, num_crop, filename):
        """
        Args:
            num_crop (int) :
        Return (np.array, np.array)
            cropped images array and labels array.
            shape of cropped images array is [num_crop, output_image_size, output_image_size,
            num_channel] and shape of labels array is [num_crop].
        """
        images_array = np.zeros(
            (num_crop, self.output_image_size, self.output_image_size, self.num_channel),
            dtype=np.float32)
        labels_array = np.zeros(num_crop)
        num_edge = int(num_crop * self.edge_ratio)
        num_background = num_crop - num_edge

        image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(image.shape) == 2:
            assert self.num_channel == 1
            image = image[..., None]

        delta = int(self.output_image_size / 2)
        mask_thresh = self.output_image_size / 4
        for i in range(num_crop):
            # repeat random crop until target (edge or background) image is obtained
            while True:
                # random crop from source image
                center_x = int(np.random.rand() * (image.shape[1] - self.output_image_size)) + delta
                center_y = int(np.random.rand() * (image.shape[0] - self.output_image_size)) + delta
                roi = image[center_y - delta:center_y + delta, center_x - delta:center_x + delta]
                # check if cropped image is valid (not masked)
                if (roi[0, :, 0] == 0).sum() > mask_thresh or \
                   (roi[-1, :, 0] == 0).sum() > mask_thresh or \
                   (roi[:, 0, 0] == 0).sum() > mask_thresh or (roi[:, -1, 0] == 0).sum() > mask_thresh:
                    continue

                # judge edge of label
                if self.is_divide_edge:
                    is_edge = self._judge_edge(roi)
                    if is_edge and num_edge > 0:
                        images_array[i, ...] = roi
                        labels_array[i] = self.edge_class_idx
                        num_edge -= 1
                        break
                    if not is_edge and num_background > 0:
                        images_array[i, ...] = roi
                        labels_array[i] = self.background_class_idx
                        num_background -= 1
                        break
                else:
                    images_array[i, ...] = roi
                    break
        return images_array, labels_array

    def _get_next_file(self):
        if self.file_que.empty():
            que_list = self.file_list
            if self.sampling_weights is not None:
                que_list = sum([[fname] * self.sampling_weights[fname] for fname in self.file_list], [])
            for fname in random.sample(que_list, len(que_list)):
                self.file_que.put(fname)
        return self.file_que.get()

    def _judge_edge(self, roi):
        return roi.max() > self.max_thresh and roi.var() > self.variance_thresh


class TrainFlawDataGenerator:
    """
    """

    def __init__(self,
                 input_image_root_dir,
                 num_flaws_per_batch,
                 classes=[1, 2, 3, 4],
                 output_image_size=24,
                 num_channel=1,
                 shift_range=8,
                 sampling_weights=None,
                 process_num=3):
        self.input_image_root_dir = input_image_root_dir
        self.num_flaws_per_batch = num_flaws_per_batch
        self.classes = classes
        self.output_image_size = output_image_size
        self.num_channel = 1
        self.shift_range = 8
        self.sampling_weights = None if sampling_weights is None else np.array(sampling_weights)
        self.class_file_dict = self._search_input_image_files()

    def _search_input_image_files(self):
        class_file_dict = {}
        for cls in self.classes:
            class_file_dict[cls] = [
                fname for fname in glob.glob(
                    os.path.join(self.input_image_root_dir, "{}*".format(cls), "*.bmp"))
            ]
        return class_file_dict

    def __call__(self):
        images_array = np.zeros(
            (self.num_flaws_per_batch, self.output_image_size, self.output_image_size, self.num_channel),
            dtype=np.float32)
        labels_array = np.zeros(self.num_flaws_per_batch, dtype=np.float32)
        weights = self.sampling_weights
        if weights is None or len(weights) != len(self.classes):
            weights = np.ones(len(self.classes))
        weights /= sum(weights)
        weights[-1] = 1 - sum(weights[0:-1])
        while True:
            labels_array[...] = np.random.choice(self.classes, size=self.num_flaws_per_batch, p=weights)
            for i in range(self.num_flaws_per_batch):
                image = self.augment_and_crop_image(labels_array[i])
                images_array[i, ...] = image
            yield images_array, labels_array

    def augment_and_crop_image(self, label):
        filename = self.get_next_file(label)
        image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(image.shape) == 2:
            assert self.num_channel == 1
            image = image[..., None]

        image = random_rotate_and_scale_and_shift(image, shift_range=self.shift_range)
        height, width = image.shape[:2]
        cx, cy = int(width / 2), int(height / 2)
        offset = int(self.output_image_size / 2)
        image = image[cy - offset:cy + offset, cx - offset:cx + offset]
        return image

    def get_next_file(self, label):
        return np.random.choice(self.class_file_dict[label])


def random_rotate_and_scale_and_shift(img, min_scale=1.0, max_scale=1.0, shift_range=8):
    angle = int(random.uniform(0, 2 * np.pi))
    scale = int(random.uniform(min_scale, max_scale))
    shift = np.random.randint(-shift_range, shift_range + 1, 2)
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), angle, scale)
    matrix[:, 2] += shift
    img = cv2.warpAffine(img, matrix, (width, height))
    if len(img.shape) == 2:
        img = img[..., None]
    return img
