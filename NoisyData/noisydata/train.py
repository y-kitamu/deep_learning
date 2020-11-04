import os
import datetime

import tensorflow as tf
import numpy as np

from . import datasets
from .utility import Logging
from .callback import MetricsTraceCallback


JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')
AUTOTUNE = tf.data.experimental.AUTOTUNE
INIT_LR = 1e-1


def lr_scheduler(epoch):
    new_lr = INIT_LR * (0.1 ** (epoch // 50))
    return new_lr


def augmentation(image, label, gt_label):
    IMG_SIZE = image.shape[1]
    IMG_SHAPE = image.shape
    image = tf.image.random_flip_up_down(image)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 8, IMG_SIZE + 8)
    image = tf.image.random_crop(image, IMG_SHAPE)
    return image, label, gt_label


class Trainer:
    EPOCHS = 300
    N_CLASSES = 10

    def __init__(self,
                 model,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.SGD(),
                 lr_scheduler=lr_scheduler,
                 batch_size=160,
                 n_pred_pool=10,
                 min_label_update_epoch=70,
                 is_one_hot=True,
                 save_model_interval=10,
                 noise_ratio=0.0,
                 output_root_dir=os.path.join(os.path.dirname(__file__), "../models")):
        self.model = model
        self.loss_object = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.n_pred_pool = n_pred_pool
        self.min_label_update_epoch = min_label_update_epoch
        self.is_one_hot = is_one_hot
        self.noise_ratio = noise_ratio
        self.save_model_interval = save_model_interval

        self.name = datetime.datetime.now(JST).strftime("%Y%m%d_%H%M%S")

        self.load_dataset()
        self.pred_pool = np.zeros((n_pred_pool, self.y_train.shape[0], self.N_CLASSES))
        self.callback = MetricsTraceCallback(self.model, self.name)
        self.log_trainer_info()

    def log_trainer_info(self):
        Logging().logging("Model name = {}".format(self.model.name))
        Logging().logging("Loss       = {}".format(self.loss_object.name))
        Logging().logging("Optimizer  = {}".format(self.optimizer.get_config()["name"]))
        Logging().logging("Epochs     = {}".format(self.EPOCHS))
        Logging().logging("Batch size = {}".format(self.batch_size))
        self.model.build(self.x_train.shape)
        self.model.summary()

    def load_dataset(self):
        self.x_train, self.y_train, self.gt_y_train, self.x_val, self.y_val,\
            self.gt_y_val, self.x_test, self.y_test = datasets.create_noised_cifar10_data(
                self.noise_ratio)
        if self.is_one_hot:
            self.y_train = tf.one_hot(self.y_train, self.N_CLASSES)
            self.y_val = tf.one_hot(self.y_val, self.N_CLASSES)

        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train, self.gt_y_train)).map(
                augmentation, num_parallel_calls=AUTOTUNE
            ).shuffle(
                self.x_train.shape[0]).batch(self.batch_size)
        self.val_ds = tf.data.Dataset.from_tensor_slices(
            (self.x_val, self.y_val, self.gt_y_val)).batch(self.batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)).batch(self.batch_size)
        Logging().logging("Successfully load noised cifar10. train = {}, val = {}, test = {}".format(
            self.x_train.shape[0], self.x_val.shape[0], self.x_test.shape[0]
        ))

    def update_label(self, epoch):
        ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.batch_size)
        pool_idx = epoch % self.n_pred_pool
        for idx, batch in enumerate(ds):
            self.pred_pool[pool_idx, self.batch_size * idx:self.batch_size * (idx + 1)] \
                = self.model(batch).numpy()

        if self.min_label_update_epoch >= epoch:
            self.y_train = self.pred_pool.mean(axis=0).reshape(*self.y_train.shape)

    def update_train_dataset(self, epoch):
        self.update_label(epoch)
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train, self.gt_y_train)).map(
                augmentation, num_parallel_calls=AUTOTUNE
            ).shuffle(self.x_train.shape[0]).batch(
                self.batch_size)

    @tf.function
    def train_step(self, images, labels, *args):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = tf.add(self.loss_object(labels, predictions), tf.reduce_sum(self.model.losses))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if self.is_one_hot:
            labels = tf.math.argmax(labels, axis=2)
        self.callback.on_train_batch_end(
            self.batch_idx, {"loss": loss, "y_true": labels, "y_pred": predictions})

    @tf.function
    def validation_step(self, images, labels, *args):
        predictions = self.model(images, training=False)
        v_loss = tf.add(self.loss_object(labels, predictions), self.model.losses)

        if self.is_one_hot:
            labels = tf.math.argmax(labels, axis=2)
        self.callback.on_test_batch_end(
            self.batch_idx, {"loss": v_loss, "y_true": labels, "y_pred": predictions})

    def train(self):
        n_data = self.x_train.shape[0]
        for epoch in range(self.EPOCHS):
            self.callback.on_epoch_begin(epoch)
            for idx, batch in enumerate(self.train_ds):
                self.batch_idx = idx
                print("\r {} / {}".format(idx * self.batch_size, n_data), end="")
                self.train_step(*batch)

            print("\r", end="")

            for idx, batch in enumerate(self.val_ds):
                self.batch_idx = idx
                self.validation_step(*batch)
            self.callback.on_epoch_end(epoch)

            if self.lr_scheduler is not None:
                new_lr = self.lr_scheduler(epoch)
                if abs(new_lr - self.optimizer.lr) > 1e-8:
                    Logging.logging("New Learning rate : {:.5f}".format(new_lr))
                    self.optimizer.lr = new_lr

    def test(self, name="best_loss"):
        self.model.load_weights(os.path.join(self.model_output_dir, name))
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
        for batch in self.test_ds:
            predictions = self.model(batch[0], training=False)
            v_loss = self.loss_object(batch[1], predictions)

            self.test_loss(v_loss)
            labels = batch[1]
            if self.is_one_hot:
                labels = tf.math.argmax(batch[1], axis=2)
            self.test_accuracy(labels, predictions)
