import os
import time
import shutil

import tensorflow as tf
import numpy as np

from . import datasets
from . import models
from .utility import Logging, CSVLogger

AUTOTUNE = tf.data.experimental.AUTOTUNE


class LRShedler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, init_lr, decay_step):
        self.init_lr = init_lr
        self.decay_step = decay_step
        self.lr = self.init_lr

    def __call__(self):
        return self.lr

    def get_config(self):
        return {
            "initial_lr": self.init_lr,
            "current_lr": self.lr,
            "decay_step_epoch": self.decay_step
        }

    def update(self, epoch):
        level = int(epoch / self.decay_step)
        self.lr = self.init_lr * 10 ** (-level)


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
                 model=models.MyModel(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.SGD(),
                 lr_scheduler=None,
                 batch_size=32,
                 n_pred_pool=10,
                 min_label_update_epoch=70,
                 is_one_hot=True,
                 save_model_interval=10,
                 output_root_dir=os.path.join(os.path.dirname(__file__), "../models")):
        self.model = model
        self.loss_object = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.n_pred_pool = n_pred_pool
        self.min_label_update_epoch = min_label_update_epoch
        self.is_one_hot = is_one_hot
        self.save_model_interval = save_model_interval

        self.load_dataset()
        self.pred_pool = np.zeros((n_pred_pool, self.y_train.shape[0], self.N_CLASSES))

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

        self.best_loss = 1e5
        self.best_acc = 0.0

        self.csv_logger = CSVLogger(self, ["train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        self.title = os.path.basename(self.csv_logger.output_filename).split(".")[0]
        self.model_output_dir = os.path.join(output_root_dir, self.title)
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)
        self.log_trainer_info()

    def log_trainer_info(self):
        Logging().logging("Model name = {}".format(self.model.name))
        Logging().logging("Loss       = {}".format(self.loss_object.name))
        Logging().logging("Optimizer  = {}".format(self.optimizer.get_config()["name"]))
        Logging().logging("Epochs     = {}".format(self.EPOCHS))
        Logging().logging("Batch size = {}".format(self.batch_size))
        Logging().logging("CSV log    = {}".format(self.csv_logger.output_filename))
        Logging().logging("Output dir = {}".format(self.model_output_dir))
        self.model.build(self.x_train.shape)
        self.model.summary()

    def load_dataset(self):
        # self.x_train, self.y_train, self.gt_y_train, self.x_val, self.y_val,\
        #     self.gt_y_val, self.x_test, self.y_test = datasets.create_noised_cifar10_data()
        self.x_train, self.y_train, self.gt_y_train, self.x_val, self.y_val,\
            self.gt_y_val, self.x_test, self.y_test = datasets.create_cifar10_data()
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

    # @tf.function
    def train_step(self, images, labels, *args):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # import pdb; pdb.set_trace()
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        # import pdb; pdb.set_trace()
        if self.is_one_hot:
            labels = tf.math.argmax(labels, axis=2)
        self.train_accuracy(labels, predictions)

    # @tf.function
    def validation_step(self, images, labels, *args):
        predictions = self.model(images, training=False)
        v_loss = self.loss_object(labels, predictions)

        self.val_loss(v_loss)
        if self.is_one_hot:
            labels = tf.math.argmax(labels, axis=2)
        self.val_accuracy(labels, predictions)

    def _save_weight(self, dst_dir, output_log=False):
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        self.model.save_weights(dst_dir)
        if output_log:
            Logging().logging("Save model : {}".format(os.path.basename(dst_dir)))

    def save_model(self, epoch):
        if (epoch + 1) % self.save_model_interval == 0:
            self._save_weight(os.path.join(self.model_output_dir, "epoch_{:04d}".format(epoch + 1)))

        if self.val_accuracy.result() > self.best_acc - 1e-5:
            self.best_acc = self.val_accuracy.result()
            self._save_weight(os.path.join(self.model_output_dir, "best_acc"), True)

        if self.val_loss.result() < self.best_loss + 1e-5:
            self.best_loss = self.val_loss.result()
            self._save_weight(os.path.join(self.model_output_dir, "best_loss"), True)

    def train(self):
        template = "Epoch : {}, Loss : {:.3f}, Accuracy : {:.3f}, Val Loss : {:.3f}, Val Accuracy : {:.3f}, Elapsed Time : {:.2f} [sec]"
        n_data = self.x_train.shape[0]

        for epoch in range(self.EPOCHS):
            start_time = time.time()
            for idx, batch in enumerate(self.train_ds):
                print("\r {} / {}".format(idx * self.batch_size, n_data), end="")
                self.train_step(*batch)
            print("\r", end="")

            for batch in self.val_ds:
                self.validation_step(*batch)

            self.save_model(epoch)

            elapsed_time = time.time() - start_time
            Logging().logging(
                template.format(epoch + 1, self.train_loss.result(),
                                self.train_accuracy.result() * 100, self.val_loss.result(),
                                self.val_accuracy.result() * 100, elapsed_time))
            self.csv_logger.logging()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            if self.lr_scheduler is not None:
                self.lr_scheduler.update(epoch)

        self._save_weights(os.path.join(self.model_output_dir, "last"))
            # self.update_train_dataset(epoch)

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
