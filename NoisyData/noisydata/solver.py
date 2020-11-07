import datetime
import os
import csv

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].plot(history.hitory["loss"])
    axes[0].plot(history.history["val_loss"])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss value")
    axes[0].set_legend(["train", "test"], loc="upper left")

    axes[1].plot(history.history["acc"])
    axes[1].plot(history.history["val_acc"])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_legend(["train", "test"], loc="upper left")
    plt.show()


def plot_csv(csv_filename):
    with open(csv_filename, "r") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        rows = [[] for _ in header]
        for row in csv_reader:
            for idx, val in enumerate(row):
                rows[idx].append(float(val))

    metrics = ["accuracy", "loss"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 7, 7))
    for idx, metric in enumerate(metrics):
        metric_idx = header.index(metric)
        val_metric_idx = header.index("val_{}".format(metric))
        # import pdb; pdb.set_trace()
        axes[idx].plot(rows[metric_idx], label="Training {}".format(metric))
        axes[idx].plot(rows[val_metric_idx], label="Validation {}".format(metric))
        axes[idx].set_xlabel("epoch")
        axes[idx].set_ylabel(metric)
    plt.show()


def default_lr_scheduler(epoch, lr=0.1, step=50):
    new_lr = lr * (0.1 ** (epoch // step))
    return new_lr


class Solver(object):

    def __init__(self, model, data, loss_func=None):
        self.model = model
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.name = datetime.datetime.now(JST).strftime("%Y%m%d_%H%M%S")

        self.loss_func = loss_func

        self.lr_scheduler = None

    def _on_epoch_end(self, epoch, logs=None):
        print(self.model.optimizer.lr.numpy())

    def prepare_callbacks(self):
        callbacks = []
        if self.lr_scheduler is not None:
            callbacks.append(LearningRateScheduler(self.lr_scheduler))

        csv_dir = os.path.dirname("./csv")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        callbacks.append(CSVLogger(filename=os.path.join(csv_dir, "{}.csv".format(self.name))))

        model_save_dir = "./model/{}/".format(self.name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        callbacks.append(ModelCheckpoint(
            os.path.join(model_save_dir, "best_val_acc.hdf5"), save_weights_only=True,
            verbose=0, monitor="val_accuracy", mode="max", save_best_only=True))
        callbacks.append(ModelCheckpoint(
            os.path.join(model_save_dir, "best_val_loss.hdf5"),
            save_weights_only=True, verbose=0, monitor="val_loss", mode="min", save_best_only=True))
        callbacks.append(ModelCheckpoint(
            os.path.join(model_save_dir, "epoch.hdf5"),
            save_weights_only=True, verbose=0, save_best_only=False))

        return callbacks

    def fit(self, epochs=200, batch_size=120, lr_scheduler=None, data_augmentation=True, callbacks=None):
        self.lr_scheduler = lr_scheduler
        callbacks = self.prepare_callbacks()
        if data_augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                width_shift_range=4,
                height_shift_range=4,
                horizontal_flip=True,
                vertical_flip=False
            )
            train_gen = datagen.flow(self.x_train, self.y_train, batch_size=batch_size)
            history = self.model.fit_generator(
                generator=train_gen,
                epochs=epochs,
                steps_per_epoch=int(len(self.x_train) / batch_size + 1),
                callbacks=callbacks,
                validation_data=(self.x_val, self.y_val),
            )
        else:
            history = self.model.fit(
                self.x_train, self.y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                steps_per_epoch=int(len(self.x_train) / batch_size + 1),
                callbacks=callbacks,
                validation_data=(self.x_val, self.y_val)
            )
        return history

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, )

    def train(self, epochs=200, batch_size=120, lr_scheduler=None, data_augmentation=True):
        if self.loss_func is None:
            pass


    def test(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print("test data loss = {:.2f}, acc = {:.4f}".format(loss, acc))
