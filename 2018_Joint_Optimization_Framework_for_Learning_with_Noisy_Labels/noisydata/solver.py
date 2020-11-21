import datetime
import os
import math

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from noisydata.utility import Logger
from noisydata.callback import CustomCallback
from noisydata.augmentation import augmentation

JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')


class Solver(object):
    """ Run training and test
    Args:
        model (tf.keras.Model)                 : training (or test) model
        data (tuple of np.array)               : tuple of data:
                                                 (x_train, y_train, x_val, y_val, x_test, y_test)
        loss_func (tf.keras.losses.Loss)       : loss function
        optimier (tf.keras.optimizers)         : optimizer
        aug_func (function)                    : image augmentation function
        callbacks (list of tf.keras.callbacks) : callbacks
    """

    def __init__(self,
                 model,
                 data,
                 loss_func=None,
                 optimizer=None,
                 aug_func=augmentation,
                 callbacks=[],
                 start_epoch=0,
                 weights_path=None):
        self.model = model
        if len(data) == 6:
            self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        elif len(data) == 8:
            self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, _, _ = data

        self.name = datetime.datetime.now(JST).strftime("%Y%m%d_%H%M%S")

        self.loss_func = loss_func
        self.optimizer = optimizer
        self.aug_func = augmentation
        self.callback = CustomCallback(self, callbacks=callbacks)
        for cb in self.callback.callbacks:
            cb.solver = self

        self.start_epoch = start_epoch
        self.lr_scheduler = None
        self.load_weights(weights_path)

    def load_weights(self, weights_path):
        if weights_path is None or not os.path.exists(weights_path):
            return
        Logger().logging("load weigths of {}".format(weights_path))
        self.model.load_weights(weights_path)

    def prepare_callbacks(self):
        callbacks = []
        if self.lr_scheduler is not None:
            callbacks.append(LearningRateScheduler(self.lr_scheduler))

        csv_dir = os.path.dirname("./results/{}/csv".format(self.name))
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        callbacks.append(CSVLogger(filename=os.path.join(csv_dir, "train_log.csv")))

        model_save_dir = "../results/{}/model/".format(self.name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        callbacks.append(
            ModelCheckpoint(os.path.join(model_save_dir, "best_val_acc.hdf5"),
                            save_weights_only=True,
                            verbose=0,
                            monitor="val_accuracy",
                            mode="max",
                            save_best_only=True))
        callbacks.append(
            ModelCheckpoint(os.path.join(model_save_dir, "best_val_loss.hdf5"),
                            save_weights_only=True,
                            verbose=0,
                            monitor="val_loss",
                            mode="min",
                            save_best_only=True))
        callbacks.append(
            ModelCheckpoint(os.path.join(model_save_dir, "epoch.hdf5"),
                            save_weights_only=True,
                            verbose=0,
                            save_best_only=False))

        return callbacks

    def fit(self, epochs=200, batch_size=120, lr_scheduler=None, data_augmentation=True, callbacks=None):
        """Training via model.fit()
        """
        self.lr_scheduler = lr_scheduler
        callbacks = self.prepare_callbacks()
        if data_augmentation:
            datagen = ImageDataGenerator(featurewise_center=False,
                                         samplewise_center=False,
                                         featurewise_std_normalization=False,
                                         samplewise_std_normalization=False,
                                         zca_whitening=False,
                                         width_shift_range=4,
                                         height_shift_range=4,
                                         horizontal_flip=True,
                                         vertical_flip=False)
            train_gen = datagen.flow(self.x_train, self.y_train, batch_size=batch_size)
            history = self.model.fit_generator(
                generator=train_gen,
                epochs=epochs,
                steps_per_epoch=int(len(self.x_train) / batch_size + 1),
                callbacks=callbacks,
                validation_data=(self.x_val, self.y_val),
            )
        else:
            history = self.model.fit(self.x_train,
                                     self.y_train,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     steps_per_epoch=int(len(self.x_train) / batch_size + 1),
                                     callbacks=callbacks,
                                     validation_data=(self.x_val, self.y_val))
        return history

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_func(labels, predictions) + self.model.losses
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss, "pred": predictions, "label": labels}

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        loss = self.loss_func(labels, predictions)
        return {"loss": loss, "pred": predictions, "label": labels}

    @tf.function
    def predict(self, images):
        predictions = self.model(images)
        return predictions

    def create_train_ds(self, batch_size):
        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        train_ds = train_ds.map(self.aug_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.repeat().shuffle(self.x_train.shape[0])
        train_ds = train_ds.prefetch(buffer_size=batch_size).batch(batch_size)
        return train_ds

    def train(self, epochs=200, batch_size=120, lr_scheduler=None):
        """customized training function (alternative to model.fit())
        """
        self.model.compile(self.optimizer, self.loss_func)
        self.callback.on_train_begin({
            "batch_size": batch_size,
            "lr_scheduler": lr_scheduler,
            "n_batches_per_epoch": int(self.x_train.shape[0] / batch_size + 1)
        })
        if self.loss_func is None:
            Logger().logging("self.loss_func must not be None. Abort")
            return
        if self.optimizer is None:
            Logger().logging("self.optimizer must not be None. Abort")
            return

        self.train_ds = self.create_train_ds(batch_size)
        steps_per_epoch = math.ceil(self.x_train.shape[0] / batch_size)
        for epoch in range(self.start_epoch, epochs):
            self.callback.on_epoch_begin(epoch)
            idx = 0
            for images, labels in self.train_ds:
                logs = self.train_step(images, labels)
                self.callback.on_train_batch_end(idx, logs)
                idx += 1
                if steps_per_epoch == idx:
                    break

            for idx in range(0, int(self.y_val.shape[0] / batch_size + 1)):
                start = idx * batch_size
                end = min((idx + 1) * batch_size, self.y_val.shape[0])
                logs = self.test_step(self.x_val[start:end], self.y_val[start:end])
                self.callback.on_test_batch_end(idx, logs)

            self.callback.on_epoch_end(epoch)

    def test(self, weight_path=None):
        if self.optimizer and self.loss_func:
            self.model.compile(self.optimizer, self.loss_func, metrics=["accuracy"])
        if weight_path:
            self.model.load_weights(weight_path)
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print("test data loss = {:.2f}, acc = {:.4f}".format(loss, acc))
