import os
import shutil
import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from .utility import CSVLogger, Logging


class MetricsTraceCallback(Callback):
    MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self, model, name):
        self.model = model
        self.csv_logger = CSVLogger(
            trainer=self,
            metrics=["train_loss", "train_accuracy", "val_loss", "val_accuracy"],
            output_filename=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "log", "{}.csv".format(name))
        )
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
        self.best_loss = 1e5
        self.best_acc = 0.0
        self.save_model_interval = 10

        self.model_output_dir = os.path.join(self.MODEL_OUTPUT_DIR, name)
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.save_model(epoch)
        self.csv_logger.logging(epoch)

        elapsed_time = time.time() - self.start_time
        template = "Epoch : {}, Loss : {:.3f}, Accuracy : {:.3f}, Val Loss : {:.3f}, Val Accuracy : {:.3f}, Elapsed Time : {:.2f} [sec]"
        Logging().logging(
            template.format(epoch + 1, self.train_loss.result(),
                            self.train_accuracy.result() * 100, self.val_loss.result(),
                            self.val_accuracy.result() * 100, elapsed_time))

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

    def on_test_batch_end(self, batch, logs=None):
        self.val_loss(logs["loss"])
        self.val_accuracy(logs["y_true"], logs["y_pred"])

    def on_train_batch_end(self, batch, logs=None):
        self.train_loss(logs["loss"])
        self.train_accuracy(logs["y_true"], logs["y_pred"])

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
