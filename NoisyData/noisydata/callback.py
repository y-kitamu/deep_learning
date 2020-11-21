import os
import time
import csv
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from noisydata.utility import Logger


class CustomCallback(Callback):
    """summary metrics, logging, save model, etc.
    """

    def __init__(self, solver, result_dir=None, callbacks=[]):
        self.solver = solver
        self.callbacks = callbacks

        if result_dir is None:
            result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results",
                                      self.solver.name)

        self.model_save_dir = os.path.join(result_dir, "models")
        self.csv_filename = os.path.join(result_dir, "csv", "train_log.csv")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(os.path.dirname(self.csv_filename)):
            os.makedirs(os.path.dirname(self.csv_filename))
        with open(self.csv_filename, 'w') as f:
            csv_writer = csv.writer(f)
            row = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
            for cb in self.callbacks:
                if hasattr(cb, "metrix"):
                    row += getattr(cb, "metrix")
            csv_writer.writerow(row)

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

        self.best_loss = 1e5
        self.best_acc = 0.0

        self.template = "Epoch {}, Loss : {:.4f}, Accuracy : {:.3f}, Val Loss : {:.4f}, Val Accuracy : {:.3f}, lr : {:.6f}, Elapsed Time : {:.2f} (sec)"
        self.lr_scheduler = None
        self.n_batches_per_epoch = None

    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)

        if "lr_scheduler" in logs:
            self.lr_scheduler = logs["lr_scheduler"]
        if "n_batches_per_epoch" in logs:
            self.n_batches_per_epoch = logs["n_batches_per_epoch"]

        Logger().logging("Model         = {}".format(self.solver.model.name))
        Logger().logging("Optimizer     = {}".format(self.solver.optimizer._name))
        Logger().logging("Loss function = {}".format(self.solver.loss_func.name))
        Logger().logging("Train / Val   = {} / {}".format(self.solver.x_train.shape[0],
                                                          self.solver.x_val.shape[0]))
        Logger().logging(
            "Batch size    = {}".format(-1 if "batch_size" not in logs else logs["batch_size"]))

    def on_train_batch_end(self, batch, logs=None):
        if "loss" in logs:
            self.train_loss(logs["loss"])
        if "label" in logs and "pred" in logs:
            self.train_accuracy(logs["label"], logs["pred"])
        print("\r {} / {} : loss={:.4f}".format(batch, self.n_batches_per_epoch,
                                                self.train_loss.result()),
              end="")
        del logs

    def on_test_batch_end(self, batch, logs=None):
        if "loss" in logs:
            self.val_loss(logs["loss"])
        if "label" in logs and "pred" in logs:
            self.val_accuracy(logs["label"], logs["pred"])
        print("\r {} : loss={:.4f}".format(batch, self.val_loss.result()), end="")
        del logs

    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

        self.start_time = time.time()
        if self.lr_scheduler is not None:
            self.solver.optimizer.lr = self.lr_scheduler(epoch)
        self.lr = self.solver.optimizer.lr.numpy()

    def on_epoch_end(self, epoch, logs=None):
        print("\r", end="")
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

        train_loss = self.train_loss.result().numpy()
        train_acc = self.train_accuracy.result().numpy()
        val_loss = self.val_loss.result().numpy()
        val_acc = self.val_accuracy.result().numpy()
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.solver.model.save(os.path.join(self.model_save_dir, "best_loss.hdf5"))
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.solver.model.save(os.path.join(self.model_save_dir, "best_acc.hdf5"))
        self.solver.model.save(os.path.join(self.model_save_dir, "last_epoch.hdf5"))

        elapsed_time = time.time() - self.start_time
        Logger().logging(
            self.template.format(epoch, train_loss, train_acc, val_loss, val_acc, self.lr, elapsed_time))
        with open(self.csv_filename, 'a') as f:
            csv_writer = csv.writer(f)
            row = [epoch, train_loss, train_acc, val_loss, val_acc, self.lr]
            for cb in self.callbacks:
                if hasattr(cb, "metrix"):
                    row += [getattr(cb, metric) for metric in getattr(cb, "metrix")]
            csv_writer.writerow(row)


class HandleNoisyLabel(Callback):
    """To handle noised label, update(modify) label, calculate ground truth accuracy.
    Update labels using softmax predictions of the target.
    reference : Tanaka et al. 2018, "Joint Optimization Framework for Learning with Noisy Labels"

    Args:
        min_epoch (int)          : epoch to start updating label
        max_epoch (int)          : epoch to end updating label
        n_accumulate (int)       : num epochs of which average output probability of the model is used
                                   to update label.
        data (tuple of np.array) : tuple of datas which contain
                                   (x_train, x_val, y_train, y_val, gt_y_train, gt_y_val).
                                   Notice that `y_train` and `y_val` is one hot encoding while
                                   `gt_y_train` and `gt_y_val` is label encoding.
    """

    def __init__(self,
                 data,
                 min_epoch=70,
                 max_epoch=200,
                 n_accumulate=10,
                 update_label=False,
                 num_class=10):
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.n_accumulate = n_accumulate
        self.x_train, self.y_train, self.x_val, self.y_val, _, _, self.gt_y_train, self.gt_y_val = data
        self.gt_y_val = self.gt_y_val.reshape(-1)
        self.y_train_preds = np.zeros((n_accumulate, *self.y_train.shape))
        self.y_val_preds = np.zeros((n_accumulate, *self.y_val.shape))
        self.metrix = ["gt_val_accuracy"]
        self.gt_val_accuracy = 0.0
        self.is_update_label = update_label
        self.num_class = num_class

        self.solver = None
        self.batch_size = None

    def on_train_begin(self, logs=None):
        self.batch_size = self.batch_size if "batch_size" not in logs else logs["batch_size"]

    def on_epoch_begin(self, epoch, logs=None):
        if self.is_update_label and epoch > self.min_epoch:
            self.solver.y_train = self.y_train
            self.solver.y_val = self.y_val
            self.solver.create_train_ds(self.batch_size)

    def on_epoch_end(self, epoch, logs=None):
        if self.solver is None or self.batch_size is None:
            Logger().logging("solver or batch_size is not set. Skip update labels.")
            return

        accum_idx = epoch % self.n_accumulate
        steps_per_valdata = math.ceil(self.y_val.shape[0] / self.batch_size)
        for batch_idx in range(steps_per_valdata):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.y_val.shape[0])
            pred = self.solver.predict(self.x_val[start:end])
            self.y_val_preds[accum_idx, start:end, ...] = pred.numpy()
            del pred

            print("\r Update validation labels... {} / {}".format(batch_idx, steps_per_valdata), end="")
        print("\r", end="")

        self.gt_val_accuracy \
            = sum(self.y_val_preds[accum_idx].argmax(axis=-1) == self.gt_y_val) / self.y_val.shape[0]
        Logger().logging("Ground Truth Val accuracy : {:.3f}".format(self.gt_val_accuracy))

        if epoch < self.min_epoch - self.n_accumulate or not self.is_update_label:
            return
        if epoch > self.max_epoch:
            return

        steps_per_traindata = math.ceil(self.y_train.shape[0] / self.batch_size)
        for batch_idx in range(steps_per_traindata):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.y_train.shape[0])
            pred = self.solver.predict(self.x_train[start:end])
            self.y_train_preds[accum_idx, start:end, ...] = pred.numpy()
            del pred

            print("\r Update training labels... {} / {}".format(batch_idx, steps_per_traindata), end="")
        print("\r", end="")

        y_train_pred_label = self.y_train_preds.mean(axis=0)
        y_val_pred_label = tf.keras.utils.to_categorical(self.y_val_preds.sum(axis=0).argmax(axis=-1),
                                                         num_classes=self.num_class)
        if epoch >= self.min_epoch:
            self.y_train = y_train_pred_label
            self.y_val = y_val_pred_label

        train_label_accuracy = float((y_train_pred_label.argmax(axis=1).astype(int).reshape(-1, 1)
                                      == self.gt_y_train.astype(int)).sum()) / self.y_train.shape[0]
        val_label_accuracy = float(
            (y_val_pred_label.argmax(axis=1) == self.gt_y_val).sum()) / self.y_val.shape[0]
        # import pdb
        # pdb.set_trace()
        Logger().logging("Train label accuracy : {:.3f}, Val label accuracy : {:.3f}".format(
            train_label_accuracy, val_label_accuracy))
