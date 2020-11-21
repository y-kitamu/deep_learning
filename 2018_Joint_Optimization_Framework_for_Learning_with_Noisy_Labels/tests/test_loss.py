import os
import math

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from noisydata import loss


@pytest.mark.loss
def test_label_optim_loss():
    # normal kl divergence
    loss_fun = loss.create_label_optim_loss(0, 0, tf.constant([0.5] * 2))
    y_true = tf.constant([[0.5, 0.5]])
    y_pred = tf.constant([[0.5, 0.5]])
    assert abs(loss_fun(y_true, y_pred).numpy()) < 1e-5

    # test for prior distribution (loss_p)
    loss_fun = loss.create_label_optim_loss(1.0, 0, tf.constant([0.5] * 2))
    y_true = tf.constant([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
    y_pred = tf.constant([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
    assert abs(loss_fun(y_true, y_pred).numpy()) < 1e-5

    y_true = tf.constant([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])
    y_pred = tf.constant([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])
    pred_loss = 0.5 * math.log(0.5 * 4 / 2.6) + 0.5 * math.log(0.5 * 4 / 1.4)
    res = loss_fun(y_true, y_pred).numpy()
    assert abs(res - pred_loss) < 1e-5

    # test for loss_e
    loss_fun = loss.create_label_optim_loss(0, 1.0, tf.constant([0.5] * 2))
    y_true = tf.constant([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])
    y_pred = tf.constant([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])
    pred_loss = -(4 * 0.8 * math.log(0.8) + 4 * 0.2 * math.log(0.2)) / 8
    res = loss_fun(y_true, y_pred).numpy()
    assert abs(res - pred_loss) < 1e-5
