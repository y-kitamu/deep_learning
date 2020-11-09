import tensorflow as tf
from tensorflow.keras.losses import KLDivergence


class LabelOptimLoss(tf.keras.losses.Loss):
    name = "LabelOptimLoss"

    def __init__(self,
                 alpha=1.2,
                 beta=0.8,
                 prior_dist=tf.constant([0.1] * 10),
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        super(LabelOptimLoss, self).__init__(reduction, self.name if name is None else name)
        self.alpha = alpha
        self.beta = beta
        self.prior_dist = prior_dist

    @tf.function
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss_c = KLDivergence()(y_true, y_pred)

        pred_dist = tf.keras.backend.mean(y_pred, axis=0)
        loss_p = self.alpha * KLDivergence()(self.prior_dist, pred_dist)

        loss_e = self.beta * tf.keras.backend.mean(tf.math.multiply(y_pred, tf.math.log(y_pred)))

        loss = tf.math.add(loss_c, loss_p)
        loss = tf.math.subtract(loss, loss_e)
        return loss
