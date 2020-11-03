import tensorflow as tf


def entropy_loss(y_true, y_pred):
    return -tf.math.reduce_sum(y_pred * tf.math.log(y_pred)) / y_pred.shape[0]


class NoisyDataLoss(tf.keras.losses.Loss):

    def __init__(self, alpha, beta, prior_dist, *args, **kwargs):
        super().__init__(name="NoisyDataLoss", *args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.prior_dist = prior_dist

        self.class_kl_loss = tf.keras.losses.KLDivergence()
        self.softlabel_kl_loss = tf.keras.losses.KLDivergence()
        self.entropy_loss = entropy_loss

    def call(self, y_true, y_pred):
        class_loss = self.class_kl_loss(y_true, y_pred)

        mean_pred = tf.math.reduce_mean(y_pred, axis=0)
        sl_loss = self.softlabel_kl_loss(self.prior_dist, mean_pred)

        pd_loss = self.entropy_loss(y_true, y_pred)

        loss = class_loss + self.alpha * sl_loss + self.beta * pd_loss
        # print("\rloss : {:.3f} = {:.3f} + {:.3f} + {:.3f}".format(
        #     loss.numpy(), class_loss.numpy(), self.alpha * sl_loss.numpy(), self.beta * pd_loss.numpy()
        # ), end="")
        return loss
