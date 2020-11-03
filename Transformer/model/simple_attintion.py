import tensorflow as tf


class SimpleAttention(tf.keras.models.Model):

    def __init__(self, depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth

        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name="q_dense_layer")
        self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name="k_dense_layer")
        self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name="v_dense_layer")
        self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name="output_dense_layer")

    def call(self, input, memory, attention_mask):
        q = self.q_dense_layer(input)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)

        q *= self.depth ** -0.5
        logit = tf.matmul(q, k, transpose_b=True)
        logit += tf.to_float(attention_mask) * input.dtype.min
        attention_weight = tf.nn.softmax(logit, name="attention_weight")

        attention_output = tf.matmul(attention_weight, v)
        return self.output_dense_layer(attention_output)
