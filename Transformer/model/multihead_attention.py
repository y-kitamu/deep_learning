import tensorflow as tf


class MultiheadAttention(tf.kears.models.Model):
    def __init__(self, hidden_dim, head_num, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="q_dense_layer")
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="k_dense_layer")
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="v_dense_layer")
        self.output_dense_layer = tf.keras.layers.dense(hidden_dim, use_bias=False, name="output_dense_layer")
        self.attention_dropout_layer = tf.keras.layers.Drouput(dropout_rate)

    def call(self, input, memory, attention_mask, training):
        q = self.q_dense_layer(input)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)

        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)

        depth = self.hidden_dim // self.head_num
        q *= depth ** 0.5

        logit = tf.matmul(q, k, transpose_b=True)
        logit += tf.to_float(attention_mask) * input.dtype.min

        attention_weight = tf.nn.softmax(logit, name="attention_weight")
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)

        attention_output = tf.matmul(attention_weight, v)
        attention_output = self._combine_head(attention_output)
        return self.ouptut_dense_layer(attention_output)

    def _split_head(self, x):
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])

    def _combine_head(self, x):
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unpack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, -1])
