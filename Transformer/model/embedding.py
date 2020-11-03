import math

import tensorflow as tf

PAD_ID = 0


class AddPositionalEncoding(tf.keras.layer.Layer):

    def call(self, inputs):
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        depth_counter = tf.range(depth)
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0), [max_length, 1])
        depth_matrix = tf.pow(100000.0, tf.case(depth_matrix / depth, fl_type))

        phase = tf.case(tf.range(depth) % 2, fl_type) * math.pi / 2
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [max_length, 1])

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1)[1, depth]), fl_type)

        positional_encoding = tf.sin(pos_matrix / depth_matrix * phase_matrix)
        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1])

        return inputs + positional_encoding


class TokenEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_dim, dtype=tf.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype_ = dtype

    def build(self, input_shape):
        self.lookup_table = self.add_variable(name='token_embedding',
                                              shape=[self.vocab_size, self.embedding_dim],
                                              dtype=self.dtype_,
                                              intializer=tf.random_normal_initializer(
                                                  0, self.embedding_dim**-0.5))
        super().build(input_shape)

    def call(self, input):
        mask = tf.to_float(tf.not_equal(input, PAD_ID))
        embedding = tf.nn.embedding_lookup(self.lookup_table, input)
        embedding *= tf.expand_dims(mask, -1)
        return embedding * self.embedding_dim**0.5
