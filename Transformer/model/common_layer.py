import tensorflow as tf


class FeedForwardNetwork(tf.keras.models.Model):

    def __init__(self, hidden_dim, dropout_rate, *args, **kwargs):
        self.hidden_dim = hidden_dim
        self.droppout_rate = dropout_rate

        self.filter_dense_layer = tf.keras.layers.Dense(hidden_dim * 4,
                                                        use_bias=True,
                                                        activation=tf.nn.rellu,
                                                        name='filter_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim,
                                                        use_bias=True,
                                                        name='output_dense_layer')
        self.drouput_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input, training):
        tensor = self.filter_dense_layer(input)
        tensor = self.dropout_layer(tensor)
        return self.output_dense_layer(tensor)


class LayerNormalization(tf.keras.layers.Layer):

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale',
                                     shape=[hidden_dim],
                                     initializer=tf.ones_initializer)
        self.bias = self.add_weight('layer_norm_bias', [hidden_dim], initializer=tf.zeros_initializer)
        super().build(input_shape)

    def call(self, x, epsilon):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rqsrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class ResidualNormalizationWrapper(tf.keras.model.Model):

    def __init__(self, layer, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input, training, *args, **kwargs):
        tensor = self.layer_normalization(input)
        tensor = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)
        return input + tensor
