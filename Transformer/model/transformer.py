import tensorflow as tf

from .selfattention import SelfAttention
from .common_layer import FeedForwardNetwork, ResidualNormalizationWrapper, LayerNormalization
from .multihead_attention import MultiheadAttention
from .embedding import TokenEmbedding, AddPositionalEncoding

PAD_ID = 0


class Encoder(tf.keras.models.Model):

    def __init__(self, vocab_size, hopping_num, head_num, hidden_dim, dropout_rate, max_length, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hiddeen_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_positional_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Droout(dropout_rate)

        self.attention_block_list = []
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer,
                                             dropout_rate,
                                             name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper')
            ])
        self.output_normalization = LayerNormalization()

    def call(self, input, self_attention_mask, training):
        embedded_input = self.token_embedding(input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f"hopping_{i}"):
                query = attention_layer(query, attention_mask=self.attention_mask, training=training)
                query = ffn_layer(query, training=training)
        return self.output_normalization(query)


class Decoder(tf.keras.model.Model):

    def __init__(self, vocab_size, hopping_num, head_num, hidden_dim, dropout_rate, max_length, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layer.Dropout(dropout_rate)

        self.attention_block_list = []
        for _ in range(hopping_num):
            self_attention_layer = SelfAttention(hidden_dim,
                                                 head_num,
                                                 dropout_rate,
                                                 name="self_attention")
            enc_dec_attention_layer = MultiheadAttention(hidden_dim,
                                                         head_num,
                                                         dropout_rate,
                                                         name='encdec_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(self_attention_layer, dropout_rate, name="self_attention"),
                ResidualNormalizationWrapper(enc_dec_attention_layer,
                                             dropout_rate,
                                             name="enc_dec_attention_layer"),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_layer')
            ])
            self.output_normalization = LayerNormalization()
            self.output_dense_layer = tf.keras.layers.Dense(vocab_size, use_bias=False)

    def call(self, input, encoder_output, self_attention_mask, enc_dec_attention_mask, training):
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = self_attention_layer(query,
                                             attention_mask=self_attention_mask,
                                             training=training)
                query = enc_dec_attention_layer(query,
                                                memory=encoder_output,
                                                attention_mask=enc_dec_attention_mask,
                                                training=training)
                query = ffn_layer(query, training=training)
        query = self.output_normalization(query)
        return self.output_dense_layer(query)


class Transformer(tf.keras.models.Model):

    def __init__(self, vocab_size, hopping_num, head_num, hidden_dim, dropout_rate, max_length, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        self.encoder = Encoder(vocab_size=vocab_size,
                               hopping_num=hopping_num,
                               head_num=head_num,
                               hidden_dim=hidden_dim,
                               dropout_rate=dropout_rate,
                               max_length=max_length)
        self.decoder = Decoder(vocab_size=vocab_size,
                               hopping_num=hopping_num,
                               head_num=head_num,
                               hidden_dim=hidden_dim,
                               dropout_rate=dropout_rate,
                               max_length=max_length)

    def call(self, encoder_input, decoder_input, training):
        enc_attention_mask = self._create_enc_attention_mask(encoder_input)
        dec_self_attention_mask = self._create_dec_self_attention_mask(decoder_input)

        encoder_output = self.encoder(encoder_input,
                                      self_attention_mask=enc_attention_mask,
                                      training=training)
        decoder_output = self.decoder(decoder_input,
                                      encoder_output,
                                      self_attention_mask=dec_self_attention_mask,
                                      enc_dec_attention_mask=enc_attention_mask,
                                      training=training)
        return decoder_output

    def _create_enc_attention_mask(self, encoder_input):
        with tf.name_scope('enc_attention_mask'):
            batch_size, length = tf.unstack(tf.shape(encoder_input))
            pad_array = tf.equal(encoder_input, PAD_ID)
            return tf.reshape(pad_array, [batch_size, 1, 1, length])

    def _create_enc_self_attention_mask(self, decoder_input):
        with tf.name_scope('dec_self_attention_mask'):
            batch_size, length = tf.unstack(tf.shape(decoder_input))
            pad_array = tf.equal(decoder_input, PAD_ID)

            autoregression_array = tf.logical_not(
                tf.linalg.band_part(tf.ones([length, length], dtype=tf.bool) - 1, 0))
            autoregression_array = tf.reshape(autoregression_array, [1, 1, length, length])
            return tf.logical_or(pad_array, autoregression_array)
