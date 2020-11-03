import tensorflow as tf

from . import multihead_attention


class SelfAttention(multihead_attention.MultiheadAttention):

    def call(self, input, attention_mask, training):
        return super().call(input, input, attention_mask, training)
