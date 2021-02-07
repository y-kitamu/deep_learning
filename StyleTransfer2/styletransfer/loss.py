import tensorflow as tf


def total_variation_loss(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))


def gram_matrix(input_tensor):
    """gram行列の計算
    Args:
        input_tensor (tf.Tensor) : 4D tensor of [Batch, Height, Width, Channel]
    Return:
        tf.Tensor : 3D Gram matrix tensor of [Batch, Num channel, Num channel]
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = input_tensor.shape
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def get_style_content_loss(style_weights=1e-2, content_weights=1e4):
    """
    """

    def style_content_loss(outputs, style_target, content_target):
        """スタイル変換のロス (style loss + content loss) の計算
        Args:
            outputs (dict)                     : keyに"style"と"content"をもつdictionary.
                (StyleContentModelの__call__のreturn value)
                "style"にはCNNの各layerのgram行列の計算結果、"content"にはCNNの各layerの出力が含まれている
            style_target (list of tf.Tensor)   :
                style image's "style" output of StyleContentModel.__call__
            content_target (list of tf.Tensor) :
                content image's "content" output of StyleContentModel.__call__
        Return:
            loss (tf.Tensor) :
        """
        num_style_layers = len(style_target)
        losses = []
        if num_style_layers > 0:
            style_loss = tf.add_n([
                tf.reduce_mean((outputs["style"][name] - style_target[name])**2)
                for name in style_target.keys()
            ])
            style_loss *= style_weights / num_style_layers
            losses.append(style_loss)

        num_content_layers = len(content_target)
        if num_content_layers > 0:
            content_loss = tf.add_n([
                tf.reduce_mean((outputs["content"][name] - content_target[name])**2)
                for name in content_target.keys()
            ])
            content_loss *= content_weights / num_content_layers
            losses.append(content_loss)

        return tf.add_n(losses)

    return style_content_loss
