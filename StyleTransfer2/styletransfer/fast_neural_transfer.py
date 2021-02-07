"""implementation of https://arxiv.org/abs/1508.06576
[Gatys, L. et al. A Neural Algorithm of Artistic Style]
"""
import os
import argparse
from collections import namedtuple

import cv2
import tensorflow as tf
import tqdm

from styletransfer.utility import tensor_to_nparray, read_image_to_array
from styletransfer.loss import gram_matrix, get_style_content_loss, total_variation_loss

# スタイル変換に用いるモデルの設定
#     model_builder (callbale) : 関数呼び出ししたときの返り値が使用するモデル
#     prprocess (callable) : 前処理関数 (ex. tf.keras.applications.vgg19.preprocess_input)
#     content_layers (list of str) :
#     style_layers (list of str) :
BaseModelConfig = namedtuple("BaseModelConfig",
                             ["model_builder", "preprocess", "style_layers", "content_layers"])

VGG19_DEFAULT = BaseModelConfig(
    lambda: tf.keras.applications.VGG19(include_top=False, weights="imagenet"),
    tf.keras.applications.vgg19.preprocess_input,
    ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
    ["block5_conv2"],
)


def create_base_model(model_config=VGG19_DEFAULT):
    """スタイル変換用のベースモデル(outputがcontent layer + style layerのモデル)の作成
    Args:
        model_config (BaseModelConfig) :
    Return:
        model (tf.keras.Model) :
    """
    model = model_config.model_builder()
    model.trainable = False

    output_layers = model_config.style_layers + model_config.content_layers
    outputs = [model.get_layer(name).output for name in output_layers]
    model = tf.keras.Model([model.input], outputs)

    return model


def clip_0to1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentModel(tf.keras.models.Model):
    """
    """

    def __init__(self,
                 model_config=VGG19_DEFAULT,
                 loss_fun=get_style_content_loss(),
                 tv_loss_fun=lambda x: x,
                 optimizer=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)):
        super(StyleContentModel, self).__init__()

        self.model = create_base_model(model_config)
        self.model.trainable = False
        self.preprocess = model_config.preprocess
        self.style_layers = model_config.style_layers
        self.num_style_layers = len(self.style_layers)
        self.content_layers = model_config.content_layers

        self.loss_fun = loss_fun
        self.tv_loss_fun = tv_loss_fun
        self.optimizer = optimizer

        self.is_set_target = False

    def calc_target_style_and_content(self, style_image, content_image):
        """loss計算時に使用する、style画像のstyle featureとcontent画像のcontent featureを計算。
        train_stepを呼び出す前に、この関数を実行しておく
        Args:
            style_image (tf.Tensor) : 4D array of [Batch, Height, Width, Channel]
            content_image (tf.Tensor) : 4D array of [Batch, Height, Width, Channel]
        """
        self.style_target = self.__call__(style_image)["style"]
        self.content_target = self.__call__(content_image)["content"]
        self.is_set_target = True

    def __call__(self, inputs):
        """style feature, content featureの計算
        Args:
            inputs (tf.Variable) : target image tensor (4D). value is 0 to 1.
        Return:
            dict : style feature and content feature.
                key = {"style" and "content"}, value = list of tf.Tensor
        """
        inputs = inputs * 255.0
        preprocessed_input = self.preprocess(inputs)
        outputs = self.model(preprocessed_input)
        style_outputs = [gram_matrix(output) for output in outputs[:self.num_style_layers]]
        content_outputs = outputs[self.num_style_layers:]

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        content_dict = {
            content_name: value for content_name, value in zip(self.content_layers, content_outputs)
        }
        return {"style": style_dict, "content": content_dict}

    @tf.function
    def train_step(self, image):
        """
        Args:
            image (tf.Tensor) : 4D array
        """
        if not self.is_set_target:
            print("Target style and content is not set. Call `calc_target_style_and_content first.`")
            return
        with tf.GradientTape() as tape:
            outputs = self.__call__(image)
            loss = self.loss_fun(outputs, self.style_target, self.content_target)
            loss += self.tv_loss_fun(image)

        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(clip_0to1(image))


def run(style_image_path,
        content_image_path,
        output_image_path,
        model_config=VGG19_DEFAULT,
        epochs=10,
        steps_per_epochs=100,
        use_tv_loss=False):
    """スタイル変換の実行、画像の保存
    Args:
        content_image_path (str)       : Path to input content image
        style_image_path (str)         : Path to input style image
        output_image_path (str)        : Path to output image
        model_config (BaseModelConfig) :
        epochs (int)                   :
        steps_per_epochs (int)         :
        use_tv_loss (bool)             :
    Output:
        output image : Style transfered image saved to `output_image_path`
    Return:
        output_image (np.ndarray) : Output (style transferred) image
    """
    content_image = read_image_to_array(content_image_path)
    style_image = read_image_to_array(style_image_path)
    print("content image shape : {}".format(content_image.shape))
    print("style image shape : {}".format(style_image.shape))

    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)

    tv_loss_fun = lambda x: x
    if use_tv_loss:
        tv_loss_fun = total_variation_loss

    extractor = StyleContentModel(model_config=model_config, tv_loss_fun=tv_loss_fun)
    extractor.calc_target_style_and_content(style_image, content_image)

    image = tf.Variable(content_image)
    for epoch in tqdm.tqdm(range(epochs)):
        for step in range(steps_per_epochs):
            extractor.train_step(image)

    output_image = tensor_to_nparray(image)
    cv2.imwrite(output_image_path, output_image)
    print("Successfully output image!")

    return output_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Fast Neural Style Transfer")
    parser.add_argument("--epoch")
    parser.add_argument("--style_image", "-s", help="Path to style image")
    parser.add_argument("--content_image", "-c", help="Path to content image")
    parser.add_argument("--output_path", "-o", help="Path to output image is saved")

    args = parser.parse_args()

    run(args.style_image, args.content_image, args.output_path)
