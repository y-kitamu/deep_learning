import argparse

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, Input, Softmax
from hydra.experimental import initialize, compose

from mlops import dataset
from mlops import constant
# from mlops import models


def train(cfg):
    train_ds = dataset.TrainDataGenerator(cfg["dataset"]).create_dataset()
    val_ds = dataset.TrainDataGenerator(cfg["dataset"], is_train=False).create_dataset()

    image_size = cfg["dataset"]["overall"]["output_image_size"]
    num_channel = cfg["dataset"]["overall"]["num_channel"]
    classes = len(cfg["dataset"]["flawless"]["classes"]) + len(cfg["dataset"]["flaws"]["classes"])

    resnet = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", classes=classes)
    top = tf.keras.Sequential()
    top.add(Conv2D(classes, kernel_size=2))
    top.add(GlobalAvgPool2D())
    input = Input(shape=(image_size, image_size, num_channel))
    x = resnet(input)
    x = top(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    optimizer = tf.keras.optimizers.get(cfg["train"]["optimizer"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    # model.fit(train_ds,
    #           epochs=1,
    #           validation_data=val_ds,
    #           steps_per_ecposh=cfg["train"]["steps_per_epoch"],
    #           validation_step=cfg["train"]["validation_step"])
    return model, train_ds, val_ds


if __name__ == "__main__":

    parser = argparse.ArgumentParser("MLOpsSampleTrain")
    parser.add_argument_group("--projectname",
                              "-p",
                              help="directory name in which hydra config file is searched.")
    args = parser.parse_args()

    result_dir = constant.RESULT_ROOT / args.projectname
    with initialize(config_path=result_dir):
        cfg = compose()
    train(cfg)
