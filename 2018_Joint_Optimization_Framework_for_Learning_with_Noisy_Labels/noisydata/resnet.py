from collections import namedtuple

from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, BatchNormalization, ReLU, Input
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


resblock = namedtuple("resblock", ["filters", "layers", "kernel_size", "downsample"])


def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides)
    layer = ReLU()(layer)
    return layer


def bn_conv2d(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = BatchNormalization()(x)
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding="same",
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay))(layer)
    return layer


def bn_relu_conv2d(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = BatchNormalization()(x)
    layer = ReLU()(layer)
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding="same",
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay))(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay=.0, downsample=True):
    if downsample:
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride)
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1)
    out = layers.add([residual_x, residual])
    out = ReLU()(out)
    return out


def PreActResidualBlock(x, filters, kernel_size, weight_decay=.0, downsample=True):
    if downsample:
        residual_x = bn_conv2d(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = bn_relu_conv2d(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride)
    residual = bn_relu_conv2d(residual,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=1)
    out = layers.add([residual_x, residual])
    return out


def ResNet18(classes, input_shape, weight_decay=1e-4):
    input = Input(shape=input_shape)
    x = input
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # conv2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)

    # conv3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)

    # conv4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)

    # conv5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)

    x = AveragePooling2D(pool_size=(4, 4), padding="valid")(x)
    x = Flatten()(x)
    x = Dense(classes, activation="softmax")(x)
    model = Model(input, x, name="ResNet18")
    return model


def PreActResNet32(classes, input_shape=(32, 32, 3), weight_decay=1e-4):
    resblocks = [
        resblock(32, 5, 3, False),
        resblock(64, 5, 3, True),
        resblock(128, 5, 3, True)
    ]

    input = Input(shape=input_shape)
    x = input
    x = bn_conv2d(x, filters=32, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))
    # x = conv2d_bn(x, filters=32, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    for block in resblocks:
        for i in range(block.layers):
            downsample = block.downsample and i == 0
            x = PreActResidualBlock(x,
                                    filters=block.filters,
                                    kernel_size=(block.kernel_size, block.kernel_size),
                                    weight_decay=weight_decay,
                                    downsample=downsample)
            # x = ResidualBlock(x,
            #                   filters=block.filters,
            #                   kernel_size=(block.kernel_size, block.kernel_size),
            #                   weight_decay=weight_decay,
            #                   downsample=downsample)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(8, 8), padding="valid")(x)
    x = Flatten()(x)
    x = Dense(classes, activation="softmax")(x)
    model = Model(input, x, name="PreActResNet32")
    return model
