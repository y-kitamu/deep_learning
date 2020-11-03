from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, BatchNormalization, ReLU, GlobalAvgPool2D, Add
from tensorflow.keras import Model, Sequential, Input


class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.bn1 = BatchNormalization()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class PreActConv2d(Layer):

    def __init__(self, n_channel, kernel_size, strides, activation=ReLU, bn_axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.bn = BatchNormalization(axis=bn_axis)
        self.activation = activation()
        self.conv = Conv2D(n_channel, kernel_size, strides, padding="same")

    def call(self, input_tensor, training=False, *args, **kwargs):
        x = self.bn(input_tensor, training=training)
        x = self.activation(x, training=training)
        return self.conv(x, training=training)


class PreActResBlock(Layer):

    def __init__(self, in_channel, out_channel, kernel_size=3, activation="relu", **kwargs):
        super(PreActResBlock, self).__init__(**kwargs)
        ratio = int(out_channel / in_channel)
        if in_channel != out_channel:
            self.projection = Conv2D(
                out_channel, kernel_size=ratio, strides=ratio, padding="same", name="projection")
        self.conv1 = PreActConv2d(
            out_channel, kernel_size, strides=ratio, name="conv1")
        self.conv2 = PreActConv2d(
            out_channel, kernel_size, strides=(1, 1), name="conv2")
        self.add = Add(name="add")

    def call(self, input_tensor, training=False, *args, **kwargs):
        # import pdb; pdb.set_trace()
        x = self.conv1(input_tensor, training=training)
        x = self.conv2(x, training=training)
        if hasattr(self, "projection"):
            input_tensor = self.projection(input_tensor, training=training)
        return self.add([x, input_tensor], training=training)


class PreActResNet32(Model):

    def __init__(self):
        super(PreActResNet32, self).__init__()
        self.conv = Conv2D(32, 3, strides=(1, 1), padding="same", name="conv")
        self.unit1 = self._make_unit(32, 32, 5, name="unit1")
        self.unit2 = self._make_unit(32, 64, 5, name="unit2")
        self.unit3 = self._make_unit(64, 128, 5, name="unit3")
        self.bn = BatchNormalization(axis=-1, name="bn")
        self.pool = GlobalAvgPool2D(data_format="channels_last", name="gap")
        self.dense = Dense(10, activation="softmax", name="dense")

    def _make_unit(self, in_channel, out_channel, n_layer, name):
        block = Sequential(name=name)
        block.add(PreActResBlock(in_channel, out_channel))
        for i in range(n_layer - 1):
            block.add(PreActResBlock(out_channel, out_channel))
        return block

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor, training=training)
        x = self.unit1(x, training=training)
        x = self.unit2(x, training=training)
        x = self.unit3(x, training=training)
        x = self.bn(x, training=training)
        x = self.pool(x, training=training)
        return self.dense(x, training=training)

    def summary(self):
        x = Input(shape=(32, 32, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()
