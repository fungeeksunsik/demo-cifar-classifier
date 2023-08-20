import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_normalization_1 = tf.keras.layers.BatchNormalization(input_shape=input_shape)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), padding="SAME")
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), padding="SAME")

    def call(self, inputs, training=False):
        x = self.batch_normalization_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv2d_1(x)
        x = self.batch_normalization_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2d_2(x)
        return inputs + x


class FiltersChangeResidualBlock(tf.keras.layers.Layer):

    def __init__(self, out_filters, **kwargs):
        super(FiltersChangeResidualBlock, self).__init__(**kwargs)
        self.out_filters = out_filters

    def build(self, input_shape):
        self.batch_normalization_1 = tf.keras.layers.BatchNormalization(input_shape=input_shape)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), padding="SAME")
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=(3, 3), padding="SAME")
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=(1, 1))

    def call(self, inputs, training=False):
        x = self.batch_normalization_1(inputs, training)
        x = tf.nn.relu(x)
        x = self.conv2d_1(x)
        x = self.batch_normalization_2(x, training)
        x = tf.nn.relu(x)
        x = self.conv2d_2(x)
        return self.conv2d_3(inputs) + x


class ResNetModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ResNetModel, self).__init__(**kwargs)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=(2, 2))
        self.residual_block_1 = ResidualBlock()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2))
        self.residual_block_2 = ResidualBlock()
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))
        self.residual_block_3 = ResidualBlock()
        self.conv2d_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1))
        self.filters_change = FiltersChangeResidualBlock(out_filters=128)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense = tf.keras.layers.Dense(units=10, activation="softmax")

    @tf.function
    def call(self, inputs: tf.Tensor, training=False):
        x = self.conv2d_1(inputs)
        x = self.residual_block_1(x, training)
        x = self.conv2d_2(x)
        x = self.residual_block_2(x, training)
        x = self.conv2d_3(x)
        x = self.residual_block_3(x, training)
        x = self.conv2d_4(x)
        x = self.filters_change(x, training)
        x = self.flatten(x)
        return self.dense(x)


def define_model_resnet() -> ResNetModel:
    model = ResNetModel(name="cifar10_classifier_resnet")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model
