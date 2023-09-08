import tensorflow as tf
from typing import Tuple


def define_model_mlp(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x), name="grayscale_convert_layer")(inputs)
    x = tf.keras.layers.Flatten(data_format="channels_last", name="flatten_layer")(x)
    x = tf.keras.layers.Dense(
        units=512,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dense_layer_1"
    )(x)
    x = tf.keras.layers.Dense(
        units=256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dense_layer_2"
    )(x)
    x = tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dense_layer_3"
    )(x)
    x = tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dense_layer_4"
    )(x)
    outputs = tf.keras.layers.Dense(
        units=10,
        activation="softmax",
        name="softmax_classifier"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="cifar10_classifier_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model
