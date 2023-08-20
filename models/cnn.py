import tensorflow as tf
from typing import Tuple


def define_model_cnn(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="SAME",
                name="convolution_1",
                input_shape=input_shape
            ),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="SAME",
                name="convolution_2"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"),
            tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
            tf.keras.layers.Dropout(0.5, name="dropout_1"),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="SAME",
                name="convolution_3"
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="SAME",
                name="convolution_4"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"),
            tf.keras.layers.BatchNormalization(name="batch_normalization_2"),
            tf.keras.layers.Dropout(0.5, name="dropout_2"),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                padding="SAME",
                name="convolution_5"
            ),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                padding="SAME",
                name="convolution_6"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_3"),
            tf.keras.layers.BatchNormalization(name="batch_normalization_3"),
            tf.keras.layers.Dropout(0.5, name="dropout_3"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="sigmoid"),
        ],
        name="cifar10_classifier_cnn"
    )
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model
