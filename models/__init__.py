import pathlib
import tensorflow as tf
from typing import List
from models.mlp import define_model_mlp
from models.cnn import define_model_cnn
from models.resnet import define_model_resnet

__all__ = [
    "define_callbacks",
    "define_model_mlp",
    "define_model_cnn",
    "define_model_resnet",
]


def define_callbacks(local_dir: pathlib.Path, prefix: str) -> List[tf.keras.callbacks.Callback]:
    model_dir = local_dir.joinpath(prefix)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_dir}/ckpt",
        save_weights_only=True,
        save_best_only=True,
        save_freq="epoch",
        monitor="val_accuracy",
        verbose=1,
    )
    csv_record_callback = tf.keras.callbacks.CSVLogger(
        filename=f"{model_dir}/training_log.csv"
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, mode="max"
    )
    return [checkpoint_callback, csv_record_callback, early_stopping_callback]
