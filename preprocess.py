import requests
import pathlib
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from typing import Dict, Tuple


def download_pretrained_model(url_prefix: str, local_dir: pathlib.Path, model_type: str):
    """
    download pretrained model and corresponding training log from S3 bucket
    :param url_prefix: configured S3 path prefix
    :param local_dir: configured local directory
    :param model_type: one of 'mlp' or 'cnn'
    :return: None
    """
    model_dir = local_dir.joinpath(model_type)
    model_dir.mkdir(exist_ok=True, parents=True)
    for file_name in ("model.zip", "training_log.csv"):
        source_url = f"{url_prefix}/{model_type}/{file_name}"
        response = requests.get(source_url)
        with open(f"{model_dir}/{file_name}", "wb") as file:
            file.write(response.content)


def unzip_downloaded_model(local_dir: pathlib.Path, model_type: str):
    """
    when pretrained model is downloaded from S3 bucket, it is in .zip format so unzip it
    :param local_dir: configured local directory
    :param model_type: one of 'mlp' or 'cnn'
    :return: None
    """
    model_dir = local_dir.joinpath(model_type)
    with ZipFile(model_dir.joinpath("model.zip"), "r") as file:
        file.extractall(model_dir)


def get_label_decoder() -> Dict[int, str]:
    """
    Return a dictionary which maps integer label to its real name
    :return: dictionary of (index, name)
    """
    names = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]
    return dict(enumerate(names))


def apply_image_augmentation(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Define series of image augmentation operations to be applied to input image
    :param image: standardized image whose pixel values fall into [0, 1]
    :param label: corresponding label
    :return: tensor of augmented images with its label
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
    return image, label


def get_tf_dataset(images: np.array, labels: np.array, apply_augmentation: bool) -> tf.data.Dataset:
    """
    wrap dataset into tf.data.Dataset API to be iterated within training loop
    :param images: array of raw images
    :param labels: array of corresponding label
    :param apply_augmentation: whether image augmentation has to be applied
    :return: tensorflow dataset
    """
    if apply_augmentation:
        return (
            tf.data.Dataset.from_tensor_slices((images / 255, labels))
            .map(apply_image_augmentation)
            .shuffle(buffer_size=256)
            .batch(batch_size=64)
        )
    else:
        return (
            tf.data.Dataset.from_tensor_slices((images / 255, labels))
            .shuffle(buffer_size=256)
            .batch(batch_size=64)
        )
