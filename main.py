import logging
import pathlib
import sys
import preprocess
import models
import tensorflow as tf
from typer import Typer

app = Typer()

LOCAL_DIR = "/tmp/cifar10"
S3_URL_PREFIX = "http://grainpowder-archive.s3.amazonaws.com/github/demo-cifar10-classifier"

formatter = logging.Formatter(
    fmt="%(asctime)s : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

local_dir = pathlib.Path(LOCAL_DIR)
local_dir.mkdir(exist_ok=True, parents=True)


@app.command("remote")
def load_from_remote():
    logger.info("Download pretrained models with corresponding training log file")
    preprocess.download_pretrained_model(S3_URL_PREFIX, local_dir, "plain")
    preprocess.download_pretrained_model(S3_URL_PREFIX, local_dir, "augmented")

    logger.info("Unzip downloaded model file")
    preprocess.unzip_downloaded_model(local_dir, "plain")
    preprocess.unzip_downloaded_model(local_dir, "augmented")


@app.command("local")
def train_in_local():
    logger.info("Load CIFAR10 data from Keras dataset")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    logger.info("Train MLP classifier with augmented data")
    model_mlp = models.define_model_mlp(train_images[0].shape)
    model_mlp.fit(
        x=preprocess.get_tf_dataset(train_images, train_labels, True),
        epochs=30,
        verbose=2,
        callbacks=models.define_callbacks(local_dir, "mlp"),
        validation_data=preprocess.get_tf_dataset(test_images, test_labels, False),
    )

    logger.info("Train CNN classifier with augmented data")
    model_cnn = models.define_model_cnn(train_images[0].shape)
    model_cnn.fit(
        x=preprocess.get_tf_dataset(train_images, train_labels, True),
        epochs=30,
        verbose=2,
        callbacks=models.define_callbacks(local_dir, "cnn"),
        validation_data=preprocess.get_tf_dataset(test_images, test_labels, False),
    )

    logger.info("Train ResNet classifier with augmented data")
    model_resnet = models.define_model_resnet()
    model_resnet.fit(
        x=preprocess.get_tf_dataset(train_images, train_labels, True),
        epochs=30,
        verbose=2,
        callbacks=models.define_callbacks(local_dir, "resnet"),
        validation_data=preprocess.get_tf_dataset(test_images, test_labels, False),
    )

    logger.info("Load best performing weight checkpoint and save model")
    model_mlp.load_weights(filepath=f"{local_dir}/mlp/ckpt")
    model_mlp.save(filepath=f"{local_dir}/mlp/model")
    model_cnn.load_weights(filepath=f"{local_dir}/cnn/ckpt")
    model_cnn.save(filepath=f"{local_dir}/cnn/model")
    model_resnet.load_weights(filepath=f"{local_dir}/resnet/ckpt")
    model_resnet.save(filepath=f"{local_dir}/resnet/model")


if __name__ == "__main__":
    app()
