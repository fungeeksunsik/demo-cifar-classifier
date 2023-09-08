import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import preprocess
import matplotlib.pyplot as plt

from typing import Tuple
from main import local_dir


@st.cache_resource
def load_pretrained_mlp_model() -> tf.keras.Model:
    model_dir = local_dir.joinpath("mlp")
    return tf.keras.models.load_model(str(model_dir.joinpath("model")))


@st.cache_resource
def load_pretrained_cnn_model() -> tf.keras.Model:
    model_dir = local_dir.joinpath("cnn")
    return tf.keras.models.load_model(str(model_dir.joinpath("model")))


@st.cache_resource
def load_pretrained_resnet_model() -> tf.keras.Model:
    model_dir = local_dir.joinpath("resnet")
    return tf.keras.models.load_model(str(model_dir.joinpath("model")))


@st.cache_data
def load_mlp_train_history() -> pd.DataFrame:
    model_dir = local_dir.joinpath("mlp")
    return pd.read_csv(model_dir.joinpath("training_log.csv"))


@st.cache_data
def load_cnn_train_history() -> pd.DataFrame:
    model_dir = local_dir.joinpath("cnn")
    return pd.read_csv(model_dir.joinpath("training_log.csv"))


@st.cache_data
def load_resnet_train_history() -> pd.DataFrame:
    model_dir = local_dir.joinpath("resnet")
    return pd.read_csv(model_dir.joinpath("training_log.csv"))


def load_test_data() -> Tuple[np.array, np.array]:
    _, (images, labels) = tf.keras.datasets.cifar10.load_data()
    return images, labels


def decode_labels(labels: np.array) -> np.array:
    labels = labels.flatten()
    label_decoder = preprocess.get_label_decoder()
    names = list(map(lambda x: label_decoder[x], labels))
    return np.array(names)


tf.config.set_visible_devices([], 'GPU')  # prevent 'CHECK failed: target + size == res' error in M2 mac
mlp_model = load_pretrained_mlp_model()
cnn_model = load_pretrained_cnn_model()
resnet_model = load_pretrained_resnet_model()
images, labels = load_test_data()
label_decoder = preprocess.get_label_decoder()
labels = labels.flatten()
result_indices = list(label_decoder.values())
st.title("Pretrained Image Classifiers Demo")

if st.sidebar.button("Refresh"):
    random_index = np.random.randint(0, images.shape[0])
    sample_image = images[random_index][np.newaxis, :, :, :] / 255.
    sample_image_label = labels[random_index]
    text_column, sample_image_column = st.columns(spec=[0.85, 0.15])
    with text_column:
        """
        As displayed example, each image in CIFAR10 dataset consists of 32x32 shaped RGB image with corresponding label 
        of the image. Each of classifier will pass the image into the model to calculate softmax values for each label.
        Decision rule is to pick label whose softmax value is highest compared to others.
        """
    with sample_image_column:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(images[random_index])
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        st.caption(f"label: {label_decoder[sample_image_label]}")

    st.header("MLP Model Prediction Result")
    mlp_scores = mlp_model.predict(sample_image, verbose=0)[0]
    mlp_prediction = np.argmax(mlp_scores)
    predicted_label = label_decoder[int(mlp_prediction)]
    if mlp_prediction == sample_image_label:
        st.success(f"Prediction result of MLP model is correct(label: {predicted_label})", icon="✅")
    else:
        st.warning(f"MLP model failed to give correct prediction(label: {predicted_label})", icon="⚠️")
    st.bar_chart(pd.DataFrame({"softmax scores(MLP)": mlp_scores}, index=result_indices))

    st.header("CNN Model Prediction Result")
    cnn_scores = cnn_model.predict(sample_image, verbose=0)[0]
    cnn_prediction = np.argmax(cnn_scores)
    predicted_label = label_decoder[int(cnn_prediction)]
    if cnn_prediction == sample_image_label:
        st.success(f"Prediction result of CNN model is correct(label: {predicted_label})", icon="✅")
    else:
        st.warning(f"CNN model failed to give correct prediction(label: {predicted_label})", icon="⚠️")
    st.bar_chart(pd.DataFrame({"softmax scores(CNN)": cnn_scores}, index=result_indices))

    st.header("ResNet Model Prediction Result")
    resnet_scores = resnet_model.predict(sample_image, verbose=0)[0]
    resnet_prediction = np.argmax(resnet_scores)
    predicted_label = label_decoder[int(resnet_prediction)]
    if resnet_prediction == sample_image_label:
        st.success(f"Prediction result of ResNet model is correct(label: {predicted_label})", icon="✅")
    else:
        st.warning(f"ResNet model failed to give correct prediction(label: {predicted_label})", icon="⚠️")
    st.bar_chart(pd.DataFrame({"softmax scores(ResNet)": resnet_scores}, index=result_indices))
else:
    st.text("Click `Refresh` button on the sidebar")
