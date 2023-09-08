import streamlit as st
from demo import (
    load_mlp_train_history,
    load_cnn_train_history,
    load_resnet_train_history
)

mlp_history = load_mlp_train_history()
cnn_history = load_cnn_train_history()
resnet_history = load_resnet_train_history()

st.subheader("Plots of loss and metrics over training epoch")

mlp_history_col, cnn_history_col, resnet_history_col = st.columns(spec=[0.3, 0.35, 0.35])
with mlp_history_col:
    st.caption("Multilayer Perceptron(MLP)")
    loss_tab, acc_tab = st.tabs(["loss", "accuracy"])
    with loss_tab:
        st.line_chart(
            data=mlp_history[["loss", "val_loss", "epoch"]],
            x="epoch", y=["loss", "val_loss"]
        )
    with acc_tab:
        st.line_chart(
            data=mlp_history[["accuracy", "val_accuracy", "epoch"]],
            x="epoch", y=["accuracy", "val_accuracy"]
        )

with cnn_history_col:
    st.caption("Convolutional Neural Network(CNN)")
    loss_tab, acc_tab = st.tabs(["loss", "accuracy"])
    with loss_tab:
        st.line_chart(
            data=cnn_history[["loss", "val_loss", "epoch"]],
            x="epoch", y=["loss", "val_loss"]
        )
    with acc_tab:
        st.line_chart(
            data=cnn_history[["accuracy", "val_accuracy", "epoch"]],
            x="epoch", y=["accuracy", "val_accuracy"]
        )

with resnet_history_col:
    st.caption("Residual Network(ResNet)")
    loss_tab, acc_tab = st.tabs(["loss", "accuracy"])
    with loss_tab:
        st.line_chart(
            data=resnet_history[["loss", "val_loss", "epoch"]],
            x="epoch", y=["loss", "val_loss"]
        )
    with acc_tab:
        st.line_chart(
            data=resnet_history[["accuracy", "val_accuracy", "epoch"]],
            x="epoch", y=["accuracy", "val_accuracy"]
        )
