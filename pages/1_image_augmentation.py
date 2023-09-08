import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from demo import load_test_data, decode_labels
from preprocess import apply_image_augmentation

"""
In practice, it is common to augment image dataset by applying various operations like rotation, hue or brightness
change for robustness. In this vein, models introduced in this project are trained with images where series of
augmentation operations including random flip(in any direction; left, right, up, down) and hue, brightness shift applied.
For example, original images are randomly transformed as following.  
"""

images, labels = load_test_data()
names = decode_labels(labels)
image_indices = np.random.choice(np.arange(images.shape[0]), 4)
sampled_images = images[image_indices] / 255.
corresponding_names = names[image_indices]
augmented_images, _ = apply_image_augmentation(sampled_images, corresponding_names)
augmented_images = augmented_images.numpy()
original_image_col, augmented_image_col = st.columns(spec=[0.5, 0.5])

with original_image_col:
    st.caption("Original images")
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(3, 3)
    for idx in range(4):
        r, c = divmod(idx, 2)
        ax[r][c].imshow(sampled_images[idx])
        ax[r][c].set_xticks([])
        ax[r][c].set_yticks([])
        ax[r][c].set_title(corresponding_names[idx])
    st.pyplot(fig)

with augmented_image_col:
    st.caption("Augmented images")
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(3, 3)
    for idx in range(4):
        r, c = divmod(idx, 2)
        ax[r][c].imshow(augmented_images[idx])
        ax[r][c].set_xticks([])
        ax[r][c].set_yticks([])
        ax[r][c].set_title(corresponding_names[idx])
    st.pyplot(fig)
