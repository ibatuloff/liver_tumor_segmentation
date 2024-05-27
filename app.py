import streamlit as st
import tensorflow as tf
import os
from liver_tumor_seg import read_image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('models/final_resunet_pp.hdf5', compile=False)

st.header("Распознавание болезней печени")

uploaded_file = st.file_uploader("Загрузите JPG-файл", type=["jpg"])

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

if uploaded_file is not None:
    image_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл успешно загружен!")

    image = read_image(image_path)

    prediction = model.predict(np.expand_dims(image,axis=0)).squeeze()
    prediction = np.argmax(prediction, axis=-1).astype(np.int32)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    ax[0].imshow(image)
    ax[0].set_title('Оригинал')
    ax[0].axis('off')

    ax[1].imshow(prediction, cmap = 'bone')
    ax[1].set_title('Сегментированное изображение')
    ax[1].axis('off')

    st.pyplot(fig)