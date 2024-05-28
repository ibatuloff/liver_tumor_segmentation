import os
os.system('pip install requirements.txt')
import streamlit as st
import tensorflow as tf
from liver_tumor_seg import read_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_slices(self, im_nii_path):
        ims = []
        nii_im_data = self.read_nii(im_nii_path)
        for idx, (im) in enumerate(nii_im_data):
            ims.append(im)
        return ims
    

model = tf.keras.models.load_model('models/final_resunet_pp.hdf5', compile=False)

st.header("Распознавание болезней печени")

uploaded_file = st.file_uploader("Загрузите JPG-файл")

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

if uploaded_file is not None:
    image_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл успешно загружен!")

    # filetype = uploaded_file.name

    if filetype.endswith(".nii"):
        ...
    elif filetype.endswith(".jpg") or filetype.endswith(".jpeg"):

        image = read_image(image_path)

        prediction = model.predict(np.expand_dims(image,axis=0)).squeeze()
        prediction = np.argmax(prediction, axis=-1).astype(np.int32)
        
        prediction[prediction == 1] = 128
        prediction[prediction == 2] = 255 

        # meanIoU = tf.keras.metrics.MeanIoU(num_classes = 3)
        # meanIoU.update_state()

        fig, ax = plt.subplots(1, 2, figsize=(15, 10))

        ax[0].imshow(image)
        ax[0].set_title('Оригинал')
        ax[0].axis('off')

        ax[1].imshow(prediction, cmap = 'bone')
        ax[1].set_title('Сегментированное изображение')
        ax[1].axis('off')

        st.pyplot(fig)

        prediction_image_path = image_path.replace('.jpg', '_prediction.png')
        prediction_image = Image.fromarray((prediction).astype(np.uint8))
        prediction_image.save(prediction_image_path)

        with open(prediction_image_path, "rb") as file:
            st.download_button(
                label="Скачать файл",
                data=file,
                file_name=uploaded_file.name.replace('.jpg', '_prediction.png'),
                mime="image/png",
            )

        os.remove(image_path)
        os.remove(prediction_image_path)