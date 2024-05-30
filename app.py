import os
# os.system('pip install requirements.txt')
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import cv2

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    x = np.repeat(x, 3, axis=-1)
    return x

def get_slices(self, im_nii_path):
    ims = []
    nii_im_data = self.read_nii(im_nii_path)
    for idx, (im) in enumerate(nii_im_data):
        ims.append(im)
    return ims


def predict_and_save(input_path: str, output_path: str):
    input_slices = nib.load(input_path).get_fdata()
    answer = []
    # st.write(input_slices.shape)
    for slice_idx in range(input_slices.shape[2]):
        image = input_slices[:,:,slice_idx]
        image = np.expand_dims(image,axis=-1)
        image = np.repeat(image, 3, axis=-1)
        # st.write(image.shape)
        prediction = model.predict(np.expand_dims(image,axis=0)).squeeze()
        
        prediction = np.argmax(prediction, axis=-1).astype(np.int32)

        prediction[prediction == 1] = 128
        prediction[prediction == 2] = 255 
        
        answer.append(prediction)

    output_array = np.array(answer).transpose(1, 2, 0)
    output_image = nib.Nifti1Image(output_array, np.eye(4))
    nib.save(output_image, output_path)
    

model = tf.keras.models.load_model('models/final_resunet_pp_andreydataset_focaltversky_alpha0.6_beta0.4_gamma_1.hdf5', compile=False)

st.header("Распознавание болезней печени")

uploaded_file = st.file_uploader("Загрузите JPG-файл")

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

if uploaded_file is not None:
    image_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл успешно загружен!")

    filename = uploaded_file.name

    if filename.endswith(".nii"):
        predict_and_save(image_path, 'uploaded_files/prediction.nii')

        prediction = nib.load('uploaded_files/prediction.nii').get_fdata()

        num_slices = prediction.shape[2]

        st.title("CT Scan Viewer")
        st.sidebar.header("Navigation")

        slice_num = st.slider("Slice Number", 0, num_slices - 1, 0)

        st.write(f"Displaying slice {slice_num}")
        plt.figure(figsize=(5, 5))
        plt.imshow(prediction[:, :, slice_num], cmap='gray')
        plt.axis('off')
        st.pyplot(plt)

    elif filename.endswith(".jpg") or filename.endswith(".png"):

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

        prediction_image_path = image_path.replace(filename.split('.')[1], '_prediction.png')
        prediction_image = Image.fromarray((prediction).astype(np.uint8))
        prediction_image.save(prediction_image_path)

        with open(prediction_image_path, "rb") as file:
            st.download_button(
                label="Скачать файл",
                data=file,
                file_name=uploaded_file.name.replace(filename.split('.')[1], '_prediction.png'),
                mime="image/png",
            )

        os.remove(image_path)
        os.remove(prediction_image_path)
    
    else:
        st.write('Wrong File Format')