import pickle
import streamlit as st
import tensorflow as tf

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array

from PIL import Image


# model_path = "./MobileNet.h5"
# model = tf.keras.models.load_model(model_path)

# Provide path to your pickled model file
# model_path = "../Hackupdated_mobilenet.sav"  
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)
def preprocess_image(image):
    # Resize the image to match the input size of your CNN model
    image = image.resize((256, 256))  # Assuming your model expects 224x224 input
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the pixel values to be in the range [0, 1]
    image = image / 255.0
    # Expand the dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# def predict(image, model):
#     # Resize image to match model input shape
#     image = image.resize((224, 224))
#     # Convert PIL image to numpy array
#     image_array = tf.keras.preprocessing.image.img_to_array(image)
#     # Normalize pixel values
#     image_array /= 255.0
#     # Expand dimensions to match model input shape
#     image_array = tf.expand_dims(image_array, 0)
#     # Predict class probabilities
#     predictions = model.predict(image_array)
#     return predictions

st.title("Hot Dog or *NOT* Hot Dog :hotdog:")
st.subheader(':blue[Delivered by *GIRLPOWER*] \n Nkechi Goodacre, Rajashree Choudhary, Elaine Chen :female-student:')



def image_uploader():
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        model = keras.models.load_model('./MobileNet.h5')
        preprocessed_image = preprocess_image(image)
        pred = model.predict(preprocessed_image)[0][0]

        return image
    else:
        st.write("Please upload a hotdog image file.")

uploaded_image = image_uploader()
if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed_image = preprocess_image(image)


    model = keras.models.load_model('./MobileNet.h5')
    pred = model.predict(preprocessed_image)[0][0]

    # Make predictions
    if st.button("Classify"):
        predictions = predict(image, model)
        st.write("Predictions:")
        st.write(pred)