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
from tensorflow.keras.layers import BatchNormalization

from PIL import Image



# Function to load and preprocess the image
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



st.title("Hot Dog or *NOT* Hot Dog :hotdog:")
st.subheader(':blue[Delivered by *GIRLPOWER*] \n Nkechi Goodacre, Rajashree Choudhary, Elaine Chen :female-student:')



uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])

if uploaded_image is not None:
    st.image(uploaded_image)


    image = Image.open(uploaded_image)
    preprocessed_image = preprocess_image(image)


    model = tf.keras.models.load_model('./yc_model.h5', 
                                        custom_objects=None,
                                        compile=True)
    pred = model.predict(preprocessed_image)[0][0]

    
    # if (pred < 0.5):
    #     pred_image = 'img/itshotdog.png'
    # else: 
    #     pred_image = 'img/nothotdog.png'


    # Make predictions
    if st.button("Classify"):
        predictions = predict(image, model)
        st.write("Predictions:")
        st.write(pred)