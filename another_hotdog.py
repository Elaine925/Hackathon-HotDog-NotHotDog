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
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from PIL import Image
import io



# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of your CNN model
    image = image.resize((224, 224))  # Assuming your model expects 224x224 input
    # Convert the image to a numpy array
    image_array = img_to_array(image)
    # Normalize the pixel values to be in the range [0, 1]
    image_array = image_array / 255.0
    # Expand the dimensions to create a batch of size 1
    image_array = np.expand_dims(image_array, axis=0)
    return image_array



st.title("Hot Dog or *NOT* Hot Dog :hotdog:")
st.subheader(':blue[Delivered by *GIRLPOWER*] \n Nkechi Goodacre, Rajashree Choudhary, Elaine Chen :female-student:')



uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])

if uploaded_image is not None:
    image_b = uploaded_image.read()
    image_b = Image.open(io.BytesIO(image_b))
    st.image(uploaded_image)
    # image = Image.open(uploaded_image)

    preprocessed_image = preprocess_image(image_b)

    # base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # base_model.trainable = False  # Freeze layers
    model = tf.keras.models.load_model('./Models/my_model.keras', 
                                        custom_objects=None,
                                        compile=True)
    pred = model.predict(preprocessed_image)[0][0]

    
    if (pred < 0.5):
        st.write("It is NOT a Hot Dog")
    else: 
        st.write("It is a Hot Dog!")


    # Make predictions
    # if st.button("Classify"):
    #     predictions = predict(image, model)
    #     st.write("Predictions:")
    #     st.write(pred)