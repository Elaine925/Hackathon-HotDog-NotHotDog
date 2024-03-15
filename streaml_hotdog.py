import streamlit as st
from PIL import Image
import tensorflow 
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers

def load_model(model_path):
    model = tensorflow.keras.models.load_model(model_path)
    return model

def predict(image, model):
    # Resize image to match model input shape
    image = image.resize((224, 224))
    # Convert PIL image to numpy array
    image_array = tensorflow.keras.preprocessing.image.img_to_array(image)
    # Normalize pixel values
    image_array /= 255.0
    # Expand dimensions to match model input shape
    image_array = tensorflow.expand_dims(image_array, 0)
    # Predict class probabilities
    predictions = model.predict(image_array)
    return predictions

st.title("Hot Dog or *NOT* Hot Dog :hotdog:")
st.subheader(':blue[Delivered by *GIRLPOWER*] \n Nkechi Goodacre, Rajashree Choudhary, Elaine Chen :female-student:')

def image_uploader():
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return image
    else:
        st.write("Please upload a hotdog image file.")

# model_path = "./MobileNet.h5"  
model_path_net = './MobileNet.h5'
model = load_model(model_path_net)

uploaded_image = image_uploader()
if uploaded_image is not None:
    st.write("You can process the uploaded image here.")