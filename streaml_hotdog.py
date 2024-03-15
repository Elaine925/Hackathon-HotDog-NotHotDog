import streamlit as st
from PIL import Image
import tensorflow
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


<<<<<<< HEAD
# Provide path to your pickled model file
model_path = "finalized_model.sav"  
with open(model_path, 'rb') as f:
    model = pickle.load(f)
=======
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

def load_model(model_path):
    model = tensorflow.keras.models.load_model(model_path)
    return model
>>>>>>> main

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



uploaded_image = image_uploader()
if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    if st.button("Classify"):
        predictions = predict(image, model)
        st.write("Predictions:")
        st.write(predictions)