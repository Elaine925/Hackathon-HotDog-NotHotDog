import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load your trained model
model = load_model('C:/Users/nkech/Documents/GeneralAssembly/Projects/project-4/Hackathon-HotDog-NotHotDog/MobileNet.h5')

st.title('Hot Dog Not Hot Dog Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(150, 150))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0  # Model expects an array of images

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Predict the class of the image
    prediction = model.predict(image_array)
    if prediction[0] < 0.5:
        st.write("It's not a hot dog!")
    else:
        st.write("It's a hot dog!")

T