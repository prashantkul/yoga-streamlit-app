import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load the models
model1 = load_model('transfer_learning_model.h5')
models = {'Model 1': model1}


# Function to load and preprocess image
def prepare_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)  # Adjust target size to match your model's input
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img


st.title("Yoga Pose Classification")
st.write("Upload an image and select a model for classification.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Model selection
selected_model_name = st.selectbox("Select a model", list(models.keys()))
selected_model = models[selected_model_name]

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Prepare image and predict
    image = prepare_image("temp.jpg")
    predictions = selected_model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Display the image and prediction
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Class: {predicted_class} using {selected_model_name}")

    # Remove the temporary file
    os.remove("temp.jpg")
