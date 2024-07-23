import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import pickle

# Load the models
model1 = load_model("transfer_learning_model.h5")
models = {"Model 1": model1}

class_names = [
    "adho mukha svanasana",
    "adho mukha vriksasana",
    "agnistambhasana",
    "ananda balasana",
    "anantasana",
    "anjaneyasana",
    "ardha bhekasana",
    "ardha chandrasana",
    "ardha matsyendrasana",
    "ardha pincha mayurasana",
    "ardha uttanasana",
    "ashtanga namaskara",
    "astavakrasana",
    "baddha konasana",
    "bakasana",
    "balasana",
    "bhairavasana",
    "bharadvajasana i",
    "bhekasana",
    "bhujangasana",
    "bhujapidasana",
    "bitilasana",
    "camatkarasana",
    "chakravakasana",
    "chaturanga dandasana",
    "dandasana",
    "dhanurasana",
    "durvasasana",
    "dwi pada viparita dandasana",
    "eka pada koundinyanasana i",
    "eka pada koundinyanasana ii",
    "eka pada rajakapotasana",
    "eka pada rajakapotasana ii",
    "ganda bherundasana",
    "garbha pindasana",
    "garudasana",
    "gomukhasana",
    "halasana",
    "hanumanasana",
    "janu sirsasana",
    "kapotasana",
    "krounchasana",
    "kurmasana",
    "lolasana",
    "makara adho mukha svanasana",
    "makarasana",
    "malasana",
    "marichyasana i",
    "marichyasana iii",
    "marjaryasana",
    "matsyasana",
    "mayurasana",
    "natarajasana",
    "padangusthasana",
    "padmasana",
    "parighasana",
    "paripurna navasana",
    "parivrtta janu sirsasana",
    "parivrtta parsvakonasana",
    "parivrtta trikonasana",
    "parsva bakasana",
    "parsvottanasana",
    "pasasana",
    "paschimottanasana",
    "phalakasana",
    "pincha mayurasana",
    "prasarita padottanasana",
    "purvottanasana",
    "salabhasana",
    "salamba bhujangasana",
    "salamba sarvangasana",
    "salamba sirsasana",
    "savasana",
    "setu bandha sarvangasana",
    "simhasana",
    "sukhasana",
    "supta baddha konasana",
    "supta matsyendrasana",
    "supta padangusthasana",
    "supta virasana",
    "tadasana",
    "tittibhasana",
    "tolasana",
    "tulasana",
    "upavistha konasana",
    "urdhva dhanurasana",
    "urdhva hastasana",
    "urdhva mukha svanasana",
    "urdhva prasarita eka padasana",
    "ustrasana",
    "utkatasana",
    "uttana shishosana",
    "uttanasana",
    "utthita ashwa sanchalanasana",
    "utthita hasta padangustasana",
    "utthita parsvakonasana",
    "utthita trikonasana",
    "vajrasana",
    "vasisthasana",
    "viparita karani",
    "virabhadrasana i",
    "virabhadrasana ii",
    "virabhadrasana iii",
    "virasana",
    "vriksasana",
    "vrischikasana",
    "yoganidrasana",
]

# Function to load and preprocess image
def prepare_image(image_path, target_size=(256, 256)):
    img = load_img(
        image_path, target_size=target_size
    )  # Adjust target size to match your model's input
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
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Convert class index to class label
    predicted_class_label = class_names[predicted_class_index]
    
    # Display the image and prediction
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Class: {predicted_class_label} using {selected_model_name}")

    # Remove the temporary file
    os.remove("temp.jpg")
