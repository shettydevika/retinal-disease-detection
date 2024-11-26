import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define the path to the trained model
MODEL_PATH = 'oct_model.h5'

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the file exists in the project directory.")
    st.stop()  # Stop the app if the model is missing

# Load the trained model
model = load_model(MODEL_PATH)

# Class Labels (Update these based on your dataset)
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Helper function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Match model input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function
def predict(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class] * 100
    return CLASS_NAMES[predicted_class], confidence

# Streamlit Interface
st.title("OCT Image Classification")
st.write("Upload an OCT image to classify it as 'Healthy' or 'Infected'.")

# File uploader
uploaded_file = st.file_uploader("Choose an OCT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "uploaded_image.png"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(temp_file_path, caption="Uploaded Image", use_container_width=True)

    # Perform prediction
    st.write("Classifying...")
    try:
        predicted_class, confidence = predict(temp_file_path)
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

    # Clean up the temporary file
    os.remove(temp_file_path)
