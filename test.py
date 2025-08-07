import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


H5_MODEL_PATH = "eyediseasemodel.tflite"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(H5_MODEL_PATH)

model = load_model()


CLASS_NAMES = [
    'Cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'
]


def preprocess_image(image):
    """Convert image to RGB, resize, normalize, and prepare for model input."""
    img = image.convert("RGB")
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect its disease.")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)

    # Get prediction result
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    st.success(f"🩺 **Prediction:** {CLASS_NAMES[class_index]}")
    st.info(f"🔬 **Confidence:** {confidence:.2%}")
