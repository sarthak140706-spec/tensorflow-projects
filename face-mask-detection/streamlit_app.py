"""
Face Mask Detection - Streamlit App
"""

import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src import config


# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_trained_model():
    model = load_model(config.TRAINED_MODEL_PATH)
    return model


model = load_trained_model()


# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = config.CLASS_NAMES


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image):
    """
    Convert uploaded image to model-ready format
    """

    img = load_img(
        image,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
    )

    img = img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


# ==========================================================
# Streamlit UI
# ==========================================================

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection App")
st.write("Upload an image and the model will detect whether the person is wearing a mask or not.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = preprocess_image(uploaded_file)

    # Prediction
    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    label = CLASS_NAMES[predicted_class]

    # Result
    st.subheader("Prediction Result")

    if label == "with_mask":
        st.success(f"✔ Person is wearing a mask")
    else:
        st.error(f"❌ Person is NOT wearing a mask")

    st.write(f"**Confidence:** {confidence * 100:.2f}%")