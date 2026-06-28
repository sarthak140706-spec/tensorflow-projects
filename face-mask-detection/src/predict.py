"""
Predict Face Mask from a Single Image
"""

import argparse
import numpy as np

import config

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================================
# Load Model
# ==========================================================

print("Loading model...")

model = load_model(config.TRAINED_MODEL_PATH)

print("Model loaded successfully.\n")

# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = config.CLASS_NAMES

# ==========================================================
# Image Preprocessing
# ==========================================================


def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.
    """

    img = load_img(
        image_path,
        target_size=(
            config.IMG_HEIGHT,
            config.IMG_WIDTH
        )
    )

    img = img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


# ==========================================================
# Prediction Function
# ==========================================================


def predict(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction))

    print("=" * 50)
    print(f"Image      : {image_path}")
    print(f"Prediction : {CLASS_NAMES[predicted_class]}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print("=" * 50)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Face Mask Detection Prediction"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    args = parser.parse_args()

    predict(args.image)