# src/evaluate.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -------------------------------
# Step 1: Paths and Parameters
# -------------------------------
MODEL_PATH = "models/efficientnet_apple_finetuned.h5"  # or use initial model
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

# -------------------------------
# Step 2: Load Model
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


# -------------------------------
# Step 3: Define Prediction Function
# -------------------------------
def predict_image(img_path):
    """
    Predicts the class of a single image.

    Args:
        img_path (str): Path to the image file.

    Returns:
        predicted_class (str): Predicted class label.
        confidence (float): Probability of predicted class.
    """
    # Load and resize image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    confidence = preds[0][predicted_index]
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# -------------------------------
# Step 4: Test on New Images
# -------------------------------
test_images_dir = "data/test"  # Folder with new images to predict

if os.path.exists(test_images_dir):
    for img_file in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_file)
        predicted_class, confidence = predict_image(img_path)
        print(f"{img_file} --> Predicted: {predicted_class}, Confidence: {confidence:.2f}")
else:
    print(f"No test folder found at {test_images_dir}. Please add images to predict.")
