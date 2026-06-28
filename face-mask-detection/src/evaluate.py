"""
Evaluate Face Mask Detection Model
"""

import numpy as np
import matplotlib.pyplot as plt

import config

from data_loader import load_data

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ==========================================================
# Load Dataset
# ==========================================================

_, _, _, _, test_images, test_labels = load_data(
    config.DATA_DIR,
    test_size=config.TEST_SIZE,
    val_size=config.VAL_SIZE,
    random_state=config.RANDOM_STATE
)

# ==========================================================
# Encode Labels
# ==========================================================

class_mapping = {
    "with_mask": 0,
    "without_mask": 1
}

test_labels = np.array(
    [class_mapping[label] for label in test_labels]
)

test_labels = to_categorical(
    test_labels,
    num_classes=config.NUM_CLASSES
)

# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_images(image_paths):

    images = []

    for path in image_paths:

        img = load_img(
            path,
            target_size=(
                config.IMG_HEIGHT,
                config.IMG_WIDTH
            )
        )

        img = img_to_array(img)

        img = preprocess_input(img)

        images.append(img)

    return np.array(images, dtype="float32")


print("Loading Test Images...")

test_data = preprocess_images(test_images)

print("Done.\n")

# ==========================================================
# Load Model
# ==========================================================

model = load_model(config.TRAINED_MODEL_PATH)

print("Model Loaded Successfully.\n")

# ==========================================================
# Evaluate
# ==========================================================

loss, accuracy = model.evaluate(
    test_data,
    test_labels,
    batch_size=config.BATCH_SIZE,
    verbose=1
)

print("=" * 50)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print("=" * 50)

# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(
    test_data,
    batch_size=config.BATCH_SIZE
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = np.argmax(
    test_labels,
    axis=1
)

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=config.CLASS_NAMES
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=config.CLASS_NAMES
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    f"{config.PLOTS_DIR}/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved successfully.")