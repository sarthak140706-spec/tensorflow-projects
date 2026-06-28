"""
Train Face Mask Detection Model
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import config

from data_loader import load_data
from augment import get_train_val_augmentations
from model import build_model

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam


# ==========================================================
# Load Dataset
# ==========================================================

train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(
    config.DATA_DIR,
    test_size=config.TEST_SIZE,
    val_size=config.VAL_SIZE,
    random_state=config.RANDOM_STATE
)

# ==========================================================
# Label Encoding
# ==========================================================

class_mapping = {
    "with_mask": 0,
    "without_mask": 1
}

train_labels = np.array(
    [class_mapping[label] for label in train_labels]
)

val_labels = np.array(
    [class_mapping[label] for label in val_labels]
)

test_labels = np.array(
    [class_mapping[label] for label in test_labels]
)

train_labels = to_categorical(train_labels, 2)
val_labels = to_categorical(val_labels, 2)
test_labels = to_categorical(test_labels, 2)

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


print("\nLoading Images...\n")

train_data = preprocess_images(train_images)
val_data = preprocess_images(val_images)
test_data = preprocess_images(test_images)

print("Images Loaded Successfully.\n")

# ==========================================================
# Data Augmentation
# ==========================================================

train_datagen, val_datagen, _ = get_train_val_augmentations()

# ==========================================================
# Build Model
# ==========================================================

model = build_model(
    config.IMG_HEIGHT,
    config.IMG_WIDTH,
    config.IMG_CHANNELS,
    config.NUM_CLASSES,
    config.USE_TRANSFER_LEARNING
)

optimizer = Adam(
    learning_rate=config.LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================================
# Callbacks
# ==========================================================

checkpoint = ModelCheckpoint(
    config.TRAINED_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

# ==========================================================
# Train
# ==========================================================

history = model.fit(

    train_datagen.flow(
        train_data,
        train_labels,
        batch_size=config.BATCH_SIZE
    ),

    validation_data=val_datagen.flow(
        val_data,
        val_labels,
        batch_size=config.BATCH_SIZE
    ),

    epochs=config.EPOCHS,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]

)

# ==========================================================
# Save Final Model
# ==========================================================

model.save(config.TRAINED_MODEL_PATH)

print("\nModel Saved Successfully.")
print(config.TRAINED_MODEL_PATH)

# ==========================================================
# Training Graphs
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    os.path.join(
        config.PLOTS_DIR,
        "accuracy.png"
    )
)

plt.close()

plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig(
    os.path.join(
        config.PLOTS_DIR,
        "loss.png"
    )
)

plt.close()

print("\nTraining Complete!")
print("Graphs saved in plots/")