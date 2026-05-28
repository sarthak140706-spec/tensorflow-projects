# src/train.py

import tensorflow as tf
from src.model import create_model
from src.data_loader import get_data_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

# -------------------------------
# Step 1: Paths and Parameters
# -------------------------------
train_dir = "data/train"
val_dir = "data/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 20
MODEL_SAVE_PATH = "models/efficientnet_apple_model.h5"

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# Step 2: Load Datasets
# -------------------------------
train_dataset, val_dataset = get_data_generators(train_dir, val_dir)

# Prefetch for better performance
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Step 3: Create the Model
# -------------------------------
model = create_model(num_classes=NUM_CLASSES, input_shape=(224,224,3))
model.summary()

# -------------------------------
# Step 4: Callbacks
# -------------------------------
checkpoint_cb = ModelCheckpoint(
    MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=1
)
earlystop_cb = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

# -------------------------------
# Step 5: Train the Model
# -------------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# -------------------------------
# Step 6: Plot Accuracy & Loss
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("outputs/accuracy_plot.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("outputs/loss_plot.png")
plt.show()
