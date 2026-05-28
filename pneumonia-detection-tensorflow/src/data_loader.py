import os
import tensorflow as tf

# -------------------------------
# Paths
# -------------------------------


TRAIN_PATH = "C:/Users/ASUS/OneDrive/Desktop/pneumonia-detection-tensorflow/data/raw/chest_xray/chest_xray/train"
VALIDATION_PATH = "C:/Users/ASUS/OneDrive/Desktop/pneumonia-detection-tensorflow/data/raw/chest_xray/chest_xray/val"
TEST_PATH = "C:/Users/ASUS/OneDrive/Desktop/pneumonia-detection-tensorflow\data/raw/chest_xray/train"

# -------------------------------
# Constants
# -------------------------------
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
SEED = 42

# -------------------------------
# Dataset Loader Function
# -------------------------------
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        labels="inferred",
        label_mode="binary",
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_PATH,
        labels="inferred",
        label_mode="binary",
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        labels="inferred",
        label_mode="binary",
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_ds, val_ds, test_ds
