"""
Configuration file for Face Mask Detection Project
"""

import os

# ==========================================================
# Paths
# ==========================================================

# Dataset folder
DATA_DIR = "data"

# Folder to save trained model
MODEL_DIR = "models"

# Trained model path
TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "mask_detector.keras")

# Folder to save graphs
PLOTS_DIR = "plots"

# ==========================================================
# Image Parameters
# ==========================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# ==========================================================
# Training Parameters
# ==========================================================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# ==========================================================
# Dataset Split
# ==========================================================

TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42

# ==========================================================
# Model Parameters
# ==========================================================

NUM_CLASSES = 2

CLASS_NAMES = [
    "with_mask",
    "without_mask"
]

USE_TRANSFER_LEARNING = True

# ==========================================================
# Create Required Folders
# ==========================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)