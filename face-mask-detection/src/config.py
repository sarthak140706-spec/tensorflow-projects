DATA_DIR = "data/raw"
TRAINED_MODEL_PATH = "models/mask_detector.h5"
PLOTS_DIR = "plots" 

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

import os
if not os.path.exists("models"):
    os.makedirs("models")
