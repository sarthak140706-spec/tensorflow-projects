# src/fine_tune.py

import tensorflow as tf
from src.model import create_model
from src.data_loader import get_data_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# -------------------------------
# Step 1: Paths and Parameters
# -------------------------------
train_dir = "data/train"
val_dir = "data/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10   # Fine-tuning usually uses fewer epochs
MODEL_SAVE_PATH = "models/efficientnet_apple_finetuned.h5"

# -------------------------------
# Step 2: Load Datasets
# -------------------------------
train_dataset, val_dataset = get_data_generators(train_dir, val_dir)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Step 3: Load Base Model
# -------------------------------
# Load previously trained model
model = create_model(num_classes=NUM_CLASSES, input_shape=(224,224,3))

# Load weights from initial training
initial_model_path = "models/efficientnet_apple_model.h5"
model.load_weights(initial_model_path)

# -------------------------------
# Step 4: Unfreeze Some Layers for Fine-Tuning
# -------------------------------
# Unfreeze last 50 layers of the base model
model.layers[0].trainable = True
for layer in model.layers[0].layers[:-50]:
    layer.trainable = False

# Compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# Step 5: Callbacks
# -------------------------------
checkpoint_cb = ModelCheckpoint(
    MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=1
)
earlystop_cb = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)

# -------------------------------
# Step 6: Fine-Tune the Model
# -------------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print("Fine-tuned model saved at:", MODEL_SAVE_PATH)
