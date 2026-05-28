import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import load_datasets
from preprocesses import preprocess_datasets
from model import get_model
from plots import plot_training
import os

# Load datasets
training_dataset, validation_dataset, test_dataset = load_datasets()

# Preprocess datasets
training_dataset, validation_dataset, test_dataset = preprocess_datasets(
    training_dataset, validation_dataset, test_dataset
)

# Load model
model = get_model(input_shape=(224,224,3), pretrained=True)

# Callbacks
checkpoint = ModelCheckpoint(
    '../models/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Train model (SAVE history)
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=5,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Plot results
plot_training(history)

# Save final model
model.save('../models/final_model.keras')
