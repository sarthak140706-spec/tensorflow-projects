import numpy as np
import config 
from data_loader import load_data
from augment import get_train_val_augmentations
from model import build_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Load train, val, test splits
train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(
    config.DATA_DIR, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
)

# 2. Map string labels to integers
class_mapping = {"with_mask": 0, "without_mask": 1}
train_labels = np.array([class_mapping[label] for label in train_labels])
val_labels = np.array([class_mapping[label] for label in val_labels])
test_labels = np.array([class_mapping[label] for label in test_labels])

# 3. Convert labels to categorical (one-hot encoding)
train_labels = to_categorical(train_labels, num_classes=2)
val_labels = to_categorical(val_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

# 4. Preprocess images
def preprocess_images(image_paths, img_height, img_width):
    data = []
    for path in image_paths:
        img = load_img(path, target_size=(img_height, img_width))
        img = img_to_array(img) / 255.0  # normalize
        data.append(img)
    return np.array(data, dtype="float32")

train_data = preprocess_images(train_images, config.IMG_HEIGHT, config.IMG_WIDTH)
val_data = preprocess_images(val_images, config.IMG_HEIGHT, config.IMG_WIDTH)
test_data = preprocess_images(test_images, config.IMG_HEIGHT, config.IMG_WIDTH)

# 5. Get data augmentations
train_datagen, val_datagen, _ = get_train_val_augmentations(config.IMG_HEIGHT, config.IMG_WIDTH, config.BATCH_SIZE)

# 6. Build and compile the model
model = build_model(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS, num_classes=2, use_transfer=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Define callbacks
checkpoint = ModelCheckpoint(config.TRAINED_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# 8. Train the model
model.fit(
    train_datagen.flow(train_data, train_labels, batch_size=config.BATCH_SIZE),
    validation_data=val_datagen.flow(val_data, val_labels, batch_size=config.BATCH_SIZE),
    epochs=config.EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# 9. Save final model (optional, already saved by checkpoint)
model.save(config.TRAINED_MODEL_PATH)
