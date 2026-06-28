"""
Data Augmentation for Face Mask Detection
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_val_augmentations():
    """
    Creates ImageDataGenerator objects for training,
    validation and testing.

    NOTE:
    Images are already preprocessed using
    MobileNetV2's preprocess_input() in train.py.
    Therefore, DO NOT use rescale=1./255 here.
    """

    # Training Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    # Validation (No augmentation)
    val_datagen = ImageDataGenerator()

    # Testing (No augmentation)
    test_datagen = ImageDataGenerator()

    return train_datagen, val_datagen, test_datagen