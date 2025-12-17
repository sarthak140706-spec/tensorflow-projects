# model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

def create_model(num_classes=4, input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Creates and compiles an EfficientNetB0 model for multiclass classification.

    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of input images (height, width, channels).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """

    # Load EfficientNetB0 base model with pretrained ImageNet weights
    base_model = EfficientNetB0(
        include_top=False,  # remove the original classification head
        weights='imagenet', # use pretrained weights
        input_shape=input_shape
    )

    # Freeze the base model
    base_model.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # dropout to prevent overfitting
    output = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    # Quick test
    model = create_model(num_classes=4)
    model.summary()
