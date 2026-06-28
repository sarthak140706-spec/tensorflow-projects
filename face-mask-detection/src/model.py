"""
Model Architecture for Face Mask Detection
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    GlobalAveragePooling2D
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2


def build_model(
    img_height,
    img_width,
    img_channels,
    num_classes=2,
    use_transfer=True
):
    """
    Builds the Face Mask Detection model.

    Parameters
    ----------
    img_height : int
    img_width : int
    img_channels : int
    num_classes : int
    use_transfer : bool

    Returns
    -------
    TensorFlow/Keras Model
    """

    if use_transfer:

        # -------------------------------------------------
        # MobileNetV2 Backbone
        # -------------------------------------------------

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(img_height, img_width, img_channels)
        )

        # Freeze pretrained layers
        base_model.trainable = False

        inputs = Input(
            shape=(img_height, img_width, img_channels)
        )

        x = base_model(inputs, training=False)

        x = GlobalAveragePooling2D()(x)

        x = BatchNormalization()(x)

        x = Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(0.001)
        )(x)

        x = Dropout(0.5)(x)

        outputs = Dense(
            num_classes,
            activation="softmax"
        )(x)

        model = Model(
            inputs,
            outputs
        )

    else:

        # -------------------------------------------------
        # Custom CNN
        # -------------------------------------------------

        model = Sequential([

            Input(
                shape=(img_height, img_width, img_channels)
            ),

            Conv2D(
                32,
                (3, 3),
                activation="relu"
            ),

            MaxPooling2D(),

            Conv2D(
                64,
                (3, 3),
                activation="relu"
            ),

            MaxPooling2D(),

            Conv2D(
                128,
                (3, 3),
                activation="relu"
            ),

            MaxPooling2D(),

            Flatten(),

            Dense(
                256,
                activation="relu"
            ),

            Dropout(0.5),

            Dense(
                num_classes,
                activation="softmax"
            )

        ])

    return model