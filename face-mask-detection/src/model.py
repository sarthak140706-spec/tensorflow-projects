
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def build_model(img_height, img_width, img_channels, num_classes=2, use_transfer=True):

    if use_transfer is True:
        # Transfer Learning
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, img_channels))
        base_model.trainable = False  # freeze base layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    else:
        # Simple CNN from scratch
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, img_channels)))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

    return model
