import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_data_generators(train_dir,val_dir):
    """"
    We are creating slightly different versions of the same
    images to prevent overfitting and make the model more robust.
    So the images here are augmented
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    """"
    here the images arent augmented 
    they are just normalized 
    """
    val_datagen = ImageDataGenerator(rescale=1./255)


    """
    This line is connecting your images to the model
    flow_from_directory()
    Reads images from folders (each folder = one class)
    """
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    return train_generator, val_generator