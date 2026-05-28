import tensorflow as tf

def preprocess_datasets(training_dataset, validation_dataset, test_dataset):

    #---------------------------------------------------------
    #                   NORMALIZATION
    #---------------------------------------------------------
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    training_dataset = training_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    #---------------------------------------------------------
    #                   AUGMENTATION
    #---------------------------------------------------------
    augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])
    training_dataset = training_dataset.map(lambda x, y: (augmentation_layer(x, training=True), y))

    #---------------------------------------------------------
    #                   OPTIMIZATION
    #---------------------------------------------------------
    AUTOTUNE = tf.data.AUTOTUNE
    training_dataset = training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return training_dataset, validation_dataset, test_dataset
