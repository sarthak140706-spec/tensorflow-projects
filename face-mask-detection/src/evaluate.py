import config
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
from data_loader import load_data
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load test dataset only
_, _, _, _, test_images, test_labels = load_data(
    config.DATA_DIR, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
)

# 2. Map string labels to integers
class_mapping = {"with_mask": 0, "without_mask": 1}
test_labels = np.array([class_mapping[label] for label in test_labels])

# 3. Convert to categorical
test_labels = to_categorical(test_labels, num_classes=2)

# 4. Preprocess test images
def preprocess_images(image_paths, img_height, img_width):
    data = []
    for path in image_paths:
        img = load_img(path, target_size=(img_height, img_width))
        img = img_to_array(img) / 255.0  # normalize
        data.append(img)
    return np.array(data, dtype="float32")

test_data = preprocess_images(test_images, config.IMG_HEIGHT, config.IMG_WIDTH)

# 5. Load the trained model
model = load_model(config.TRAINED_MODEL_PATH)

# 6. Evaluate
loss, accuracy = model.evaluate(test_data, test_labels, batch_size=config.BATCH_SIZE)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# 7. Predictions and detailed metrics
predictions = model.predict(test_data, batch_size=config.BATCH_SIZE)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

class_names = ["with_mask", "without_mask"]
cm = confusion_matrix(true_classes, predicted_classes)
report = classification_report(true_classes, predicted_classes, target_names=class_names)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
