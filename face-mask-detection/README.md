Face Mask Detection Using CNN and Transfer Learning
Project Overview

This project implements a face mask detection system using Convolutional Neural Networks (CNNs) and MobileNetV2 transfer learning. The goal is to classify images into two categories:

with_mask

without_mask

The system loads raw images, preprocesses them, trains a model, and evaluates performance on a test dataset.

Dataset

The dataset consists of images of faces with and without masks.

Images were split into training, validation, and test sets.

Usage
Training
python src/train.py


Trains the model using the training dataset.

Saves the best model to models/mask_detector.h5.

Saves accuracy and loss plots in the plots/ folder.

Evaluation
python src/evaluate.py


Loads the trained model.

Evaluates performance on the test dataset.

Prints loss, accuracy, confusion matrix, and classification report.

Saves confusion matrix plot in the plots/ folder.

Observations

Despite using transfer learning with MobileNetV2, the model achieved ~44% accuracy on the test set.

Possible reasons for low performance:

Small dataset or class imbalance.

Poor image quality or inconsistent face alignment.

Insufficient data augmentation or training epochs.

Overfitting or underfitting due to limited training data.

Improvements that can be tried:

Collect more labeled images.

Apply stronger augmentations.

Fine-tune more layers of MobileNetV2.

Experiment with different architectures or optimizers.

Conclusion

This project demonstrates a complete workflow for image classification using CNNs and transfer learning.

While the current model's accuracy is low, it provides a foundation for further improvements in face mask detection systems.