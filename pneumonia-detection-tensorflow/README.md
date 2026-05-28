Pneumonia Detection using Deep Learning
Overview

This project uses deep learning techniques to detect pneumonia from chest X-ray images. The model is built using TensorFlow and Keras, and is trained to classify X-ray images as Normal or Pneumonia.

The project includes data preprocessing, model training, evaluation, and visualization of results.

Features

Image preprocessing and augmentation

Convolutional Neural Network (CNN) model

Training and validation accuracy tracking

Model evaluation with metrics like accuracy, loss, and confusion matrix

Easy-to-use Python scripts for training and testing

Dataset

The dataset folder (data/) has been removed from this repository due to size constraints.
You can download the dataset from: Kaggle Chest X-Ray Images (Pneumonia)

After downloading, create a folder named data in the project root and place the dataset there.

Folder structure inside data/ should look like:

data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

Installation

Clone the repository:

git clone <your-repo-link>
cd pneumonia-detection-tensorflow


Create a virtual environment (optional but recommended):

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

Usage

Training the model:

python 2_model_training.py


Evaluating the model:

python 3_model_evaluation.py


Testing with custom images:

python 4_predict_image.py --image <path-to-image>

Results

Training and validation accuracy graphs are generated automatically.

Confusion matrix and classification reports are available after evaluation.

Requirements

Python 3.8+

TensorFlow

NumPy

Matplotlib

OpenCV (for image preprocessing, optional)

scikit-learn

(All dependencies are listed in requirements.txt.)

Notes

Ensure that the data/ folder is structured correctly before running the scripts.

GPU is recommended for faster training but CPU will work for smaller experiments.

Model architecture and hyperparameters can be modified in 2_model_training.py.