# 😷 Face Mask Detection Using CNN & Transfer Learning

A deep learning-based web application that detects whether a person is **wearing a face mask or not** using a Convolutional Neural Network (CNN) with **MobileNetV2 Transfer Learning**.

## 🌐 Live Demo

🚀 **Application:** https://face-mask-detection-cz54.onrender.com/

---

## 📌 Project Overview

This project uses **TensorFlow**, **Keras**, and **MobileNetV2** to classify facial images into two categories:

- ✅ With Mask
- ❌ Without Mask

The application allows users to upload an image and receive an instant prediction with the trained deep learning model.

---

## ✨ Features

- Upload image for prediction
- CNN model built using MobileNetV2 Transfer Learning
- Image preprocessing and normalization
- Real-time predictions through Streamlit
- Model evaluation with accuracy and loss visualization
- Confusion Matrix and Classification Report

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Render

---

## 📂 Project Structure

```
face-mask-detection/
│
├── data/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   └── mask_detector.h5
│
├── plots/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocess.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/sarthak140706-spec/tensorflow-projects.git
```

Navigate to the project

```bash
cd tensorflow-projects/face-mask-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Model Training

Train the CNN model

```bash
python src/train.py
```

The trained model is saved to

```
models/mask_detector.h5
```

Training graphs are saved inside

```
plots/
```

---

## 📊 Model Evaluation

Evaluate the trained model

```bash
python src/evaluate.py
```

The evaluation includes

- Test Accuracy
- Test Loss
- Confusion Matrix
- Classification Report

---

## 📈 Results

Current model accuracy:

**≈ 44%**

The relatively low accuracy may be due to:

- Limited training dataset
- Class imbalance
- Insufficient image augmentation
- Variations in lighting and face orientation

---

## 🔮 Future Improvements

- Increase dataset size
- Apply advanced image augmentation
- Fine-tune additional MobileNetV2 layers
- Experiment with EfficientNet and ResNet architectures
- Add webcam-based real-time face mask detection
- Deploy using Docker and Kubernetes
- Improve UI with prediction confidence and probability charts

---

## 📷 Sample Workflow

```
Upload Image
      │
      ▼
Image Preprocessing
      │
      ▼
MobileNetV2 CNN
      │
      ▼
Prediction
      │
      ├── 😷 With Mask
      └── 🚫 Without Mask
```

---

## 👨‍💻 Author

**Sarthak Jadhav**

B.Tech Artificial Intelligence & Data Science

AISSMS Institute of Information Technology, Pune

---

## ⭐ If you found this project useful, consider giving it a star!
