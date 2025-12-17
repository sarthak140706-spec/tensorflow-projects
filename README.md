# EfficientNet Plant Disease Classifier

This project implements a deep learning pipeline for **plant disease classification** using the **EfficientNetB0** architecture. It focuses on identifying diseases in apple leaves from the PlantVillage dataset.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Folder Structure](#folder-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Training and Fine-Tuning](#model-training-and-fine-tuning)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)

---

## Project Overview

This project builds a **convolutional neural network** based on **EfficientNetB0** to classify apple leaf images into one of four categories:

1. Apple Scab  
2. Black Rot  
3. Cedar Apple Rust  
4. Healthy  

The pipeline includes:

- Data preprocessing and augmentation  
- Model creation using EfficientNetB0 as the backbone  
- Training on a subset of data  
- Fine-tuning the last layers for improved accuracy  
- Evaluation of model performance  

---

## Dataset

The model is trained on the **PlantVillage dataset**:

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- Only apple leaf images are used for this project.  
- Training and validation directories are structured as follows:

data/
├── train/
│ ├── Apple___Apple_scab/
│ ├── Apple___Black_rot/
│ ├── Apple___Cedar_apple_rust/
│ └── Apple___healthy/
└── val/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
└── Apple___healthy/

yaml
Copy code

💡 **Note:** The `data/` folder is **not included** in this repository due to size. Please download the dataset from Kaggle and place it in the above structure.

---

## Folder Structure

efficientnet-plant-disease-classifier/
│ README.md
│ requirements.txt
│
├── data/ # Training and validation images
├── models/ # Saved models (.h5)
├── notebooks/ # Jupyter notebooks
│ efficientnet_finetuning.ipynb
├── outputs/ # Optional outputs (plots, logs)
└── src/ # Source code
init.py
data_loader.py
model.py
train.py
fine_tune.py
evaluate.py

yaml
Copy code

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/efficientnet-plant-disease-classifier.git
cd efficientnet-plant-disease-classifier
Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows:

bash
Copy code
venv\Scripts\activate
Linux/Mac:

bash
Copy code
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
1. Prepare the dataset
Download the PlantVillage apple leaf dataset.

Arrange it in the data/train and data/val directories as shown above.

2. Train the model
Run training script:

bash
Copy code
python src/train.py
3. Fine-tune the model
Run fine-tuning script:

bash
Copy code
python src/fine_tune.py
4. Evaluate the model
Run evaluation script:

bash
Copy code
python src/evaluate.py
5. Jupyter Notebook
Open the notebook for an interactive exploration:

bash
Copy code
jupyter notebook notebooks/efficientnet_finetuning.ipynb
Model Training and Fine-Tuning
Initial training uses EfficientNetB0 with the base model frozen.

Fine-tuning unfreezes the last 50 layers of the base model.

Early stopping and model checkpoints are used to prevent overfitting.

The final model achieves ~50-60% validation accuracy on a small subset. Accuracy can improve with more epochs or full dataset training.

Results
The model predicts apple leaf diseases with four classes.

Saved model file: models/efficientnet_apple_model.h5

💡 Tip: Experiment with different learning rates, batch sizes, and more data for higher accuracy.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.

Create a feature branch.

Make your changes.

Submit a pull request.

License
This project is licensed under the MIT License.
See LICENSE file for details.

pgsql
Copy code

This is fully **copy-paste ready** for your `README.md`.  

Do you want me to make it **even shorter and more concise**, suitable for GitHub display without scrolling too much?
