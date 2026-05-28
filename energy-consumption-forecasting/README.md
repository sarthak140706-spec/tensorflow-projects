# Energy Consumption Forecasting

This project implements a **time-series forecasting pipeline** for energy consumption using Python and machine learning. It includes data preprocessing, feature engineering, model training, evaluation, and prediction.

---

## **Project Structure**

energy-consumption-forecasting/
│
├── data/
│ └── energy_dataset.csv # Raw dataset
│
├── models/ # Folder to save trained models
│ └── xgboost_energy_model.pkl
│
├── src/
│ ├── data_loader.py # Loads and parses dataset
│ ├── preprocessing.py # Handles missing values, outliers, scaling, feature engineering
│ ├── train.py # Training pipeline with model evaluation
│ └── predict.py # Forecasting using trained model
│
├── main.py # Main script to run the full pipeline
├── README.md # Project documentation
└── requirements.txt # Python dependencies

yaml
Copy code

---

## **Installation**

1. Clone the repository:
```bash
git clone <repository-url>
cd energy-consumption-forecasting
Create a virtual environment (recommended):

bash
Copy code
python -m venv tf
Activate the environment:

Windows:

bash
Copy code
tf\Scripts\activate
Linux/macOS:

bash
Copy code
source tf/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Place your dataset energy_dataset.csv inside the data/ folder.

Run the main pipeline:

bash
Copy code
python main.py
This will:

Load the dataset

Preprocess it (handle missing values, remove outliers, create lag features)

Train an XGBoost model

Save the model to models/xgboost_energy_model.pkl

Print evaluation metrics (MAE, MSE, R²)

Forecast energy consumption for future steps (optional using predict.py)

Customization
Change model type:
Modify model_type in main.py (linear, random_forest, xgboost).

Adjust lag features:
Modify lags in preprocessing.py (default is [1]).

Change scaling method:
scale_method='minmax' or 'standard'.

Evaluation Metrics
MAE (Mean Absolute Error): Measures average magnitude of errors.

MSE (Mean Squared Error): Measures squared errors.

R² Score: Explains variance captured by the model.

Dependencies
Python 3.10+

pandas, numpy, scikit-learn, scipy, xgboost, joblib

Notes
Ensure the models/ folder exists for saving trained models.

Make sure the time column in the CSV is correctly formatted (ISO 8601 or datetime-like).

Reduce the number of lag features if you encounter “No data left after lag feature creation” errors.