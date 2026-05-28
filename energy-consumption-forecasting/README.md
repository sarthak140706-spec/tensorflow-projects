# ⚡ Energy Consumption Forecasting

A Machine Learning powered web application for forecasting energy consumption using time-series analysis, feature engineering, and XGBoost.

---

## 🚀 Live Demo

Live Streamlit App:

https://tensorflow-projects-eth49cwgnlcgtmvdhgjkbc.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/sarthak140706-spec/tensorflow-projects

---

# 📌 Features

* 📊 Energy consumption forecasting
* ⚡ Interactive Streamlit dashboard
* 🧠 XGBoost regression model
* 🕒 Time-series feature engineering
* 📈 Real-time prediction interface
* 🛠 Automated preprocessing pipeline
* 💾 Model persistence using Joblib

---

# 🏗 Project Structure

```bash
energy-consumption-forecasting/
│
├── data/
│   └── energy_dataset.csv
│
├── models/
│   └── xgboost_energy_model.pkl
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── main.py
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/sarthak140706-spec/tensorflow-projects.git
```

```bash
cd tensorflow-projects/energy-consumption-forecasting
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv tf
tf\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv tf
source tf/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

## Streamlit Web App

```bash
streamlit run app.py
```

---

## Training Pipeline

```bash
python main.py
```

This pipeline performs:

* Dataset loading
* Missing value handling
* Outlier processing
* Feature engineering
* Model training
* Model evaluation
* Model saving

---

# 📊 Machine Learning Workflow

## Data Preprocessing

* Missing value handling
* Time conversion
* Feature extraction
* Scaling
* Lag feature generation

---

## Feature Engineering

Extracted features include:

* Hour
* Day
* Month
* Year
* Weekday

---

## Model Used

* XGBoost Regressor

---

# 📈 Evaluation Metrics

The project evaluates model performance using:

* **MAE** → Mean Absolute Error
* **MSE** → Mean Squared Error
* **R² Score** → Variance explanation score

---

# 🛠 Tech Stack

* Python
* Streamlit
* XGBoost
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

# 🔮 Future Improvements

* Deep Learning (LSTM/GRU)
* Real-time API integration
* Energy trend visualization
* Interactive analytics dashboard
* Multi-step forecasting
* Cloud deployment optimization

---

# 👨‍💻 Author

Sarthak Jadhav

AI & Data Science Engineering Student

---

# ⭐ Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the project and submit pull requests.

---

# 📜 License

This project is open-source and available under the MIT License.
