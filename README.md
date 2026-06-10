# 🔋 EV Battery Capacity Prediction

A Deep Learning based regression model that predicts Electric Vehicle (EV) battery capacity using voltage, current, temperature, and load measurements. The project includes data preprocessing, feature engineering, model training with TensorFlow, and an interactive Streamlit web application for real-time predictions.

---

## Overview

Battery capacity is one of the most important indicators of battery health and remaining useful life in electric vehicles. This project leverages machine learning to estimate battery capacity from sensor measurements, enabling intelligent battery monitoring and predictive maintenance.

---

## Features

* Battery capacity prediction using Deep Learning
* Statistical feature extraction from battery sensor data
* Data preprocessing with StandardScaler
* Interactive Streamlit web application
* Real-time battery health visualization
* Trained TensorFlow regression model
* Saved model and scaler for deployment

---

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-Learn
* NumPy
* Pandas
* Streamlit
* Joblib
* Matplotlib

---

## Project Structure

```
EV-BATTERY-LIFE-PREDICTION/
│
├── data/
│   ├── metadata.csv
│   ├── raw_data/
│   └── extra_infos/
│
├── models/
│   ├── battery_life_model.keras
│   └── scaler.pkl
│
├── notebooks/
│   └── training.ipynb
│
├── utils/
│   └── preprocess.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE (optional)
```

---

## Workflow

### 1. Data Collection

Battery cycle data and metadata are loaded from the dataset.

### 2. Feature Engineering

The following statistical features are extracted:

* Voltage Mean
* Voltage Standard Deviation
* Current Mean
* Current Standard Deviation
* Temperature Mean
* Temperature Maximum
* Current Load Mean
* Voltage Load Mean
* Time Maximum

### 3. Data Preprocessing

* Train-Test Split
* Feature Scaling using StandardScaler
* Scaler saved using Joblib

### 4. Model Training

A TensorFlow Neural Network is trained for battery capacity regression.

Architecture:

```
Input Layer (9 Features)
        ↓
Dense (64, ReLU)
        ↓
Dense (32, ReLU)
        ↓
Dense (16, ReLU)
        ↓
Dense (1)
```

### 5. Model Evaluation

The trained model is evaluated using multiple regression metrics.

| Metric   | Score  |
| -------- | ------ |
| MAE      | 0.0163 |
| MSE      | 0.0019 |
| RMSE     | 0.0431 |
| R² Score | 0.9918 |

The model explains approximately **99.18%** of the variance in battery capacity while maintaining a very low prediction error.

---

## Streamlit Application

The application allows users to:

* Enter battery sensor measurements
* Predict remaining battery capacity
* View battery health status
* Display model evaluation metrics

Battery Health Categories:

| Capacity | Status    |
| -------- | --------- |
| ≥ 80%    | Excellent |
| 60–79%   | Good      |
| 40–59%   | Moderate  |
| < 40%    | Poor      |

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/EV-Battery-Life-Prediction.git
```

Move into the project directory

```bash
cd EV-Battery-Life-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Model Performance

```
MAE    : 0.0163
MSE    : 0.0019
RMSE   : 0.0431
R²     : 0.9918
```

---

## Future Improvements

* Battery Remaining Useful Life (RUL) prediction
* LSTM-based time-series modeling
* Hyperparameter optimization
* Explainable AI with SHAP values
* Cloud deployment with Docker and CI/CD
* REST API integration

---

## Author

Developed as a Machine Learning project for Electric Vehicle Battery Capacity Prediction using TensorFlow and Streamlit.
