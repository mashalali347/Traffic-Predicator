# 🚦 Smart Traffic Prediction & Analysis System

## 📌 Project Overview

The **Smart Traffic Prediction & Analysis System** is a machine learning project designed to analyze traffic data, predict traffic congestion levels, and estimate the number of vehicles based on different traffic conditions.

The system uses two Random Forest machine learning models and provides an interactive web dashboard built with Streamlit.

## 🎯 Objectives

- Predict traffic levels: Low, Medium, or High.
- Estimate vehicle counts based on traffic conditions.
- Analyze traffic patterns using data visualizations.
- Explore how weather, time, temperature, holidays, and road types relate to traffic.
- Present predictions through a user-friendly dashboard.

## ✨ Features

- **Traffic Level Classification:** Predicts Low, Medium, or High traffic.
- **Vehicle Count Regression:** Estimates the number of vehicles.
- **Data Visualization:** Displays traffic distribution and average vehicle counts by hour.
- **Interactive Dashboard:** Allows users to enter traffic conditions and view predictions.
- **Dataset Overview:** Displays dataset statistics and traffic information.

## 🧠 Machine Learning Models

### 1. Random Forest Classifier

Used to classify traffic conditions into three categories:

- Low
- Medium
- High

### 2. Random Forest Regressor

Used to estimate the expected number of vehicles based on the selected traffic conditions.

## 📊 Dataset

The project uses a synthetic dataset containing 1,000 records and nine columns:

| Column | Description |
|---|---|
| Date | Date of the traffic record |
| Hour | Hour of the day |
| Day | Day of the week |
| Weather | Weather conditions |
| Temperature | Temperature in Celsius |
| Holiday | Whether the day is a holiday |
| Road_Type | Type of road |
| Vehicle_Count | Number of vehicles |
| Traffic_Level | Traffic category |

**Note:** The dataset is synthetic and intended for educational and demonstration purposes. Predictions should not be treated as real-world traffic forecasts without validation using actual traffic data.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib
- Git and GitHub

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Smart-Traffic-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create and activate a virtual environment

For Windows Git Bash:

```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application should open in your browser.

## 📁 Project Structure

```text
Smart-Traffic-AI/
├── data/
│   └── traffic_data_1000_rows.csv
├── models/
│   ├── traffic_classifier.pkl
│   └── traffic_regressor.pkl
├── src/
│   ├── preprocessing.py
│   ├── train_classification.py
│   ├── train_regression.py
│   └── utils.py
├── app.py
├── README.md
└── requirements.txt
```

## 🚀 Future Improvements

- Train and validate the models using real-world traffic data.
- Add model evaluation metrics and confusion matrix visualizations.
- Introduce interactive filters for traffic analysis.
- Explore time-series forecasting.
- Add location-specific traffic data and map-based visualization.

## 🎓 Project Purpose

This project was developed as an educational machine learning project to practice data preprocessing, classification, regression, model evaluation, data visualization, and deployment using Streamlit.

## 👨‍💻 Author

Add your name here.