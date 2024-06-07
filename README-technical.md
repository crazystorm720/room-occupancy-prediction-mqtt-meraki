# Room Occupancy Prediction Using MQTT Sensor Data from Meraki

This repository contains the complete workflow and code for predicting room occupancy using sensor data collected from Meraki devices via MQTT. The project demonstrates the end-to-end process from data collection and preprocessing to model training, inference, and monitoring.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Project Setup](#project-setup)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Monitoring and Logging](#monitoring-and-logging)
- [Advanced Techniques](#advanced-techniques)
- [Example Project Walkthrough](#example-project-walkthrough)
- [References](#references)

## Introduction
This project aims to predict the number of people in a room using sensor data from Meraki devices. We use MQTT for real-time data ingestion and focus on chassis fan speed and environmental sensors (temperature, humidity) to build a predictive model.

## Project Structure
The project is structured as follows:

```
room-occupancy-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_inference.py
│   └── monitoring.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── tests/
│   └── unit_tests.py
│
├── README.md
├── PROJECT_SETUP.md
├── requirements.txt
└── Dockerfile
```

- The `data` directory contains subdirectories for raw data, processed data, and trained models.
- The `src` directory contains the Python scripts for different components of the project.
- The `notebooks` directory contains Jupyter notebooks for exploratory data analysis and experimentation.
- The `tests` directory contains unit tests for the project.
- The `README.md` file provides an overview of the project and instructions for usage.
- The `PROJECT_SETUP.md` file contains instructions for setting up the project environment.
- The `requirements.txt` file lists the project dependencies.
- The `Dockerfile` defines the Docker image for the project.

## Project Setup
To set up the project, follow the instructions in the [PROJECT_SETUP.md](PROJECT_SETUP.md) file. It provides a step-by-step guide to configure the project using Docker, Conda, Python, and Git.

## Data Collection
### Real-time Data Ingestion
- **Tools:** MQTT broker, Meraki sensors.
- **Process:** Collecting real-time data on fan speed, temperature, and humidity.
- **Implementation:** The `data_collection.py` file contains functions to connect to the MQTT broker, process MQTT messages, and store the collected data in a database.

### Batch Data Collection
- **Tools:** TimescaleDB.
- **Process:** Aggregating data over specified periods for batch inference.
- **Implementation:** The `data_collection.py` file includes functions to retrieve data from the database for a specified date range.

## Data Preprocessing
### Steps
- Handling missing values.
- Removing outliers.
- Feature engineering (lag features, rolling statistics, time-based features).
- Normalization and scaling.
- **Implementation:** The `data_preprocessing.py` file contains functions to load data, handle missing values, remove outliers, and perform feature scaling.

## Feature Engineering
- **Lag Features:** Capture temporal dependencies.
- **Rolling Statistics:** Calculate moving averages and other rolling metrics.
- **Time-Based Features:** Extract features like hour of the day and day of the week.
- **Interaction Terms:** Combine features to capture interactions.
- **Implementation:** The `feature_engineering.py` file includes functions to create lag features, rolling statistics, time-based features, and interaction terms.

## Model Training
### Training Process
- **Training Set:** Used to train the model.
- **Validation Set:** Used to tune hyperparameters and avoid overfitting.
- **Test Set:** Used to evaluate the final model's performance.
- **Tools:** scikit-learn, hyperparameter tuning libraries.
- **Implementation:** The `model_training.py` file contains functions to split the data, train the machine learning model, and evaluate its performance.

### Metrics for Evaluation
- Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared.

## Model Inference
### Real-time Inference
- Collect real-time data via MQTT.
- Preprocess data and apply the trained model to make predictions.
- **Implementation:** The `model_inference.py` file includes functions to load the trained model, preprocess input data, and make predictions in real-time.

### Batch Inference
- Aggregate data over a period, preprocess, and predict.
- **Implementation:** The `model_inference.py` file also contains functions to perform batch inference on aggregated data.

## Monitoring and Logging
- **Log Predictions:** Store predictions for future analysis and auditing.
- **Monitor Performance:** Track model performance over time to detect drift and degradation.
- **Implementation:** The `monitoring.py` file includes functions to log predictions, calculate performance metrics, and detect anomalies.

## Advanced Techniques
- **Feature Selection:** Identify the most important features to reduce dimensionality.
- **Ensemble Methods:** Combine predictions from multiple models to improve accuracy.
- **Cross-Validation:** Use cross-validation techniques to ensure generalization.

## Example Project Walkthrough
### Objective
Predicting room occupancy using chassis fan speed, temperature, and humidity.

### Steps
1. **Data Collection**
2. **Data Preprocessing**
3. **Feature Engineering**
4. **Model Training and Evaluation**
5. **Real-time and Batch Inference**
6. **Monitoring and Logging**

## References
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [MQTT Protocol](https://mqtt.org/)
- [TimescaleDB Documentation](https://docs.timescale.com/latest/main)
- [Meraki Sensors](https://meraki.cisco.com/products/sensors)