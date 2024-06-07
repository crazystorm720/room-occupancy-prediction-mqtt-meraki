# Room Occupancy Prediction Using MQTT Sensor Data from Meraki

This repository contains the complete workflow and code for predicting room occupancy using sensor data collected from Meraki devices via MQTT. The project demonstrates the end-to-end process from data collection and preprocessing to model training, inference, and monitoring.

## Table of Contents
- [Introduction](#introduction)
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

## Data Collection
### Real-time Data Ingestion
- **Tools:** MQTT broker, Meraki sensors.
- **Process:** Collecting real-time data on fan speed, temperature, and humidity.

### Batch Data Collection
- **Tools:** TimescaleDB.
- **Process:** Aggregating data over specified periods for batch inference.

## Data Preprocessing
### Steps
- Handling missing values.
- Removing outliers.
- Feature engineering (lag features, rolling statistics, time-based features).
- Normalization and scaling.

## Feature Engineering
- **Lag Features:** Capture temporal dependencies.
- **Rolling Statistics:** Calculate moving averages and other rolling metrics.
- **Time-Based Features:** Extract features like hour of the day and day of the week.
- **Interaction Terms:** Combine features to capture interactions.

## Model Training
### Training Process
- **Training Set:** Used to train the model.
- **Validation Set:** Used to tune hyperparameters and avoid overfitting.
- **Test Set:** Used to evaluate the final model's performance.
- **Tools:** scikit-learn, hyperparameter tuning libraries.

### Metrics for Evaluation
- Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared.

## Model Inference
### Real-time Inference
- Collect real-time data via MQTT.
- Preprocess data and apply the trained model to make predictions.

### Batch Inference
- Aggregate data over a period, preprocess, and predict.

## Monitoring and Logging
- **Log Predictions:** Store predictions for future analysis and auditing.
- **Monitor Performance:** Track model performance over time to detect drift and degradation.

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
