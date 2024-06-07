# Room Occupancy Prediction: Source Code Guide

This guide provides an overview of the source code structure and functionality for the Room Occupancy Prediction project. The project is divided into several modules, each responsible for a specific aspect of the data pipeline.

## 1. Data Collection (`src/data_collection.py`)

The `data_collection.py` module is responsible for collecting real-time sensor data from Meraki devices using MQTT. It performs the following tasks:

- Connects to the MQTT broker and subscribes to the specified topic.
- Implements callback functions to handle MQTT events, such as connection and message reception.
- Processes the received MQTT messages and extracts relevant data.
- Stores the processed data in a database using the `psycopg2` library.

The module provides functions like `connect_mqtt()`, `on_connect()`, `on_message()`, `process_mqtt_message()`, and `store_data()` to facilitate the data collection process.

## 2. Data Preprocessing (`src/data_preprocessing.py`)

The `data_preprocessing.py` module handles the preprocessing of the collected sensor data. It performs the following tasks:

- Loads data from the database for a specified date range using the `load_data()` function.
- Handles missing values in the data using appropriate techniques implemented in the `handle_missing_values()` function.
- Removes outliers from the data using statistical methods implemented in the `remove_outliers()` function.
- Performs feature scaling using the `scale_features()` function to normalize the data.

The module utilizes libraries such as `pandas`, `numpy`, and `scikit-learn` for data manipulation and preprocessing.

## 3. Feature Engineering (`src/feature_engineering.py`)

The `feature_engineering.py` module is responsible for creating new features from the preprocessed data to improve the predictive power of the machine learning model. It performs the following tasks:

- Creates lag features by shifting the data by specified periods using the `create_lag_features()` function.
- Generates rolling statistics features, such as moving averages, using the `create_rolling_features()` function.
- Extracts time-based features, such as hour of the day and day of the week, using the `create_time_features()` function.

The module uses the `pandas` library for data manipulation and feature engineering.

## 4. Model Training (`src/model_training.py`)

The `model_training.py` module handles the training and evaluation of the machine learning model. It performs the following tasks:

- Splits the data into training and testing sets using the `split_data()` function.
- Trains the machine learning model (e.g., Linear Regression) using the `train_model()` function.
- Evaluates the trained model using metrics such as mean squared error (MSE) and R-squared (R2) score using the `evaluate_model()` function.

The module utilizes the `scikit-learn` library for model training and evaluation.

## 5. Model Inference (`src/model_inference.py`)

The `model_inference.py` module is responsible for loading the trained model and making predictions on new data. It performs the following tasks:

- Loads the trained model from a file using the `load_model()` function.
- Preprocesses the input data using the same steps as in the training phase using the `preprocess_input_data()` function.
- Makes predictions using the loaded model and preprocessed data using the `predict()` function.

The module uses the `pickle` library for model serialization and deserialization.

## 6. Monitoring and Logging (`src/monitoring.py`)

The `monitoring.py` module handles the monitoring and logging of the model's performance and predictions. It performs the following tasks:

- Logs the predictions along with the timestamp using the `log_prediction()` function.
- Calculates performance metrics, such as accuracy and F1 score, using the `calculate_performance_metrics()` function.
- Detects anomalies in the predictions based on a specified threshold using the `detect_anomalies()` function.

The module uses the `logging` and `datetime` libraries for logging and timestamp generation.

## 7. Main Script (`src/main.py`)

The `main.py` script serves as the entry point of the application and orchestrates the flow of data through the various modules. It performs the following tasks:

- Imports the necessary functions from the other modules.
- Implements the `main()` function, which calls the functions from the other modules in the appropriate order.
- Collects data using the `data_collection` module.
- Preprocesses the data using the `data_preprocessing` module.
- Performs feature engineering using the `feature_engineering` module.
- Trains and evaluates the model using the `model_training` module.
- Performs model inference using the `model_inference` module.
- Monitors and logs the predictions and performance using the `monitoring` module.

The script is executed by calling the `main()` function when run directly.

---

1. **Data Collection (`src/data_collection.py`):**
   - Import the necessary libraries:
     ```python
     import paho.mqtt.client as mqtt
     import psycopg2
     ```
   - Implement functions to connect to the MQTT broker and subscribe to the specified topic:
     ```python
     def connect_mqtt(broker_url, port, topic):
         client = mqtt.Client()
         client.on_connect = on_connect
         client.on_message = on_message
         client.connect(broker_url, port)
         client.subscribe(topic)
         client.loop_forever()
     ```
   - Implement callback functions for MQTT events:
     ```python
     def on_connect(client, userdata, flags, rc):
         print(f"Connected to MQTT broker with result code {rc}")

     def on_message(client, userdata, message):
         data = process_mqtt_message(message)
         store_data(data)
     ```
   - Implement functions to process MQTT messages and store data in the database:
     ```python
     def process_mqtt_message(message):
         # Extract relevant data from the MQTT message
         # Return the processed data

     def store_data(data):
         # Store the data in the database
         # Use psycopg2 library to interact with the database
     ```
   - Document the usage and functionality of each function using docstrings.

2. **Data Preprocessing (`src/data_preprocessing.py`):**
   - Import the necessary libraries:
     ```python
     import pandas as pd
     import numpy as np
     from sklearn.preprocessing import StandardScaler
     ```
   - Implement functions to load data from the database:
     ```python
     def load_data(start_date, end_date):
         # Load data from the database for the specified date range
         # Return the loaded data as a pandas DataFrame
     ```
   - Implement functions to handle missing values and remove outliers:
     ```python
     def handle_missing_values(data):
         # Handle missing values in the data using appropriate techniques
         # Return the processed data

     def remove_outliers(data):
         # Remove outliers from the data using statistical methods
         # Return the processed data
     ```
   - Implement functions to perform feature scaling:
     ```python
     def scale_features(data):
         scaler = StandardScaler()
         scaled_data = scaler.fit_transform(data)
         return scaled_data
     ```
   - Document the usage and functionality of each function using docstrings.

3. **Feature Engineering (`src/feature_engineering.py`):**
   - Import the necessary libraries:
     ```python
     import pandas as pd
     ```
   - Implement functions to create lag features:
     ```python
     def create_lag_features(data, lag_periods):
         # Create lag features by shifting the data by specified periods
         # Return the data with lag features
     ```
   - Implement functions to create rolling statistics features:
     ```python
     def create_rolling_features(data, window_size):
         # Create rolling statistics features using a specified window size
         # Return the data with rolling features
     ```
   - Implement functions to create time-based features:
     ```python
     def create_time_features(data):
         # Create time-based features, such as hour of the day and day of the week
         # Return the data with time-based features
     ```
   - Document the usage and functionality of each function using docstrings.

4. **Model Training (`src/model_training.py`):**
   - Import the necessary libraries:
     ```python
     from sklearn.model_selection import train_test_split
     from sklearn.linear_model import LinearRegression
     from sklearn.metrics import mean_squared_error, r2_score
     ```
   - Implement functions to split the data into training and testing sets:
     ```python
     def split_data(data, target_variable, test_size):
         X = data.drop(target_variable, axis=1)
         y = data[target_variable]
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
         return X_train, X_test, y_train, y_test
     ```
   - Implement functions to train the machine learning model:
     ```python
     def train_model(X_train, y_train):
         model = LinearRegression()
         model.fit(X_train, y_train)
         return model
     ```
   - Implement functions to evaluate the trained model:
     ```python
     def evaluate_model(model, X_test, y_test):
         y_pred = model.predict(X_test)
         mse = mean_squared_error(y_test, y_pred)
         r2 = r2_score(y_test, y_pred)
         return mse, r2
     ```
   - Document the usage and functionality of each function using docstrings.

5. **Model Inference (`src/model_inference.py`):**
   - Import the necessary libraries:
     ```python
     import pickle
     ```
   - Implement functions to load the trained model:
     ```python
     def load_model(model_path):
         with open(model_path, 'rb') as file:
             model = pickle.load(file)
         return model
     ```
   - Implement functions to preprocess input data for inference:
     ```python
     def preprocess_input_data(data):
         # Preprocess the input data using the same steps as in training
         # Return the preprocessed data
     ```
   - Implement functions to make predictions using the loaded model:
     ```python
     def predict(model, input_data):
         preprocessed_data = preprocess_input_data(input_data)
         predictions = model.predict(preprocessed_data)
         return predictions
     ```
   - Document the usage and functionality of each function using docstrings.

6. **Monitoring and Logging (`src/monitoring.py`):**
   - Import the necessary libraries:
     ```python
     import logging
     from datetime import datetime
     ```
   - Implement functions to log predictions:
     ```python
     def log_prediction(prediction, timestamp):
         logging.info(f"Prediction: {prediction}, Timestamp: {timestamp}")
     ```
   - Implement functions to calculate performance metrics:
     ```python
     def calculate_performance_metrics(predictions, actual_values):
         # Calculate performance metrics, such as accuracy and F1 score
         # Log the calculated metrics
     ```
   - Implement functions to detect anomalies:
     ```python
     def detect_anomalies(predictions, threshold):
         # Detect anomalies in the predictions based on a specified threshold
         # Log any detected anomalies
     ```
   - Document the usage and functionality of each function using docstrings.

7. **Main Script (`src/main.py`):**
   - Import the necessary functions from the other modules:
     ```python
     from data_collection import connect_mqtt
     from data_preprocessing import load_data, handle_missing_values, remove_outliers, scale_features
     from feature_engineering import create_lag_features, create_rolling_features, create_time_features
     from model_training import split_data, train_model, evaluate_model
     from model_inference import load_model, predict
     from monitoring import log_prediction, calculate_performance_metrics, detect_anomalies
     ```
   - Implement the main function to orchestrate the flow of data through the pipeline:
     ```python
     def main():
         # Data Collection
         connect_mqtt(broker_url, port, topic)

         # Data Preprocessing
         data = load_data(start_date, end_date)
         data = handle_missing_values(data)
         data = remove_outliers(data)
         data = scale_features(data)

         # Feature Engineering
         data = create_lag_features(data, lag_periods)
         data = create_rolling_features(data, window_size)
         data = create_time_features(data)

         # Model Training
         X_train, X_test, y_train, y_test = split_data(data, target_variable, test_size)
         model = train_model(X_train, y_train)
         mse, r2 = evaluate_model(model, X_test, y_test)

         # Model Inference
         model = load_model(model_path)
         predictions = predict(model, input_data)

         # Monitoring and Logging
         timestamp = datetime.now()
         log_prediction(predictions, timestamp)
         calculate_performance_metrics(predictions, actual_values)
         detect_anomalies(predictions, threshold)
     ```
   - Call the main function when the script is executed:
     ```python
     if __name__ == "__main__":
         main()
     ```

This guide provides an overview of the initial items and relevant codebase for each module in the `src` directory. It includes sample code snippets and explanations for each component.