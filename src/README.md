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

- Splits the data into training, validation, and test sets using the `split_data()` function.
- Trains the machine learning model (e.g., Gradient Boosting Classifier) using the `train_model()` function.
- Evaluates the trained model using metrics such as mean squared error (MSE) and R-squared (R2) score using the `evaluate_model()` function.

The module utilizes the `scikit-learn` library for model training and evaluation.

## 5. Model Inference (`src/model_inference.py`)

The `model_inference.py` module is responsible for loading the trained model and making predictions on new data. It performs the following tasks:

- Loads the trained model from a file using the `load_model()` function.
- Preprocesses the input data using the same steps as in the training phase using the `preprocess_input_data()` function.
- Makes predictions using the loaded model and preprocessed data using the `predict()` function.

The module uses the `joblib` library for model serialization and deserialization.

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

## Detailed Code Snippets

### 1. Data Collection (`src/data_collection.py`)

#### Import the necessary libraries:

```python
import paho.mqtt.client as mqtt
import psycopg2
import json
```

#### Connect to the MQTT broker and subscribe to the specified topic:

```python
def connect_mqtt(broker_url, port, topic):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_url, port)
    client.subscribe(topic)
    client.loop_forever()
```

#### Callback functions for MQTT events:

```python
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")

def on_message(client, userdata, message):
    data = process_mqtt_message(message)
    store_data(data)
```

#### Process MQTT messages and store data in the database:

```python
def process_mqtt_message(message):
    # Extract relevant data from the MQTT message
    data = json.loads(message.payload)
    return data

def store_data(data):
    # Store the data in the database
    conn = psycopg2.connect(host="localhost", database="room_occupancy", user="postgres", password="mysecretpassword")
    cur = conn.cursor()
    cur.execute("INSERT INTO sensor_data (time, temperature, humidity, chassis_fan_speed) VALUES (%s, %s, %s, %s)",
                (data['timestamp'], data['temperature'], data['humidity'], data['chassis_fan_speed']))
    conn.commit()
    cur.close()
    conn.close()
```

### 2. Data Preprocessing (`src/data_preprocessing.py`)

#### Import the necessary libraries:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
```

#### Load data from the database:

```python
def load_data(start_date, end_date):
    conn = psycopg2.connect(host="localhost", database="room_occupancy", user="postgres", password="mysecretpassword")
    query = f"SELECT * FROM sensor_data WHERE time BETWEEN '{start_date}' AND '{end_date}'"
    data = pd.read_sql(query, conn)
    conn.close()
    return data
```

#### Handle missing values and remove outliers:

```python
def handle_missing_values(data):
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    return data

def remove_outliers(data):
    # Remove outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data
```

#### Perform feature scaling:

```python
def scale_features(data):
    scaler = StandardScaler()
    data[['temperature', 'humidity', 'chassis_fan_speed']] = scaler.fit_transform(data[['temperature', 'humidity', 'chassis_fan_speed']])
    return data
```

### 3. Feature Engineering (`src/feature_engineering.py`)

#### Import the necessary libraries:

```python
import pandas as pd
```

#### Create lag features:

```python
def create_lag_features(data, lag_periods):
    for period in lag_periods:
        data[f'temp_lag_{period}'] = data['temperature'].shift(period)
        data[f'humidity_lag_{period}'] = data['humidity'].shift(period)
    data.fillna(method='bfill', inplace=True)
    return data
```

#### Create rolling statistics features:

```python
def create_rolling_features(data, window_size):
    data['temp_rolling_mean'] = data['temperature'].rolling(window=window_size).mean()
    data['humidity_rolling_mean'] = data['humidity'].rolling(window=window_size).mean()
    data.fillna(method='bfill', inplace=True)
    return data
```

#### Create time-based features:

```python
def create_time_features(data):
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'] >= 5
    return data
```

### 4. Model Training (`src/model_training.py`)

#### Import the necessary libraries:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score
```

#### Split the data into training, validation, and test sets:

```python
def split_data(data, target_variable, test_size=0.2):
    X = data.drop(columns=[target_variable])
   

 y = data[target_variable]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

#### Train the machine learning model:

```python
def train_model(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model
```

#### Evaluate the trained model:

```python
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    return mse, r2
```

### 5. Model Inference (`src/model_inference.py`)

#### Import the necessary libraries:

```python
import joblib
```

#### Load the trained model:

```python
def load_model(model_path):
    model = joblib.load(model_path)
    return model
```

#### Preprocess input data for inference:

```python
def preprocess_input_data(data, scaler):
    data[['temperature', 'humidity', 'chassis_fan_speed']] = scaler.transform(data[['temperature', 'humidity', 'chassis_fan_speed']])
    return data
```

#### Make predictions using the loaded model:

```python
def predict(model, input_data):
    preprocessed_data = preprocess_input_data(input_data)
    predictions = model.predict(preprocessed_data)
    return predictions
```

### 6. Monitoring and Logging (`src/monitoring.py`)

#### Import the necessary libraries:

```python
import logging
from datetime import datetime
```

#### Log predictions:

```python
def log_prediction(prediction, timestamp):
    logging.info(f"Prediction: {prediction}, Timestamp: {timestamp}")
```

#### Calculate performance metrics:

```python
def calculate_performance_metrics(predictions, actual_values):
    accuracy = (predictions == actual_values).mean()
    logging.info(f"Accuracy: {accuracy}")
    return accuracy
```

#### Detect anomalies:

```python
def detect_anomalies(predictions, threshold):
    anomalies = predictions[predictions > threshold]
    if len(anomalies) > 0:
        logging.warning(f"Anomalies detected: {anomalies}")
    return anomalies
```

### 7. Main Script (`src/main.py`)

#### Import the necessary functions from the other modules:

```python
from data_collection import connect_mqtt
from data_preprocessing import load_data, handle_missing_values, remove_outliers, scale_features
from feature_engineering import create_lag_features, create_rolling_features, create_time_features
from model_training import split_data, train_model, evaluate_model
from model_inference import load_model, predict
from monitoring import log_prediction, calculate_performance_metrics, detect_anomalies
```

#### Implement the main function to orchestrate the flow of data through the pipeline:

```python
def main():
    # Data Collection
    connect_mqtt("localhost", 1883, "meraki/sensors")

    # Data Preprocessing
    data = load_data("2023-01-01", "2023-12-31")
    data = handle_missing_values(data)
    data = remove_outliers(data)
    data = scale_features(data)

    # Feature Engineering
    data = create_lag_features(data, [1, 2, 3])
    data = create_rolling_features(data, 60)
    data = create_time_features(data)

    # Model Training
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, 'occupancy', test_size=0.2)
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_val, y_val)
    print(f"Validation MSE: {mse}, R-squared: {r2}")

    # Model Inference
    model_path = 'trained_model.pkl'
    joblib.dump(model, model_path)
    model = load_model(model_path)
    input_data = load_data("2024-01-01", "2024-01-31")
    predictions = predict(model, input_data)

    # Monitoring and Logging
    timestamp = datetime.now()
    log_prediction(predictions, timestamp)
    accuracy = calculate_performance_metrics(predictions, input_data['occupancy'])
    anomalies = detect_anomalies(predictions, threshold=0.8)
    print(f"Accuracy: {accuracy}, Anomalies: {anomalies}")

if __name__ == "__main__":
    main()
```

This guide provides an overview of the initial items and relevant codebase for each module in the `src` directory. It includes sample code snippets and explanations for each component, ensuring that you can understand and implement each part of the data pipeline effectively.
