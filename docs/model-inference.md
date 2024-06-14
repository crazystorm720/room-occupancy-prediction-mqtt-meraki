# Model Inference Guide

## Overview
This document provides a step-by-step guide to perform model inference using the trained machine learning model for predicting room occupancy. The inference process includes making real-time predictions on new sensor data and performing batch inference on historical data.

## Prerequisites
Before you start, ensure you have the following installed on your system:
- Python 3.7 or higher
- `pip` (Python package installer)
- Required Python packages (`pandas`, `numpy`, `scikit-learn`, `paho-mqtt`)

Install the necessary Python packages:
```bash
pip install pandas numpy scikit-learn paho-mqtt
```

## Loading the Trained Model

### Step 1: Load the Preprocessed Data
Load the preprocessed sensor data from the CSV file:

```python
import pandas as pd

# Load preprocessed data
df = pd.read_csv('preprocessed_sensor_data.csv')
```

### Step 2: Load the Trained Model
Load the trained machine learning model using `joblib`:

```python
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')
```

## Real-time Inference

### Step 3: Set Up MQTT Client for Real-time Data
Set up the MQTT client to receive real-time sensor data and make predictions:

```python
import paho.mqtt.client as mqtt
import numpy as np

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "meraki/sensors"

# MQTT callback function
def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    timestamp = data['timestamp']
    temperature = data['temperature']
    humidity = data['humidity']
    chassis_fan_speed = data['chassis_fan_speed']
    
    # Preprocess the data
    temp_humidity_index = temperature * 0.55 + humidity * 0.45
    hour = pd.to_datetime(timestamp).hour
    day_of_week = pd.to_datetime(timestamp).dayofweek
    is_weekend = day_of_week >= 5
    
    # Create a DataFrame for the new data point
    new_data = pd.DataFrame([[temperature, humidity, chassis_fan_speed, temp_humidity_index, hour, day_of_week, is_weekend]],
                            columns=['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index', 'hour', 'day_of_week', 'is_weekend'])
    
    # Normalize the new data point using the same scaler
    new_data[['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index']] = scaler.transform(new_data[['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index']])
    
    # Make prediction
    prediction = model.predict(new_data)
    print(f"Prediction for {timestamp}: {prediction[0]}")

# Set up MQTT client
client = mqtt.Client()
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC)

client.loop_forever()
```

## Batch Inference

### Step 4: Perform Batch Inference on Historical Data
Load the historical data, preprocess it, and make batch predictions:

```python
# Load historical data
historical_data = pd.read_csv('historical_sensor_data.csv')

# Preprocess the historical data
historical_data['time'] = pd.to_datetime(historical_data['time'])
historical_data['hour'] = historical_data['time'].dt.hour
historical_data['day_of_week'] = historical_data['time'].dt.dayofweek
historical_data['is_week

```python
historical_data['is_weekend'] = historical_data['day_of_week'] >= 5
historical_data['temp_humidity_index'] = historical_data['temperature'] * 0.55 + historical_data['humidity'] * 0.45

# Normalize the features using the same scaler
historical_data[['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index']] = scaler.transform(historical_data[['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index']])

# Define features for prediction
features = ['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index', 'hour', 'day_of_week', 'is_weekend']

# Make batch predictions
predictions = model.predict(historical_data[features])

# Add predictions to the DataFrame
historical_data['predicted_occupancy'] = predictions

# Save the predictions to a CSV file
historical_data.to_csv('batch_inference_predictions.csv', index=False)

print("Batch inference completed. Predictions saved to 'batch_inference_predictions.csv'.")
```

## Conclusion

You have now set up both real-time and batch inference pipelines for predicting room occupancy using the trained machine learning model. The real-time inference uses MQTT to collect live sensor data and make immediate predictions, while the batch inference processes historical data to generate predictions for analysis.

For any issues or further customization, refer to the respective documentation for pandas, numpy, scikit-learn, and paho-mqtt.
