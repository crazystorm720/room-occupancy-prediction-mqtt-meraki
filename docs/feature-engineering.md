# Feature Engineering Guide

## Overview
This document provides a comprehensive guide to feature engineering for the Room Occupancy Prediction project. Feature engineering involves creating new features from the raw sensor data to improve the performance of machine learning models.

## Prerequisites
Before you start, ensure you have the following installed on your system:
- Python 3.7 or higher
- `pip` (Python package installer)
- Required Python packages (`pandas`, `numpy`)

Install the necessary Python packages:
```bash
pip install pandas numpy
```

## Loading Preprocessed Data

### Step 1: Load Preprocessed Data
Load the preprocessed sensor data from the CSV file:

```python
import pandas as pd

# Load preprocessed data
df = pd.read_csv('preprocessed_sensor_data.csv')

# Display the first few rows of the DataFrame
print(df.head())
```

## Time-Based Features

### Step 2: Create Time-Based Features
Create new features based on the timestamp column to capture temporal dependencies:

```python
# Convert timestamp to datetime
df['time'] = pd.to_datetime(df['time'])

# Create hour of the day feature
df['hour'] = df['time'].dt.hour

# Create day of the week feature
df['day_of_week'] = df['time'].dt.dayofweek

# Create weekend feature
df['is_weekend'] = df['day_of_week'] >= 5

# Display the first few rows with new features
print(df[['time', 'hour', 'day_of_week', 'is_weekend']].head())
```

## Aggregated Features

### Step 3: Create Aggregated Features
Aggregate sensor readings over specific time intervals to capture trends and patterns:

```python
# Calculate rolling mean for temperature over 1-hour window
df['temp_rolling_mean_1h'] = df['temperature'].rolling(window=60).mean()

# Calculate rolling mean for humidity over 1-hour window
df['humidity_rolling_mean_1h'] = df['humidity'].rolling(window=60).mean()

# Fill NaN values resulting from rolling operation
df.fillna(method='bfill', inplace=True)

# Display the first few rows with new aggregated features
print(df[['temperature', 'temp_rolling_mean_1h', 'humidity', 'humidity_rolling_mean_1h']].head())
```

## Interaction Features

### Step 4: Create Interaction Features
Create interaction terms between features to capture relationships between them:

```python
# Create temperature-humidity interaction term
df['temp_humidity_index'] = df['temperature'] * 0.55 + df['humidity'] * 0.45

# Display the first few rows with the new interaction feature
print(df[['temperature', 'humidity', 'temp_humidity_index']].head())
```

## Lag Features

### Step 5: Create Lag Features
Create lag features to capture temporal dependencies in the data:

```python
# Create lag feature for temperature with a lag of 1 hour
df['temp_lag_1h'] = df['temperature'].shift(60)

# Create lag feature for humidity with a lag of 1 hour
df['humidity_lag_1h'] = df['humidity'].shift(60)

# Fill NaN values resulting from lag operation
df.fillna(method='bfill', inplace=True)

# Display the first few rows with new lag features
print(df[['temperature', 'temp_lag_1h', 'humidity', 'humidity_lag_1h']].head())
```

## Feature Selection

### Step 6: Select Features for Model Training
Select the relevant features for model training, including the newly created features:

```python
# Define the list of features to be used for model training
selected_features = [
    'temperature', 'humidity', 'chassis_fan_speed', 'hour', 'day_of_week', 'is_weekend',
    'temp_rolling_mean_1h', 'humidity_rolling_mean_1h', 'temp_humidity_index', 'temp_lag_1h', 'humidity_lag_1h'
]

# Create a DataFrame with the selected features
df_selected = df[selected_features]

# Display the first few rows of the DataFrame with selected features
print(df_selected.head())
```

## Save Engineered Features

### Step 7: Save the Data with Engineered Features
Save the DataFrame with engineered features to a new CSV file for model training:

```python
# Save the DataFrame with selected features
df_selected.to_csv('engineered_features_data.csv', index=False)

print("Engineered features saved to 'engineered_features_data.csv'")
```

## Conclusion

You have now completed the feature engineering process for the Room Occupancy Prediction project. The steps include creating time-based features, aggregated features, interaction features, lag features, and selecting the relevant features for model training. The engineered features are saved to a new CSV file, ready for training machine learning models.

For any issues or further customization, refer to the respective documentation for pandas and numpy.
