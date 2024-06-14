# Data Preprocessing Guide

## Overview
This document provides a step-by-step guide to preprocess the sensor data collected from Meraki devices. The preprocessing steps include handling missing values, removing outliers, feature engineering, normalization, and scaling to prepare the data for machine learning model training.

## Prerequisites
Before you start, ensure you have the following installed on your system:
- Python 3.7 or higher
- `pip` (Python package installer)
- Required Python packages (`pandas`, `numpy`, `scikit-learn`)

Install the necessary Python packages:
```bash
pip install pandas numpy scikit-learn
```

## Data Loading

### Step 1: Load Data from TimescaleDB
You can use the following script to load data from TimescaleDB into a pandas DataFrame:

```python
import pandas as pd
import psycopg2

# TimescaleDB settings
DB_HOST = "localhost"
DB_NAME = "room_occupancy"
DB_USER = "postgres"
DB_PASS = "mysecretpassword"

# Connect to TimescaleDB
conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()

# Load data into pandas DataFrame
query = "SELECT * FROM sensor_data;"
df = pd.read_sql(query, conn)
conn.close()

# Display the first few rows of the DataFrame
print(df.head())
```

## Data Cleaning

### Step 2: Handle Missing Values
Handle missing values by forward filling (`ffill`) method:
```python
# Fill missing values
df.fillna(method='ffill', inplace=True)
```

### Step 3: Convert Timestamps
Convert the timestamp column to datetime format:
```python
# Convert timestamp to datetime
df['time'] = pd.to_datetime(df['time'])
```

## Data Exploration

### Step 4: Data Exploration and Visualization
Visualize the data to understand patterns, trends, and anomalies:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize temperature data
sns.lineplot(data=df, x='time', y='temperature')
plt.show()

# Visualize humidity data
sns.lineplot(data=df, x='time', y='humidity')
plt.show()

# Visualize chassis fan speed data
sns.lineplot(data=df, x='time', y='chassis_fan_speed')
plt.show()
```

## Removing Outliers

### Step 5: Identify and Remove Outliers
Remove outliers using the interquartile range (IQR) method:

```python
# Define a function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for temperature, humidity, and chassis fan speed
df = remove_outliers(df, 'temperature')
df = remove_outliers(df, 'humidity')
df = remove_outliers(df, 'chassis_fan_speed')
```

## Feature Engineering

### Step 6: Create Time-Based Features
Create new features based on the timestamp column:

```python
# Create time-based features
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5
```

### Step 7: Aggregate Sensor Readings
Aggregate sensor readings over specific time intervals:

```python
# Aggregate sensor readings
df['temp_humidity_index'] = df['temperature'] * 0.55 + df['humidity'] * 0.45
```

## Normalization and Scaling

### Step 8: Normalize and Scale Features
Normalize and scale the sensor values:

```python
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Normalize and scale the features
df[['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index']] = scaler.fit_transform(df[['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index']])
```

## Save Preprocessed Data

### Step 9: Save Preprocessed Data
Save the preprocessed data to a new CSV file:

```python
# Save preprocessed data
df.to_csv('preprocessed_sensor_data.csv', index=False)
```

## Conclusion
You have now preprocessed the sensor data collected from Meraki devices. The steps include loading data from TimescaleDB, handling missing values, removing outliers, feature engineering, normalization, and scaling. This preprocessed data is now ready for machine learning model training.

For any issues or further customization, refer to the respective documentation for pandas, numpy, and scikit-learn.
