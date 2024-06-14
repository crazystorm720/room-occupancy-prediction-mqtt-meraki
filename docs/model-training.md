# Model Training Guide

## Overview
This document provides a step-by-step guide to train a machine learning model for predicting room occupancy using the preprocessed sensor data. The training process involves splitting the data, selecting the model, tuning hyperparameters, and evaluating the model's performance.

## Prerequisites
Before you start, ensure you have the following installed on your system:
- Python 3.7 or higher
- `pip` (Python package installer)
- Required Python packages (`pandas`, `numpy`, `scikit-learn`, `joblib`)

Install the necessary Python packages:
```bash
pip install pandas numpy scikit-learn joblib
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

## Data Splitting

### Step 2: Split Data into Training, Validation, and Test Sets
Split the data into training, validation, and test sets:

```python
from sklearn.model_selection import train_test_split

# Define features and target variable
features = ['temperature', 'humidity', 'chassis_fan_speed', 'temp_humidity_index', 'hour', 'day_of_week', 'is_weekend']
target = 'occupancy'  # Example target variable

X = df[features]
y = df[target]

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

## Model Selection and Training

### Step 3: Initialize and Train the Gradient Boosting Model
Initialize the Gradient Boosting model and perform hyperparameter tuning:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Initialize the model
model = GradientBoostingClassifier()

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f'Best Hyperparameters: {grid_search.best_params_}')
```

## Model Evaluation

### Step 4: Evaluate the Model on Validation Set
Evaluate the trained model's performance using the validation set:

```python
from sklearn.metrics import accuracy_score, f1_score

# Predictions on validation set
y_val_pred = best_model.predict(X_val)

# Evaluation metrics
accuracy = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f'Validation Accuracy: {accuracy:.2f}')
print(f'Validation F1 Score: {f1:.2f}')
```

### Step 5: Evaluate the Model on Test Set
Evaluate the model's performance on the test set:

```python
# Predictions on test set
y_test_pred = best_model.predict(X_test)

# Evaluation metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test F1 Score: {test_f1:.2f}')
```

## Save the Trained Model

### Step 6: Save the Trained Model
Save the trained model to a file for later use in inference:

```python
import joblib

# Save the model
joblib.dump(best_model, 'trained_model.pkl')
print("Model saved to 'trained_model.pkl'")
```

## Conclusion

You have now completed the model training process for predicting room occupancy using the preprocessed sensor data. The steps include loading the data, splitting it into training, validation, and test sets, initializing and training the gradient boosting model, evaluating its performance, and saving the trained model.

For any issues or further customization, refer to the respective documentation for pandas, numpy, scikit-learn, and joblib.
