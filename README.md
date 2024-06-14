# Room Occupancy Prediction Using MQTT Sensor Data from Meraki

This project aims to predict room occupancy using sensor data from Meraki devices. MQTT is used for real-time data ingestion, focusing on chassis fan speed and environmental sensors (temperature, humidity) to build a predictive model using gradient boosting techniques.

## Project Overview

The Room Occupancy Prediction project encompasses the following key components:
1. **Data Collection**: Collect real-time sensor data from Meraki devices using MQTT and store it in a database for further analysis.
2. **Data Preprocessing**: Clean and preprocess the collected data, handle missing values, remove outliers, and perform feature scaling.
3. **Feature Engineering**: Create new features based on the raw data to capture temporal dependencies, calculate rolling statistics, extract time-based features, and create interaction terms.
4. **Model Training**: Train machine learning models using the preprocessed data and evaluate their performance using appropriate metrics.
5. **Model Inference**: Apply the trained models to make real-time predictions on new sensor data and perform batch inference on historical data.
6. **Monitoring and Logging**: Monitor the model's performance over time, log predictions for auditing and analysis, and detect anomalies or drift.

## Getting Started

To get started with the Room Occupancy Prediction project, follow these steps:

1. **Prerequisites**: Ensure that you have Docker, Conda, Python, and Git installed on your system.
2. **Installation**: Clone the project repository and set up the project environment using Docker and Conda. Refer to the [Installation Guide](docs/installation.md) for detailed instructions.
3. **Data Collection**: Set up the data collection pipeline to ingest real-time sensor data from Meraki devices using MQTT. Refer to the [Data Collection Guide](docs/data-collection.md) for more information.
4. **Data Preprocessing**: Preprocess the collected data using the provided scripts and guidelines. Refer to the [Data Preprocessing Guide](docs/data-preprocessing.md) for details.
5. **Feature Engineering**: Generate new features from the preprocessed data using the feature engineering techniques outlined in the [Feature Engineering Guide](docs/feature-engineering.md).
6. **Model Training**: Train machine learning models using the preprocessed data and evaluate their performance. Refer to the [Model Training Guide](docs/model-training.md) for instructions.
7. **Model Inference**: Apply the trained models to make real-time predictions and perform batch inference. Refer to the [Model Inference Guide](docs/model-inference.md) for more information.
8. **Monitoring and Logging**: Set up monitoring and logging mechanisms to track the model's performance and detect anomalies. Refer to the [Monitoring and Logging Guide](docs/monitoring-logging.md) for details.

## Technologies Used

1. **MQTT**: Lightweight messaging protocol for real-time data collection from Meraki sensors.
2. **Meraki Sensors**: Environmental sensors providing real-time data streams.
3. **TimescaleDB**: Time-series database optimized for storing and querying time-stamped data.
4. **Python**: Programming language for data processing, analysis, and machine learning model development.
5. **scikit-learn**: Python library for machine learning, used for building and training models.
6. **Docker**: Platform for containerizing applications to ensure consistent environments across deployments.

## Skills and Capabilities

By working on this project, you will gain valuable skills in:
1. **Internet of Things (IoT)**: Collecting and processing data from IoT devices using MQTT.
2. **Data Analysis and Preprocessing**: Handling missing values, removing outliers, and performing feature engineering.
3. **Machine Learning**: Building and training models using Python and scikit-learn.
4. **Real-time Data Processing**: Analyzing data in real-time using MQTT and TimescaleDB.
5. **Containerization and Deployment**: Containerizing applications with Docker for consistent deployments.

## Project Structure

```
room-occupancy-prediction/
├── data/
├── docs/
│   ├── installation.md
│   ├── data-collection.md
│   ├── data-preprocessing.md
│   ├── feature-engineering.md
│   ├── model-training.md
│   ├── model-inference.md
│   ├── monitoring-logging.md
│   └── project-structure.md
├── notebooks/
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_inference.py
│   ├── monitoring.py
│   ├── utils.py
│   └── main.py
├── tests/
└── README.md
```

- `data`: Contains raw and processed data.
- `docs`: Project documentation and guides.
- `notebooks`: Jupyter notebooks for data exploration and experimentation.
- `src`: Source code for data collection, preprocessing, feature engineering, model training, inference, and monitoring.
- `tests`: Unit tests for the project.

Refer to the [Project Structure Guide](docs/project-structure.md) for more details.

## Project Setup

To set up the project, follow the instructions in the [PROJECT_SETUP.md](PROJECT_SETUP.md) file. It provides a step-by-step guide to configure the project using Docker, Conda, Python, and Git.

## Data Collection

### Real-time Data Ingestion
- **Tools**: MQTT broker, Meraki sensors.
- **Process**: Collect real-time data on fan speed, temperature, and humidity.
- **Implementation**: The `data_collection.py` file contains functions to connect to the MQTT broker, process MQTT messages, and store the collected data in a database.

### Batch Data Collection
- **Tools**: TimescaleDB.
- **Process**: Aggregating data over specified periods for batch inference.
- **Implementation**: The `data_collection.py` file includes functions to retrieve data from the database for a specified date range.

## Data Preprocessing

### Steps
- Handling missing values.
- Removing outliers.
- Feature engineering (lag features, rolling statistics, time-based features).
- Normalization and scaling.
- **Implementation**: The `data_preprocessing.py` file contains functions to load data, handle missing values, remove outliers, and perform feature scaling.

## Feature Engineering

- **Lag Features**: Capture temporal dependencies.
- **Rolling Statistics**: Calculate moving averages and other rolling metrics.
- **Time-Based Features**: Extract features like hour of the day and day of the week.
- **Interaction Terms**: Combine features to capture interactions.
- **Implementation**: The `feature_engineering.py` file includes functions to create lag features, rolling statistics, time-based features, and interaction terms.

## Model Training

### Training Process
- **Training Set**: Used to train the model.
- **Validation Set**: Used to tune hyperparameters and avoid overfitting.
- **Test Set**: Used to evaluate the final model's performance.
- **Tools**: scikit-learn, hyperparameter tuning libraries.
- **Implementation**: The `model_training.py` file contains functions to split the data, train the machine learning model, and evaluate its performance.

### Metrics for Evaluation
- Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared.

## Model Inference

### Real-time Inference
- Collect real-time data via MQTT.
- Preprocess data and apply the trained model to make predictions.
- **Implementation**: The `model_inference.py` file includes functions to load the trained model, preprocess input data, and make predictions in real-time.

### Batch Inference
- Aggregate data over a period, preprocess, and predict.
- **Implementation**: The `model_inference.py` file also contains functions to perform batch inference on aggregated data.

## Monitoring and Logging

- **Log Predictions**: Store predictions for future analysis and auditing.
- **Monitor Performance**: Track model performance over time to detect drift and degradation.
- **Implementation**: The `monitoring.py` file includes functions to log predictions, calculate performance metrics, and detect anomalies.

## Advanced Techniques

- **Feature Selection**: Identify the most important features to reduce dimensionality.
- **Ensemble Methods**: Combine predictions from multiple models to improve accuracy.
- **Cross-Validation**: Use cross-validation techniques to ensure generalization.

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

---

# Machine Learning Model Training Workflow for Meraki MQTT Sensor Data Using Gradient Boosting

## 1. Data Collection and Preprocessing

### Collect Data
```python
import pandas as pd

# Assuming data is stored in CSV files
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
```

### Data Cleaning
```python
# Handle missing values
train_data.fillna(method='ffill', inplace=True)
test_data.fillna

(method='ffill', inplace=True)

# Convert timestamps to datetime format
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])

# Normalize sensor values (if necessary)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[['temperature', 'humidity']] = scaler.fit_transform(train_data[['temperature', 'humidity']])
test_data[['temperature', 'humidity']] = scaler.transform(test_data[['temperature', 'humidity']])
```

### Data Exploration
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Explore data patterns
sns.lineplot(data=train_data, x='timestamp', y='temperature')
plt.show()
```

## 2. Problem Definition and Feature Engineering

### Define Problem
- **Example Problem**: Predicting room occupancy based on sensor data.

### Feature Engineering
```python
# Create time-based features
train_data['hour'] = train_data['timestamp'].dt.hour
train_data['day_of_week'] = train_data['timestamp'].dt.dayofweek
train_data['is_weekend'] = train_data['day_of_week'] >= 5

test_data['hour'] = test_data['timestamp'].dt.hour
test_data['day_of_week'] = test_data['timestamp'].dt.dayofweek
test_data['is_weekend'] = test_data['day_of_week'] >= 5

# Aggregate sensor readings
train_data['temp_humidity_index'] = train_data['temperature'] * 0.55 + train_data['humidity'] * 0.45
test_data['temp_humidity_index'] = test_data['temperature'] * 0.55 + test_data['humidity'] * 0.45
```

## 3. Data Splitting and Model Selection

### Data Splitting
```python
from sklearn.model_selection import train_test_split

# Define features and target
features = ['temperature', 'humidity', 'hour', 'day_of_week', 'is_weekend', 'temp_humidity_index']
target = 'occupancy'  # Example target variable

X = train_data[features]
y = train_data[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Selection
- **Model**: Gradient Boosting Classifier

## 4. Model Training and Evaluation

### Train Models
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Initialize model
model = GradientBoostingClassifier()

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f'Best Hyperparameters: {grid_search.best_params_}')
```

### Evaluate Models
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

## 5. Model Testing and Deployment

### Test Model
```python
# Predictions on test set
X_test = test_data[features]
y_test_pred = best_model.predict(X_test)

# Prepare submission (if applicable)
submission = pd.DataFrame({'timestamp': test_data['timestamp'], 'occupancy': y_test_pred})
submission.to_csv('submission.csv', index=False)
```

## 6. Monitoring and Continuous Improvement

### Monitor Performance
```python
# Assuming real-time data comes in
# new_data = ...  # Collect new real-time data

# Process new data similarly
# new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
# new_data[['temperature', 'humidity']] = scaler.transform(new_data[['temperature', 'humidity']])
# new_data['hour'] = new_data['timestamp'].dt.hour
# new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek
# new_data['is_weekend'] = new_data['day_of_week'] >= 5
# new_data['temp_humidity_index'] = new_data['temperature'] * 0.55 + new_data['humidity'] * 0.45

# Predictions on new data
# new_predictions = best_model.predict(new_data[features])
```

### Continuous Improvement
```python
# Periodically retrain the model with new data
# updated_train_data = pd.concat([train_data, new_data])
# X_updated = updated_train_data[features]
# y_updated = updated_train_data[target]

# best_model.fit(X_updated, y_updated)
```

By following this detailed and code-focused workflow, you can effectively train, evaluate, and deploy gradient boosting models tailored for Meraki MQTT sensor data. This structured approach ensures that each step is well-defined and thoroughly documented, providing a clear path from data collection to model deployment and continuous improvement.

---

# Project Logic Overview

## 1. Data Collection

### Real-time Data Ingestion

- **Objective**: Collect real-time sensor data from Meraki devices using MQTT.
- **Workflow**:
  1. **Connect to MQTT Broker**: Establish a connection to the MQTT broker using `paho-mqtt`.
  2. **Subscribe to Topic**: Subscribe to the specified MQTT topic to receive sensor data.
  3. **Handle Incoming Messages**: Process incoming MQTT messages to extract relevant data (e.g., temperature, humidity, chassis fan speed).
  4. **Store Data**: Save the extracted data into TimescaleDB for further processing.

### Batch Data Collection

- **Objective**: Collect historical data for model training and evaluation.
- **Workflow**:
  1. **Retrieve Data**: Query TimescaleDB to retrieve sensor data for specified date ranges.
  2. **Store Data**: Save the retrieved data locally in CSV format for preprocessing and feature engineering.

## 2. Data Preprocessing

- **Objective**: Clean and preprocess the collected sensor data to make it suitable for model training.
- **Workflow**:
  1. **Load Data**: Load data from TimescaleDB or CSV files.
  2. **Handle Missing Values**: Use forward fill (`ffill`) method to handle missing values.
  3. **Remove Outliers**: Apply statistical methods (e.g., IQR) to remove outliers.
  4. **Normalize and Scale**: Normalize and scale sensor values using `StandardScaler` from `scikit-learn`.

## 3. Feature Engineering

- **Objective**: Create new features from the raw data to improve model performance.
- **Workflow**:
  1. **Create Time-based Features**: Extract features such as hour of the day, day of the week, and weekend indicator.
  2. **Generate Lag Features**: Create lagged versions of the sensor data to capture temporal dependencies.
  3. **Calculate Rolling Statistics**: Compute rolling averages and other statistics over defined time windows.
  4. **Interaction Terms**: Create interaction terms like temperature-humidity index.

## 4. Model Training

- **Objective**: Train machine learning models to predict room occupancy based on the engineered features.
- **Workflow**:
  1. **Data Splitting**: Split the preprocessed data into training, validation, and test sets.
  2. **Model Selection**: Choose an appropriate machine learning model (e.g., Gradient Boosting Classifier).
  3. **Hyperparameter Tuning**: Use techniques like GridSearchCV to find the best hyperparameters.
  4. **Model Training**: Train the model using the training set.
  5. **Model Evaluation**: Evaluate the model on the validation set using metrics like accuracy and F1 score.

## 5. Model Inference

### Real-time Inference

- **Objective**: Use the trained model to make real-time predictions on new sensor data.
- **Workflow**:
  1. **Receive New Data**: Collect new sensor data via MQTT in real-time.
  2. **Preprocess Data**: Apply the same preprocessing steps used during training.
  3. **Make Predictions**: Use the trained model to predict room occupancy based on the new data.

### Batch Inference

- **Objective**: Perform batch predictions on historical data for analysis and evaluation.
- **Workflow**:
  1. **Load Historical Data**: Load historical sensor data from TimescaleDB.
  2. **Preprocess Data**: Apply the same preprocessing steps used during training.
  3. **Make Predictions**: Use the trained model to predict room occupancy on the historical data.

## 6. Monitoring and Logging

- **Objective**: Monitor the model's performance over time, log predictions, and detect anomalies.
- **Workflow**:
  1. **Log Predictions**: Store model predictions along with timestamps for auditing and analysis.
  2. **Performance Monitoring**: Calculate performance metrics (e.g., accuracy, F1 score) on new predictions.
  3. **Anomaly Detection**: Identify anomalies in predictions based on predefined thresholds.

## Integration and Flow

### Main Script (`src/main.py`)

- **Objective**: Orchestrate the entire workflow from data collection to model inference and monitoring.
- **Workflow**:
  1. **Data Collection**: Call functions from `data_collection.py` to start real-time data ingestion.
  2. **Data Preprocessing**: Preprocess the collected data using functions from `data_preprocessing.py`.
  3. **Feature Engineering**: Generate new features using functions from `feature_engineering.py`.
  4. **Model Training**: Train and evaluate the model using functions from `model_training.py`.
  5. **Model Inference**: Perform real-time and batch inference using functions from `model_inference.py`.
  6. **Monitoring and Logging**: Log predictions and monitor performance using functions from `monitoring.py`.

### Configuration (`config.yml`)

- **Objective**: Centralize configuration settings for easy management and modification.
- **Configuration Options**:
  - MQTT broker details.
  - Database connection settings.
  - Meraki sensor information.
  - Hyperparameters for model training.

### Example Workflow

1. **Initialize and Configure**: Start by configuring the MQTT broker, database, and Meraki sensors in `config.yml`.
2. **Start Data Collection**: Run the main script to initiate real-time data collection from Meraki sensors via MQTT.
3. **Preprocess Data**: Periodically preprocess the collected data to handle missing values and remove outliers.
4. **Generate Features**: Apply feature engineering techniques to create new features from the preprocessed data.
5. **Train Model**: Train the Gradient Boosting model using the engineered features and evaluate its performance.
6. **Make Predictions**: Use the trained model to make real-time predictions on new sensor data and batch predictions on historical data.
7. **Monitor and Log**: Continuously monitor the model's performance, log predictions, and detect any anomalies.

By following this detailed and logical workflow, you can ensure that each component of the Room Occupancy Prediction project is accounted for and integrated seamlessly. This modular approach allows for easy maintenance, scalability, and potential enhancements in the future.

---

# Gradient Boosting Overview

## Introduction
Gradient Boosting is a machine learning technique used for regression and classification problems, which builds a model in a stage-wise fashion from an ensemble of weak learners (usually decision trees).

## Key Concepts

### Weak Learners
- **Weak Learners** are models that perform slightly better than random guessing.
- In Gradient Boosting, decision trees with limited depth (often called stumps) are used as weak learners.

### Boosting
- **Boosting** is an ensemble technique that combines multiple weak learners to form a strong learner.
- The key idea is to add new models to correct the errors made by the previous models.

## How Gradient Boosting Works

1. **Initialize the model** with a constant value (usually the mean of the target values for regression).
2. **Fit a weak learner** to the residuals (errors) of the current model.
3. **Update the model** by adding the new weak learner to the ensemble.
4. **Repeat** steps 2 and 3 for a specified number of iterations or until a stopping criterion is met.

## Components of Gradient Boosting

### Loss Function
- The loss function measures the difference between the predicted and actual values.
- Gradient Boosting aims to minimize the loss function.

### Learning Rate
- The learning rate (shrinkage) controls the contribution of each weak learner.
- Lower learning rates require more trees but can lead to better performance.

### Number of Trees
- The number of trees (iterations) determines the number of weak learners in the ensemble.
- More trees can improve accuracy but also increase the risk of overfitting.

### Tree Depth
- The depth of the trees controls the complexity of the weak learners.
- Shallow trees (stumps) are often used to avoid overfitting.

## Gradient Boosting Algorithm

1. Initialize model with a constant value:
   \( F_0(x) = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, \gamma) \)

2. For \( m = 1 \) to \( M \) (number of trees):
   1. Compute the negative gradient (pseudo-residuals):
      \( r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x)=F_{m-1}(x)} \)
   2. Fit a weak learner \( h_m(x) \) to the pseudo-residuals.
   3. Compute the step size \( \gamma_m \):
      \( \gamma_m = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)) \)
   4. Update the model:
      \( F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) \)

## Advantages of Gradient Boosting

- **High Accuracy**: Gradient Boosting often achieves high predictive accuracy.
- **Flexibility**: It can be used with various loss functions and is adaptable to both regression and classification tasks.
- **Feature Importance**: Provides insights into the importance of features.

## Disadvantages of Gradient Boosting

- **Computationally Intensive**: Training can be time-consuming, especially with a large number of trees.
- **Prone to Overfitting**: Careful tuning of hyperparameters is necessary to avoid overfitting.
- **Parameter Sensitivity**: Requires careful tuning of parameters like learning rate, number of trees, and tree depth.

## Popular Gradient Boosting Libraries

- **XGBoost**: Known for its speed and performance.
- **LightGBM**: Optimized for efficiency and scalability.
- **CatBoost**: Handles categorical features automatically and reduces the need for extensive preprocessing.
- **Scikit-learn**: Provides a simple implementation for basic use cases.

## Example in Python using Scikit-learn

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Sample data
X_train, X_test, y_train, y_test = ...

# Initialize the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Fit the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

## Conclusion
Gradient Boosting is a powerful and flexible machine learning technique suitable for various tasks. By understanding its key concepts, components, and algorithm, one can effectively apply it to solve complex problems.
