# Room Occupancy Prediction Using MQTT Sensor Data from Meraki

This project aims to predict the number of people in a room using sensor data from Meraki devices. We use MQTT for real-time data ingestion and focus on chassis fan speed and environmental sensors (temperature, humidity) to build a predictive model.

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

## Project Structure

The project structure is organized as follows:
```
room-occupancy-prediction/
├── data/
├── docs/
├── notebooks/
├── src/
├── tests/
└── ...
```

- The `data` directory contains the raw and processed data.
- The `docs` directory contains the project documentation and guides.
- The `notebooks` directory contains Jupyter notebooks for exploratory data analysis and experimentation.
- The `src` directory contains the source code for data collection, preprocessing, feature engineering, model training, inference, and monitoring.
- The `tests` directory contains unit tests for the project.

Refer to the [Project Structure Guide](docs/project-structure.md) for more details on each directory and file.

## Technologies Used
This project combines several cutting-edge technologies to achieve its goals:

1. **MQTT (Message Queuing Telemetry Transport):** MQTT is a lightweight messaging protocol that allows devices to communicate with each other in real-time. In this project, we use MQTT to collect sensor data from Meraki devices, such as chassis fan speed, temperature, and humidity.

2. **Meraki Sensors:** Meraki is a company that provides a range of networking and security products, including sensors. These sensors can measure various environmental factors and provide real-time data streams. We utilize Meraki sensors to gather the necessary data for our room occupancy prediction model.

3. **TimescaleDB:** TimescaleDB is a time-series database that is optimized for storing and querying large amounts of time-stamped data. We use TimescaleDB to store the sensor data collected via MQTT, enabling efficient data retrieval and aggregation for analysis and model training.

4. **Python:** Python is a versatile programming language widely used for data analysis, machine learning, and web development. In this project, we use Python to process and analyze the sensor data, build and train machine learning models, and create the necessary scripts for data collection, preprocessing, and inference.

5. **scikit-learn:** scikit-learn is a popular Python library for machine learning. It provides a wide range of tools and algorithms for data preprocessing, feature engineering, model training, and evaluation. We leverage scikit-learn to build and train our room occupancy prediction model.

6. **Docker:** Docker is a platform that allows us to package our application and its dependencies into containers. By using Docker, we can ensure that our project runs consistently across different environments, making it easier to deploy and scale.

## Skills and Capabilities
By working on this project, you will gain valuable skills and knowledge in several areas:

1. **Internet of Things (IoT):** You will learn how to collect and process data from IoT devices, such as sensors, using protocols like MQTT. This skill is applicable to a wide range of IoT projects, including smart home automation, industrial monitoring, and more.

2. **Data Analysis and Preprocessing:** You will develop skills in data analysis and preprocessing, including handling missing values, removing outliers, and performing feature engineering. These skills are essential for any data-related project, as they help in preparing the data for machine learning tasks.

3. **Machine Learning:** You will gain hands-on experience in building and training machine learning models using Python and scikit-learn. You will learn about different algorithms, evaluation metrics, and techniques for improving model performance. These skills are highly sought after in various domains, such as predictive analytics, recommendation systems, and anomaly detection.

4. **Real-time Data Processing:** You will learn how to process and analyze data in real-time using MQTT and TimescaleDB. Real-time data processing is crucial for applications that require immediate insights and actions, such as monitoring systems, fraud detection, and real-time personalization.

5. **Containerization and Deployment:** You will gain experience in containerizing applications using Docker. Containerization skills are valuable for deploying and scaling applications in cloud environments, ensuring consistent and reproducible deployments.

By combining these technologies and skills, you will be well-equipped to tackle a wide range of data-driven projects across different domains. Whether you're interested in IoT, data analysis, machine learning, or real-time systems, this project provides a solid foundation for further exploration and growth.

Feel free to explore the project repository and dive into the code to learn more about the implementation details. Happy learning and predicting!

---

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

---

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

---

# Room Occupancy Prediction Using MQTT Sensor Data from Meraki

## I. Introduction
   A. Project Objective
      1. Predict the number of people in a room using sensor data from Meraki devices
      2. Focus on chassis fan speed and environmental sensors (temperature, humidity)
      3. Use MQTT for real-time data ingestion
   B. Project Overview
      1. Data Collection
      2. Data Preprocessing
      3. Feature Engineering
      4. Model Training
      5. Model Inference
      6. Monitoring and Logging

## II. Technologies Used
   A. MQTT (Message Queuing Telemetry Transport)
      1. Lightweight messaging protocol
      2. Real-time communication between devices
      3. Used to collect sensor data from Meraki devices
   B. Meraki Sensors
      1. Meraki company and its products
      2. Sensors for measuring environmental factors
      3. Real-time data streams
   C. TimescaleDB
      1. Time-series database
      2. Optimized for storing and querying time-stamped data
      3. Used to store sensor data collected via MQTT
   D. Python
      1. Versatile programming language
      2. Used for data analysis, machine learning, and web development
      3. Used to process and analyze sensor data, build and train models
   E. scikit-learn
      1. Python library for machine learning
      2. Tools and algorithms for data preprocessing, feature engineering, model training, and evaluation
      3. Used to build and train the room occupancy prediction model
   F. Docker
      1. Platform for containerizing applications
      2. Ensures consistent project execution across different environments
      3. Facilitates deployment and scaling

## III. Skills and Capabilities
   A. Internet of Things (IoT)
      1. Collecting and processing data from IoT devices
      2. Using protocols like MQTT
      3. Applicable to various IoT projects
   B. Data Analysis and Preprocessing
      1. Handling missing values
      2. Removing outliers
      3. Performing feature engineering
   C. Machine Learning
      1. Building and training machine learning models
      2. Using Python and scikit-learn
      3. Learning algorithms, evaluation metrics, and performance improvement techniques
   D. Real-time Data Processing
      1. Processing and analyzing data in real-time
      2. Using MQTT and TimescaleDB
      3. Crucial for applications requiring immediate insights and actions
   E. Containerization and Deployment
      1. Containerizing applications using Docker
      2. Deploying and scaling applications in cloud environments
      3. Ensuring consistent and reproducible deployments

## IV. Project Structure
   A. Data Directory
      1. Raw data
      2. Processed data
      3. Trained models
   B. Source Code Directory
      1. Data collection script
      2. Data preprocessing script
      3. Feature engineering script
      4. Model training script
      5. Model inference script
      6. Monitoring script
   C. Notebooks Directory
      1. Jupyter notebooks for exploratory data analysis and experimentation
   D. Tests Directory
      1. Unit tests for the project
   E. Configuration Files
      1. README.md
      2. PROJECT_SETUP.md
      3. requirements.txt
      4. Dockerfile

## V. Project Setup
   A. Instructions in PROJECT_SETUP.md
   B. Using Docker, Conda, Python, and Git

## VI. Data Collection
   A. Real-time Data Ingestion
      1. Tools: MQTT broker, Meraki sensors
      2. Process: Collecting real-time data on fan speed, temperature, and humidity
      3. Implementation: Functions in `data_collection.py`
   B. Batch Data Collection
      1. Tools: TimescaleDB
      2. Process: Aggregating data over specified periods for batch inference
      3. Implementation: Functions in `data_collection.py`

## VII. Data Preprocessing
   A. Steps
      1. Handling missing values
      2. Removing outliers
      3. Feature engineering (lag features, rolling statistics, time-based features)
      4. Normalization and scaling
   B. Implementation: Functions in `data_preprocessing.py`

## VIII. Feature Engineering
   A. Lag Features
      1. Capture temporal dependencies
   B. Rolling Statistics
      1. Calculate moving averages and other rolling metrics
   C. Time-Based Features
      1. Extract features like hour of the day and day of the week
   D. Interaction Terms
      1. Combine features to capture interactions
   E. Implementation: Functions in `feature_engineering.py`

## IX. Model Training
   A. Training Process
      1. Training Set: Used to train the model
      2. Validation Set: Used to tune hyperparameters and avoid overfitting
      3. Test Set: Used to evaluate the final model's performance
      4. Tools: scikit-learn, hyperparameter tuning libraries
   B. Metrics for Evaluation
      1. Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared
   C. Implementation: Functions in `model_training.py`

## X. Model Inference
   A. Real-time Inference
      1. Collect real-time data via MQTT
      2. Preprocess data and apply the trained model to make predictions
      3. Implementation: Functions in `model_inference.py`
   B. Batch Inference
      1. Aggregate data over a period, preprocess, and predict
      2. Implementation: Functions in `model_inference.py`

## XI. Monitoring and Logging
   A. Log Predictions
      1. Store predictions for future analysis and auditing
   B. Monitor Performance
      1. Track model performance over time to detect drift and degradation
   C. Implementation: Functions in `monitoring.py`

## XII. Advanced Techniques
   A. Feature Selection
      1. Identify the most important features to reduce dimensionality
   B. Ensemble Methods
      1. Combine predictions from multiple models to improve accuracy
   C. Cross-Validation
      1. Use cross-validation techniques to ensure generalization

## XIII. Example Project Walkthrough
   A. Objective: Predicting room occupancy using chassis fan speed, temperature, and humidity
   B. Steps
      1. Data Collection
      2. Data Preprocessing
      3. Feature Engineering
      4. Model Training and Evaluation
      5. Real-time and Batch Inference
      6. Monitoring and Logging

## XIV. References
   A. scikit-learn Documentation
   B. MQTT Protocol
   C. TimescaleDB Documentation
   D. Meraki Sensors

## XV. Getting Started
   A. Prerequisites
      1. Docker
      2. Conda
      3. Python
      4. Git
   B. Installation
      1. Clone the project repository
      2. Set up the project environment using Docker and Conda
      3. Refer to the Installation Guide
   C. Data Collection
      1. Set up the data collection pipeline
      2. Ingest real-time sensor data from Meraki devices using MQTT
      3. Refer to the Data Collection Guide
   D. Data Preprocessing
      1. Preprocess the collected data using provided scripts and guidelines
      2. Refer to the Data Preprocessing Guide
   E. Feature Engineering
      1. Generate new features from the preprocessed data
      2. Use feature engineering techniques outlined in the Feature Engineering Guide
   F. Model Training
      1. Train machine learning models using the preprocessed data
      2. Evaluate model performance
      3. Refer to the Model Training Guide
   G. Model Inference
      1. Apply trained models to make real-time predictions and perform batch inference
      2. Refer to the Model Inference Guide
   H. Monitoring and Logging
      1. Set up monitoring and logging mechanisms
      2. Track model performance and detect anomalies
      3. Refer to the Monitoring and Logging Guide

This outline organizes the information into a logical structure, starting with an introduction to the project and its objectives. It then covers the technologies used, the skills and capabilities gained through the project, and the project structure.
