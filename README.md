# Room Occupancy Prediction

Welcome to the Room Occupancy Prediction project! This project aims to predict the number of people in a room using sensor data collected from Meraki devices via MQTT. By leveraging real-time data and machine learning techniques, we can create a system that provides valuable insights into room occupancy patterns.

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
