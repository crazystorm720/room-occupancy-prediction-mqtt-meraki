# Project Setup Guide: Room Occupancy Prediction

This document provides a step-by-step guide to set up the Room Occupancy Prediction project using Docker, Conda, Python, and Git.

## Prerequisites

- Docker installed on your machine
- Git installed on your machine

## Project Setup

1. **Create a new directory for your project:**
   ```
   mkdir room-occupancy-prediction
   cd room-occupancy-prediction
   ```

2. **Initialize a new Git repository:**
   ```
   git init
   ```

3. **Create a `.gitignore` file to exclude unnecessary files from version control:**
   ```
   touch .gitignore
   ```
   Open the `.gitignore` file and add the following lines to exclude common files and directories:
   ```
   .DS_Store
   .idea/
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   env/
   venv/
   .env
   ```

4. **Create a `Dockerfile` to define your Docker image:**
   ```
   touch Dockerfile
   ```
   Open the `Dockerfile` and add the following content:
   ```
   FROM continuumio/miniconda3

   # Create a working directory
   WORKDIR /app

   # Copy the environment.yml file to the working directory
   COPY environment.yml .

   # Create a new Conda environment
   RUN conda env create -f environment.yml

   # Activate the Conda environment
   RUN echo "conda activate room-occupancy-prediction-env" >> ~/.bashrc
   ENV PATH /opt/conda/envs/room-occupancy-prediction-env/bin:$PATH

   # Copy the project files to the working directory
   COPY . .

   # Set the default command to run when the container starts
   CMD ["python", "main.py"]
   ```

5. **Create a `environment.yml` file to define your Conda environment:**
   ```
   touch environment.yml
   ```
   Open the `environment.yml` file and add the following content:
   ```yaml
   name: room-occupancy-prediction-env
   dependencies:
     - python=3.9
     - numpy
     - pandas
     - scikit-learn
     - matplotlib
     - seaborn
     - paho-mqtt
     - psycopg2
   ```

6. **Create a `main.py` file as the entry point for your project:**
   ```
   touch main.py
   ```
   Open the `main.py` file and add a placeholder for your project code:
   ```python
   def main():
       # Your project code goes here
       print("Room Occupancy Prediction")

   if __name__ == "__main__":
       main()
   ```

7. **Create a `README.md` file with your project description:**
   ```
   touch README.md
   ```
   Copy the content you provided for the README file into this `README.md` file.

8. **Commit your changes to the Git repository:**
   ```
   git add .
   git commit -m "Initial project setup"
   ```

9. **Build the Docker image:**
   ```
   docker build -t room-occupancy-prediction .
   ```

10. **Run the Docker container:**
    ```
    docker run room-occupancy-prediction
    ```
    You should see the output "Room Occupancy Prediction" in the console.

## Next Steps

- Implement the various components of your project, such as data collection, preprocessing, feature engineering, model training, inference, and monitoring, within the Docker container.
- Update the `main.py` file with your project code.
- Create separate Python files for different functionalities.
- Modify the `environment.yml` file if you require additional dependencies.
- Document your progress and update the README file.
- Regularly commit your changes to the Git repository.

Happy coding and best of luck with your room occupancy prediction project!

1. **Data Collection:**
   - Create a new file named `data_collection.py` in your project directory.
   - Implement functions to collect real-time data from MQTT topics and store it in a database (e.g., TimescaleDB).
   - Example functions:
     - `connect_mqtt(broker_url, port, topic)`: Connect to the MQTT broker and subscribe to the specified topic.
     - `process_mqtt_message(message)`: Process the received MQTT message and extract relevant data.
     - `store_data(data)`: Store the collected data in the database.

2. **Data Preprocessing:**
   - Create a new file named `data_preprocessing.py` in your project directory.
   - Implement functions to preprocess the collected data, handle missing values, remove outliers, and perform feature scaling.
   - Example functions:
     - `load_data(start_date, end_date)`: Load the data from the database for a specified date range.
     - `handle_missing_values(data)`: Handle missing values in the data (e.g., interpolation, imputation).
     - `remove_outliers(data)`: Remove outliers from the data using statistical methods.
     - `scale_features(data)`: Perform feature scaling (e.g., normalization, standardization) on the data.

3. **Feature Engineering:**
   - Create a new file named `feature_engineering.py` in your project directory.
   - Implement functions to create new features based on the existing data, such as lag features, rolling statistics, and time-based features.
   - Example functions:
     - `create_lag_features(data, lag_periods)`: Create lag features by shifting the data by specified periods.
     - `create_rolling_features(data, window_size)`: Create rolling statistics features using a specified window size.
     - `create_time_features(data)`: Create time-based features, such as hour of the day and day of the week.

4. **Model Training:**
   - Create a new file named `model_training.py` in your project directory.
   - Implement functions to train the machine learning model using the preprocessed and engineered features.
   - Example functions:
     - `split_data(data, target_variable, test_size)`: Split the data into training and testing sets.
     - `train_model(X_train, y_train)`: Train the machine learning model using the training data.
     - `evaluate_model(model, X_test, y_test)`: Evaluate the trained model using the testing data.

5. **Model Inference:**
   - Create a new file named `model_inference.py` in your project directory.
   - Implement functions to perform real-time and batch inference using the trained model.
   - Example functions:
     - `load_model(model_path)`: Load the trained model from a file.
     - `preprocess_input_data(data)`: Preprocess the input data for inference.
     - `predict(model, input_data)`: Make predictions using the loaded model and preprocessed input data.

6. **Monitoring and Logging:**
   - Create a new file named `monitoring.py` in your project directory.
   - Implement functions to monitor the model's performance, log predictions, and detect anomalies.
   - Example functions:
     - `log_prediction(prediction, timestamp)`: Log the predicted values along with the timestamp.
     - `calculate_performance_metrics(predictions, actual_values)`: Calculate performance metrics, such as accuracy and F1 score.
     - `detect_anomalies(predictions, threshold)`: Detect anomalies in the predictions based on a specified threshold.

7. **Update the `main.py` file:**
   - Modify the `main.py` file to integrate the above components and orchestrate the flow of data through the pipeline.
   - Example flow:
     1. Collect real-time data using the functions from `data_collection.py`.
     2. Preprocess the collected data using the functions from `data_preprocessing.py`.
     3. Generate new features using the functions from `feature_engineering.py`.
     4. Train the model using the functions from `model_training.py`.
     5. Perform real-time or batch inference using the functions from `model_inference.py`.
     6. Monitor the model's performance and log predictions using the functions from `monitoring.py`.

8. **Update the `Dockerfile`:**
   - Modify the `Dockerfile` to include the necessary dependencies and commands to run the updated `main.py` file.

9. **Commit the changes:**
   - Add the newly created files to the Git repository and commit the changes.
   - Push the changes to the remote repository.
