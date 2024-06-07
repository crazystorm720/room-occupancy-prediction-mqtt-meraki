# Installation Guide

This guide provides step-by-step instructions for setting up and running the Room Occupancy Prediction project using Docker, Conda, Python, and Git.

## Prerequisites

Before proceeding with the installation, ensure that you have the following prerequisites installed on your system:

- Docker
- Git

## Installation Steps

1. Clone the repository:
   ```shell
   git clone https://github.com/crazystorm720/room-occupancy-prediction-mqtt-meraki.git
   ```

2. Navigate to the project directory:
   ```shell
   cd room-occupancy-prediction-mqtt-meraki
   ```

3. Create a `.gitignore` file to exclude unnecessary files from version control:
   ```shell
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

4. Create a `Dockerfile` to define your Docker image:
   ```shell
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
   CMD ["python", "src/main.py"]
   ```

5. Create an `environment.yml` file to define your Conda environment:
   ```shell
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

6. Create the project structure:
   ```shell
   mkdir -p data/raw data/processed data/models
   mkdir -p src tests notebooks
   touch src/__init__.py tests/__init__.py
   touch src/data_collection.py src/data_preprocessing.py src/feature_engineering.py
   touch src/model_training.py src/model_inference.py src/monitoring.py
   touch src/main.py
   touch notebooks/exploratory_analysis.ipynb
   ```

7. Set up the configuration:
   - Create a copy of the `config.example.yml` file and rename it to `config.yml`.
   - Open the `config.yml` file and update the configuration settings according to your environment, such as MQTT broker details, database credentials, and Meraki sensor information.

8. Build the Docker image:
   ```shell
   docker build -t room-occupancy-prediction .
   ```

9. Run the Docker container:
   ```shell
   docker run -d --name room-occupancy-prediction -v /path/to/config.yml:/app/config.yml room-occupancy-prediction
   ```
   Replace `/path/to/config.yml` with the actual path to your `config.yml` file.

   This command starts the Docker container in detached mode (`-d`) and mounts the `config.yml` file from your local system to the container's `/app/config.yml` path.

10. Access the project:
    - The project will start collecting data from the configured MQTT broker and Meraki sensors.
    - You can access the project's web interface (if provided) by opening a web browser and navigating to `http://localhost:5000` (or the appropriate URL and port specified in the configuration).

## Configuration

The project's configuration is stored in the `config.yml` file. Here's an overview of the available configuration options:

- `mqtt`:
  - `broker`: The URL or IP address of the MQTT broker.
  - `port`: The port number of the MQTT broker.
  - `username`: The username for authenticating with the MQTT broker (if required).
  - `password`: The password for authenticating with the MQTT broker (if required).
  - `topic`: The MQTT topic to subscribe to for receiving sensor data.

- `database`:
  - `url`: The URL or connection string for the database.
  - `name`: The name of the database.
  - `username`: The username for accessing the database.
  - `password`: The password for accessing the database.

- `meraki`:
  - `api_key`: The API key for accessing the Meraki dashboard.
  - `organization_id`: The ID of the Meraki organization.
  - `network_id`: The ID of the Meraki network.
  - `sensor_serial`: The serial number of the Meraki sensor.

Make sure to update these configuration settings based on your specific environment and requirements.

## Troubleshooting

If you encounter any issues during the installation or running of the project, consider the following troubleshooting steps:

- Verify that Docker is properly installed and running on your system.
- Double-check the configuration settings in the `config.yml` file.
- Ensure that the MQTT broker and database are accessible and running.
- Check the Docker logs for any error messages or exceptions using the command: `docker logs room-occupancy-prediction`.
- Refer to the project's documentation or seek assistance from the project maintainers if the issue persists.

## Conclusion

By following this installation guide, you should have the Room Occupancy Prediction project up and running in a containerized environment using Docker and Conda. You can now start collecting sensor data, training models, and performing occupancy predictions based on the provided functionality.

If you have any further questions or need additional assistance, please refer to the project's documentation or contact the project maintainers.

---

# Room Occupancy Prediction Project: Installation Guide

This guide provides step-by-step instructions for setting up and running the Room Occupancy Prediction project using Docker, Conda, Python, and Git.

## Prerequisites

Before proceeding with the installation, ensure that you have the following prerequisites installed on your system:

- Docker
- Git

## Installation Steps

1. **Clone the repository:**
   ```shell
   git clone https://github.com/crazystorm720/room-occupancy-prediction-mqtt-meraki.git
   ```

2. **Navigate to the project directory:**
   ```shell
   cd room-occupancy-prediction-mqtt-meraki
   ```

3. **Create a `.gitignore` file to exclude unnecessary files from version control:**
   ```shell
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
   ```shell
   touch Dockerfile
   ```
   Open the `Dockerfile` and add the following content:
   ```dockerfile
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
   CMD ["python", "src/main.py"]
   ```

5. **Create an `environment.yml` file to define your Conda environment:**
   ```shell
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

6. **Create the project structure:**
   ```shell
   mkdir -p data/raw data/processed data/models
   mkdir -p src tests notebooks
   touch src/__init__.py tests/__init__.py
   touch src/data_collection.py src/data_preprocessing.py src/feature_engineering.py
   touch src/model_training.py src/model_inference.py src/monitoring.py
   touch src/main.py
   touch notebooks/exploratory_analysis.ipynb
   ```

7. **Set up the configuration:**
   - Create a copy of the `config.example.yml` file and rename it to `config.yml`.
   - Open the `config.yml` file and update the configuration settings according to your environment, such as MQTT broker details, database credentials, and Meraki sensor information.

8. **Build the Docker image:**
   ```shell
   docker build -t room-occupancy-prediction .
   ```

9. **Run the Docker container:**
   ```shell
   docker run -d --name room-occupancy-prediction -v /path/to/config.yml:/app/config.yml room-occupancy-prediction
   ```
   Replace `/path/to/config.yml` with the actual path to your `config.yml` file.

   This command starts the Docker container in detached mode (`-d`) and mounts the `config.yml` file from your local system to the container's `/app/config.yml` path.

10. **Access the project:**
    - The project will start collecting data from the configured MQTT broker and Meraki sensors.
    - You can access the project's web interface (if provided) by opening a web browser and navigating to `http://localhost:5000` (or the appropriate URL and port specified in the configuration).

## Configuration

The project's configuration is stored in the `config.yml` file. Here's an overview of the available configuration options:

- **mqtt**:
  - `broker`: The URL or IP address of the MQTT broker.
  - `port`: The port number of the MQTT broker.
  - `username`: The username for authenticating with the MQTT broker (if required).
  - `password`: The password for authenticating with the MQTT broker (if required).
  - `topic`: The MQTT topic to subscribe to for receiving sensor data.

- **database**:
  - `url`: The URL or connection string for the database.
  - `name`: The name of the database.
  - `username`: The username for accessing the database.
  - `password`: The password for accessing the database.

- **meraki**:
  - `api_key`: The API key for accessing the Meraki dashboard.
  - `organization_id`: The ID of the Meraki organization.
  - `network_id`: The ID of the Meraki network.
  - `sensor_serial`: The serial number of the Meraki sensor.

Make sure to update these configuration settings based on your specific environment and requirements.

## Troubleshooting

If you encounter any issues during the installation or running of the project, consider the following troubleshooting steps:

- Verify that Docker is properly installed and running on your system.
- Double-check the configuration settings in the `config.yml` file.
- Ensure that the MQTT broker and database are accessible and running.
- Check the Docker logs for any error messages or exceptions using the command: `docker logs room-occupancy-prediction`.
- Refer to the project's documentation or seek assistance from the project maintainers if the issue persists.

## Conclusion

By following this installation guide, you should have the Room Occupancy Prediction project up and running in a containerized environment using Docker and Conda. You can now start collecting sensor data, training models, and performing occupancy predictions based on the provided functionality.

If you have any further questions or need additional assistance, please refer to the project's documentation or contact the project maintainers.



