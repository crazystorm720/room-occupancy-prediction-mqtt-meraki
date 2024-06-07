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
   CMD ["python", "src/main.py"]
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

6. **Create the project structure:**
   ```
   mkdir -p data/raw data/processed data/models
   mkdir -p src tests notebooks
   touch src/__init__.py tests/__init__.py
   touch src/data_collection.py src/data_preprocessing.py src/feature_engineering.py
   touch src/model_training.py src/model_inference.py src/monitoring.py
   touch src/main.py
   touch notebooks/exploratory_analysis.ipynb
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
    docker run -it room-occupancy-prediction bash
    ```
    This will start the Docker container and open a bash terminal inside the container.

## Next Steps

- Implement the various components of your project, such as data collection, preprocessing, feature engineering, model training, inference, and monitoring, within the Docker container.
- Update the respective files in the `src` directory with your project code.
- Modify the `environment.yml` file if you require additional dependencies.
- Document your progress and update the README file.
- Regularly commit your changes to the Git repository.

Happy coding and best of luck with your room occupancy prediction project!
