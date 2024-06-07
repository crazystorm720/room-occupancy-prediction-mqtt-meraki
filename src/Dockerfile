FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file to the working directory
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the Conda environment
SHELL ["conda", "run", "-n", "room-occupancy-prediction", "/bin/bash", "-c"]

# Copy the project files to the working directory
COPY . .

# Copy the startup script to the working directory
COPY start.sh .

# Expose the Jupyter notebook port
EXPOSE 8888

# Set the entry point command
CMD ["conda", "run", "-n", "room-occupancy-prediction", "./start.sh"]

