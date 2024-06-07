FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file to the working directory
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the Conda environment
RUN echo "conda activate room-occupancy-prediction" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy the project files to the working directory
COPY . .

# Expose the Jupyter notebook port
EXPOSE 8888

# Set the entry point command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
