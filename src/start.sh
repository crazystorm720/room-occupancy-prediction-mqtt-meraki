#!/bin/bash

# Start the Jupyter Notebook server
exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

