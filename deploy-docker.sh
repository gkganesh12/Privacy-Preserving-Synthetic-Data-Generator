#!/bin/bash

# Build the Docker image
docker build -t synthetic-data-generator .

# Run the Docker container
docker run -p 8501:8501 synthetic-data-generator