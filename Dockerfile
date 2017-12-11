FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

# Copy the main source code to the container
COPY . /app
WORKDIR /app

# Install system dependencies
RUN  apt-get update \
  && apt-get install -y wget git protobuf-compiler python3-tk

# Run setup script
RUN /app/setup.sh

# Add tensorflow object detection library to path
ENV PYTHONPATH ${PYTHONPATH}:/app/lib/tensorflow_models/research/:/app/lib/tensorflow_models/research/slim