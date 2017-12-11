#!/bin/bash

# Install python requirements
pip3 install -r requirements.txt

# Download data & models from s3
wget http://skycatch-challenge.s3.amazonaws.com/data.zip
wget http://skycatch-challenge.s3.amazonaws.com/models.zip
unzip data.zip
unzip models.zip

# Download & install tensorflow object detection library
git clone https://github.com/tensorflow/models.git /app/lib/tensorflow_models
cd /app/lib/tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=.

# Download ffmpeg plugin
mkdir -p /root/.imageio/ffmpeg 
wget https://github.com/imageio/imageio-binaries/raw/master/ffmpeg/ffmpeg.linux64 -O /root/.imageio/ffmpeg/ffmpeg.linux64

# Make directory to share data between host and docker container
mkdir docker_output