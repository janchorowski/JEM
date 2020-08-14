ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/jem
RUN apt-get update && apt-get install -y rsync
WORKDIR /workspace/jem
