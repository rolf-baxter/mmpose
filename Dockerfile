# Use NVIDIA CUDA as the base image
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install prerequisites
RUN apt-get update && apt-get install -y wget git ffmpeg libsm6 libxext6

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda

ENV PATH="/miniconda/bin:${PATH}"

# Create a conda environment
RUN conda create --name openmmlab python=3.8 -y

# Set shell to use conda activation
SHELL ["conda", "run", "-n", "openmmlab", "/bin/bash", "-c"]

# Install PyTorch

#RUN conda install pytorch torchvision -c pytorch \
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Do we need this?
#RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run \
#RUN sh cuda_12.3.2_545.23.08_linux.run
#

# Install OpenCV
run pip install opencv-python && \
    pip install json_tricks



# Install MMEngine, MMCV, and MMPose dependencies
RUN pip install -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0"

# Clone MMPose repo and install
RUN git clone https://github.com/open-mmlab/mmpose.git && \
    cd mmpose && \
    pip install -r requirements.txt && \
    pip install -v -e .

# Reset the shell to default for subsequent commands
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /application

# Instructions for running the container might include mounting volumes
# for the config and checkpoint files and then running the verification steps.

# Run this is you're in interactive shell:
# source activate {env}" > ~/.bashrc