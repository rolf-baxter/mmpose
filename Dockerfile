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


# Build mmdet
RUN git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && \
    pip install -v -e .

# Install MMEngine, MMCV, and MMPose dependencies
RUN pip install -U openmim && \
    mim install mmengine
#    mim install "mmcv>=2.0.1"
#    mim install "mmdet>=3.1.0" - now built from source above

# Build mmcv from source
RUN git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    pip install -r requirements/optional.txt && \
    pip install -e . -v

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
# source activate openmmlab > ~/.bashrc

RUN echo "source activate openmmlab" > ~/.bashrc
# Sample demo video processing command
# python demo/topdown_demo_with_mmdet.py     demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py     https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth     configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py     https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth     --input /personyx-data/Movement/Bend_and_Lift.mp4     --output-root=/personyx-data/Movement/out/ --device cuda


# Weird hack - no idea why we have to do this again here once we're in the container otherwise  cuda wont work
RUN cd /mmcv && \
    pip install -r requirements/optional.txt && \
    pip install -e . -v
