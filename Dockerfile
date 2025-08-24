# Dockerfile (Definitive Final Version)

# Step 1: Start from the official NVIDIA image to guarantee system-level GPU drivers.
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Step 2: Install system tools (wget) and then install Miniconda.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Step 3: Add the base conda installation to the system PATH.
ENV PATH /opt/conda/bin:$PATH

# Step 4: Accept the Anaconda Terms of Service to prevent build errors.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Step 5: Copy our environment definition file into the container.
COPY environment.yml .

# Step 6: Create the Conda environment using our file (which includes the MKL fix).
RUN conda env create -f environment.yml && conda clean -afy

# Step 7: Prepend our new environment's bin directory to the system PATH.
# This makes `ssvep-env` the default environment for all subsequent commands.
ENV PATH /opt/conda/envs/ssvep-env/bin:$PATH

# Step 8: Copy your project code into the image and set it as the working directory.
COPY . /root/mtgnet
WORKDIR /root/mtgnet