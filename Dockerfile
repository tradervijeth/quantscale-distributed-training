# QuantScale: Distributed Training Framework for Financial Time Series
#
# CUDA-enabled Docker container for distributed training
#
# Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
# Licensed under the Apache License 2.0
# Contact: finance@vijeth.com

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Vithushan Jeyapahan <finance@vijeth.com>"
LABEL description="QuantScale: Distributed Training for Financial ML"
LABEL version="0.1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/quantscale

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.5.0 \
    isort>=5.12.0 \
    jupyter>=1.0.0 \
    ipython>=8.0.0

# Copy the rest of the code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# Set permissions
RUN chmod +x scripts/*.py

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command
CMD ["/bin/bash"]
