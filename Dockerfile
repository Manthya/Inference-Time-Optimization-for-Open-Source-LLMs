# =============================================================================
# Inference-Time Optimization Experiment - Docker Image
# =============================================================================
# This Dockerfile pre-builds everything needed to run the experiment:
# - CUDA runtime environment
# - Python dependencies (PyTorch, vLLM, Triton, etc.)
# - Model weights (Qwen2.5-7B-Instruct)
# - Evaluation datasets
# =============================================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Python Dependencies
# =============================================================================
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional tools for profiling
RUN pip install --no-cache-dir \
    py-spy \
    memory_profiler \
    nvitop

# =============================================================================
# Download Model (Pre-built into image)
# =============================================================================
COPY docker/download_model.py /app/docker/download_model.py

# Download model during build (cached in image)
ARG MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
ENV MODEL_NAME=${MODEL_NAME}
ENV HF_HOME=/app/models

# Download model weights
RUN python /app/docker/download_model.py --model ${MODEL_NAME} --cache-dir ${HF_HOME}

# =============================================================================
# Download Evaluation Datasets
# =============================================================================
COPY docker/download_datasets.py /app/docker/download_datasets.py

# Download datasets during build
RUN python /app/docker/download_datasets.py --output-dir /app/data

# =============================================================================
# Copy Application Code
# =============================================================================
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY run_experiment.py /app/
COPY test_setup.py /app/

# Create output directories
RUN mkdir -p /app/results/plots /app/results/logs

# Set permissions
RUN chmod +x /app/scripts/*.sh

# =============================================================================
# Entrypoint
# =============================================================================
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["run"]

# =============================================================================
# Labels
# =============================================================================
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="Inference-Time Optimization Experiment for LLMs"
