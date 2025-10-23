# ===============================================================
# pyAFQ GPU Dockerfile (fixed niimath + read-only FS safe)
# ===============================================================

# Base image with CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    apt-transport-https ca-certificates gnupg software-properties-common \
    gcc git wget curl numactl cmake && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# Install Miniconda
# ---------------------------------------------------------------
RUN curl -L "https://repo.anaconda.com/miniconda/Miniconda3-py311_25.1.1-1-Linux-x86_64.sh" \
      -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/anaconda && \
    rm /tmp/miniconda.sh

# Add Conda to PATH
ENV PATH=/opt/anaconda/bin:$PATH

# ---------------------------------------------------------------
# Create and populate AFQ environment
# ---------------------------------------------------------------
RUN /opt/anaconda/bin/conda create -y -n afq python=3.11 && \
    /opt/anaconda/envs/afq/bin/pip install \
        numpy>=2.0.0 scipy>=1.13.0 cython nibabel dipy tqdm fslpy && \
    /opt/anaconda/envs/afq/bin/pip install git+https://github.com/dipy/GPUStreamlines.git && \
    /opt/anaconda/envs/afq/bin/pip install git+https://github.com/36000/pyAFQ.git@dam && \
    #Address read only to executable permissions
    # --- Fix niimath ---
    /opt/anaconda/envs/afq/bin/pip install -U git+https://github.com/spikedoanz/niimath.git@niimath-permissions

# ---------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------
ENV CONDA_DEFAULT_ENV=afq
ENV PATH=/opt/anaconda/envs/afq/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/anaconda/lib:$LD_LIBRARY_PATH

# ---------------------------------------------------------------
# Default command (runs Python in AFQ env)
# ---------------------------------------------------------------
ENTRYPOINT ["/usr/bin/env", "CUDA=1", "/opt/anaconda/envs/afq/bin/python"]