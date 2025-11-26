# --------------------------------------------------------
# 1. Base image: PyTorch + CUDA 11.6 + cuDNN 8
# --------------------------------------------------------
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# --------------------------------------------------------
# 2. Environment setup
# --------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --------------------------------------------------------
# 3. Install system dependencies
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        ninja-build \
        vim \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# 4. Copy requirements and install Python dependencies
# --------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch_geometric==2.4.0 \
        -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# --------------------------------------------------------
# 5. Copy project code and assets
# --------------------------------------------------------
COPY inference.py /app/inference.py
COPY data_processing.py /app/data_processing.py
COPY noise_cancel.py /app/noise_cancel.py
COPY model/ /app/model/
COPY datasets/ /app/datasets/
COPY guided_diffusion/ /app/guided_diffusion/

# --------------------------------------------------------
# 6. Environment variables for easier configuration
# --------------------------------------------------------
ENV MODEL_PATH=/app/model/LiVS_reannotated_subset_F0_model160000.pt 

# --------------------------------------------------------
# 7. Entrypoint (default inference)
# --------------------------------------------------------
ENTRYPOINT ["python", "/app/inference.py"]
# CMD ["python", "/app/inference.py"]