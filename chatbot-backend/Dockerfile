# 1. Base image with CUDA runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 2. Install system dependencies (Python + build tools)
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Make 'python' point to 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# 4. Set working directory
WORKDIR /app

# 5. Upgrade pip
RUN pip3 install --upgrade pip

# 6. Install PyTorch with CUDA 12.1
RUN pip install torch==2.3.1+cu121 torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# 7. Install bitsandbytes (CUDA kernels)
RUN pip install -U bitsandbytes

# 8. Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# 9. Copy application code
COPY . .

# 10. Expose FastAPI port
EXPOSE 8000

# 11. Run with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

