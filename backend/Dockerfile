# AI Navigation Assistant Backend - Docker Configuration
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV and AI packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies with proper error handling
RUN pip install --no-cache-dir --timeout 1000 --retries 3 -r requirements.txt || \
    (echo "Full requirements install failed, installing core packages..." && \
     pip install --no-cache-dir fastapi uvicorn websockets requests pydantic python-multipart psutil)

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs models && \
    chmod 755 logs models

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV YOLO_CONFIG_DIR=/app/models
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Try to download YOLO model, but don't fail if it doesn't work
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
try:\n\
    from ultralytics import YOLO\n\
    print("Downloading YOLO model...")\n\
    model = YOLO("yolo11n.pt")\n\
    print("YOLO model downloaded successfully")\n\
except Exception as e:\n\
    print(f"Warning: Could not download YOLO model: {e}")\n\
    print("The application will attempt to download it at runtime")\n\
' > /app/download_yolo.py && \
    python /app/download_yolo.py || echo "YOLO download failed, will try at runtime"

# Expose port
EXPOSE 8000

# Health check with proper error handling
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)" || exit 1

# Run the application with proper error handling
CMD ["python", "-u", "entrypoint.py"]