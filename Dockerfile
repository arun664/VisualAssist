# AI Navigation Assistant - Complete Application Docker Image
# This builds both frontend and backend into a single container

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for both frontend and backend
RUN apt-get update && apt-get install -y \
    # Backend dependencies
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
    # Frontend/Web server dependencies
    nginx \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for better caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies with proper error handling
RUN pip install --no-cache-dir --timeout 1000 --retries 3 -r /app/backend/requirements.txt || \
    (echo "Full requirements install failed, installing core packages..." && \
     pip install --no-cache-dir fastapi uvicorn websockets requests pydantic python-multipart psutil)

# Copy all application code
COPY . /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/backend/logs /app/backend/models /var/log/nginx /var/lib/nginx && \
    chmod 755 /app/backend/logs /app/backend/models

# Configure Nginx for serving frontend and proxying backend
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Set environment variables
ENV PYTHONPATH=/app/backend
ENV ENVIRONMENT=production
ENV SERVER_HOST=127.0.0.1
ENV SERVER_PORT=8000
ENV FRONTEND_PORT=80
ENV YOLO_CONFIG_DIR=/app/backend/models
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /var/log/nginx && \
    chown -R appuser:appuser /var/lib/nginx && \
    chown -R appuser:appuser /etc/nginx

# Try to download YOLO model, but don't fail if it doesn't work
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
sys.path.append("/app/backend")\n\
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

# Switch to non-root user
USER appuser

# Expose ports (80 for frontend, 8000 for backend API)
EXPOSE 80 8000

# Health check for both services
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || curl -f http://localhost:8000/health || exit 1

# Copy startup script
COPY docker/start.sh /app/start.sh

# Make startup script executable and run it
USER root
RUN chmod +x /app/start.sh
USER appuser

# Run both frontend (nginx) and backend (python) services
CMD ["/app/start.sh"]