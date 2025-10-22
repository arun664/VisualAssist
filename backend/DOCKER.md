# AI Navigation Assistant Backend - Docker Guide

This guide explains how to build and run the AI Navigation Assistant Backend using Docker.

## 🐳 Quick Start

### Option 1: Using Build Scripts (Recommended)

**Linux/macOS:**
```bash
# Build the Docker image
./build-docker.sh

# Run the container
./run-docker.sh
```

**Windows:**
```cmd
# Build the Docker image
build-docker.bat

# Run the container
run-docker.bat
```

### Option 2: Using Docker Compose
```bash
# Build and run with docker-compose
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop the services
docker-compose down
```

### Option 3: Manual Docker Commands
```bash
# Build the image
docker build -t ai-navigation-backend:latest .

# Run the container
docker run -p 8000:8000 ai-navigation-backend:latest
```

## 📦 Available Docker Images

### Full Version (Default)
- **Dockerfile:** `Dockerfile`
- **Features:** Complete AI navigation system with computer vision, speech recognition, WebRTC
- **Size:** ~2-3GB
- **Use case:** Production deployment with all features

### Simple Version
- **Dockerfile:** `Dockerfile.simple`
- **Features:** Basic API endpoints, minimal dependencies
- **Size:** ~500MB
- **Use case:** Testing, development, or when full AI features aren't needed

## 🔧 Build Options

### Build Full Version
```bash
# Linux/macOS
./build-docker.sh

# Windows
build-docker.bat

# Manual
docker build -t ai-navigation-backend:latest .
```

### Build Simple Version
```bash
# Linux/macOS
./build-docker.sh --simple

# Windows
build-docker.bat --simple

# Manual
docker build -f Dockerfile.simple -t ai-navigation-backend:simple .
```

### Build with Custom Tag
```bash
# Linux/macOS
./build-docker.sh --tag v1.0.0

# Windows
build-docker.bat --tag v1.0.0
```

## 🚀 Run Options

### Basic Run
```bash
# Linux/macOS
./run-docker.sh

# Windows
run-docker.bat

# Manual
docker run -p 8000:8000 ai-navigation-backend:latest
```

### Run Simple Version
```bash
# Linux/macOS
./run-docker.sh --simple

# Windows
run-docker.bat --simple
```

### Run in Development Mode
```bash
# Linux/macOS
./run-docker.sh --dev

# Windows
run-docker.bat --dev
```

### Run in Detached Mode
```bash
# Linux/macOS
./run-docker.sh --detach

# Windows
run-docker.bat --detach
```

### Run on Custom Port
```bash
# Linux/macOS
./run-docker.sh --port 8001

# Windows
run-docker.bat --port 8001
```

## 🔍 Container Management

### View Logs
```bash
# Using scripts
./run-docker.sh --logs        # Linux/macOS
run-docker.bat --logs         # Windows

# Manual
docker logs -f ai-navigation-backend
```

### Stop Container
```bash
# Using scripts
./run-docker.sh --stop        # Linux/macOS
run-docker.bat --stop         # Windows

# Manual
docker stop ai-navigation-backend
docker rm ai-navigation-backend
```

### Access Container Shell
```bash
docker exec -it ai-navigation-backend /bin/bash
```

### View Container Status
```bash
docker ps
docker images ai-navigation-backend
```

## 🌐 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `production` | Environment mode (`development`, `production`) |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8000` | Server port |
| `PYTHONPATH` | `/app` | Python module path |
| `YOLO_CONFIG_DIR` | `/app/models` | YOLO model directory |
| `NUMBA_CACHE_DIR` | `/tmp/numba_cache` | Numba compilation cache |

### Custom Environment Variables
```bash
docker run -p 8000:8000 \
  -e ENVIRONMENT=development \
  -e SERVER_PORT=8000 \
  ai-navigation-backend:latest
```

## 📁 Volume Mounts

The container supports persistent storage for:

- **Logs:** `/app/logs` - Application logs
- **Models:** `/app/models` - AI model files (YOLO, etc.)

### Example with Volume Mounts
```bash
docker run -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  ai-navigation-backend:latest
```

## 🏥 Health Checks

The container includes built-in health checks:

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Manual health check
curl http://localhost:8000/health
```

Health check endpoint returns:
```json
{
  "status": "healthy",
  "components": {
    "api": "running",
    "webrtc": "ready",
    "computer_vision": "ready"
  }
}
```

## 🐛 Debugging

### Run in Interactive Mode
```bash
# Linux/macOS
./run-docker.sh --interactive

# Windows
run-docker.bat --interactive

# Manual
docker run -it -p 8000:8000 ai-navigation-backend:latest /bin/bash
```

### View Detailed Logs
```bash
# Container logs
docker logs -f ai-navigation-backend

# Application logs (if mounted)
tail -f logs/backend.log
```

### Common Issues

1. **Port already in use:**
   ```bash
   # Use different port
   ./run-docker.sh --port 8001
   ```

2. **Image not found:**
   ```bash
   # Build image first
   ./build-docker.sh
   ```

3. **Permission denied (Linux/macOS):**
   ```bash
   # Make scripts executable
   chmod +x build-docker.sh run-docker.sh
   ```

4. **Out of disk space:**
   ```bash
   # Clean up Docker
   docker system prune -a
   ```

## 🔄 Docker Compose

### Basic Usage
```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and start
docker-compose up --build
```

### Service Profiles

**Default:** Full version on port 8000
```bash
docker-compose up
```

**Simple version:** Minimal version on port 8001
```bash
docker-compose --profile simple up
```

## 🚀 Production Deployment

### Using Docker Compose (Recommended)
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# With custom environment
ENVIRONMENT=production docker-compose up -d
```

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ai-navigation
```

### Using Kubernetes
```bash
# Generate Kubernetes manifests
docker-compose config | kompose convert -f -

# Apply to cluster
kubectl apply -f .
```

## 📊 Resource Requirements

### Full Version
- **CPU:** 2+ cores recommended
- **RAM:** 4GB+ recommended
- **Disk:** 5GB+ for image and models
- **Network:** Ports 8000 (HTTP), WebRTC ports

### Simple Version
- **CPU:** 1+ core
- **RAM:** 1GB+ recommended
- **Disk:** 1GB+ for image
- **Network:** Port 8000 (HTTP)

## 🔐 Security Considerations

1. **Non-root user:** Container runs as `appuser` (UID 1000)
2. **Minimal base image:** Uses `python:3.9-slim`
3. **No sensitive data:** No hardcoded secrets or credentials
4. **Health checks:** Built-in monitoring
5. **Resource limits:** Configure in production

### Production Security
```bash
# Run with resource limits
docker run -p 8000:8000 \
  --memory=2g \
  --cpus=2 \
  --read-only \
  --tmpfs /tmp \
  ai-navigation-backend:latest
```

## 📝 Troubleshooting

### Build Issues
```bash
# Clean build (no cache)
docker build --no-cache -t ai-navigation-backend:latest .

# Check build logs
docker build -t ai-navigation-backend:latest . 2>&1 | tee build.log
```

### Runtime Issues
```bash
# Check container status
docker ps -a

# Inspect container
docker inspect ai-navigation-backend

# Check resource usage
docker stats ai-navigation-backend
```

### Network Issues
```bash
# Check port binding
netstat -tulpn | grep 8000

# Test connectivity
curl -v http://localhost:8000/health
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Python Docker Best Practices](https://docs.docker.com/language/python/)