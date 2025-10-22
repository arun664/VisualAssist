# AI Navigation Assistant - Complete Docker Application

This guide explains how to build and run the complete AI Navigation Assistant application using Docker. The complete application includes both frontend and backend in a single container with Nginx as a reverse proxy.

## 🏗️ Architecture

The complete Docker image contains:

- **Frontend**: HTML/CSS/JavaScript client served by Nginx
- **Backend**: Python FastAPI server with AI capabilities
- **Nginx**: Web server and reverse proxy
- **All Dependencies**: Computer vision, speech recognition, WebRTC support

```
┌─────────────────────────────────────────┐
│              Docker Container           │
├─────────────────────────────────────────┤
│  Nginx (Port 80)                       │
│  ├── Serves frontend files             │
│  ├── Proxies /api → Backend            │
│  ├── Proxies /ws → WebSocket           │
│  └── Proxies /webrtc → WebRTC          │
├─────────────────────────────────────────┤
│  Python Backend (Port 8000)            │
│  ├── FastAPI server                    │
│  ├── Computer vision processing        │
│  ├── Speech recognition                │
│  ├── WebRTC handling                   │
│  └── Navigation FSM                    │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Using Build Scripts (Recommended)

**Linux/macOS:**
```bash
# Build the complete image
./build-complete.sh

# Run the complete application
./run-complete.sh
```

**Windows:**
```cmd
# Build the complete image
build-complete.bat

# Run the complete application
run-complete.bat
```

### Option 2: Using Docker Compose
```bash
# Build and run with docker-compose
docker-compose -f docker-compose.complete.yml up --build

# Run in detached mode
docker-compose -f docker-compose.complete.yml up -d

# Stop the services
docker-compose -f docker-compose.complete.yml down
```

### Option 3: Manual Docker Commands
```bash
# Build the image
docker build -t ai-navigation-complete:latest .

# Run the container
docker run -p 80:80 -p 8000:8000 ai-navigation-complete:latest
```

## 🔧 Build Options

### Basic Build
```bash
# Linux/macOS
./build-complete.sh

# Windows
build-complete.bat

# Manual
docker build -t ai-navigation-complete:latest .
```

### Build with Custom Options
```bash
# Custom tag
./build-complete.sh --tag v1.0.0

# Custom name
./build-complete.sh --name my-navigation-app

# No cache (clean build)
./build-complete.sh --no-cache
```

## 🚀 Run Options

### Basic Run
```bash
# Default ports (80 for frontend, 8000 for backend)
./run-complete.sh

# Access at: http://localhost:80
```

### Custom Ports
```bash
# Frontend on port 3000, backend on port 8001
./run-complete.sh --frontend-port 3000 --backend-port 8001

# Access at: http://localhost:3000
```

### Development Mode
```bash
# Run in development environment
./run-complete.sh --dev

# Build and run in development
./run-complete.sh --build --dev
```

### Detached Mode
```bash
# Run in background
./run-complete.sh --detach

# Or short form
./run-complete.sh -d
```

## 🌐 Application URLs

When running on default ports:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:80 | Main application interface |
| **Backend API** | http://localhost:80/api | API endpoints (proxied) |
| **Health Check** | http://localhost:80/health | Application health status |
| **WebSocket** | ws://localhost:80/ws | Real-time communication |
| **WebRTC** | http://localhost:80/webrtc | WebRTC endpoints |
| **Video Stream** | http://localhost:80/processed_video_stream | MJPEG video stream |
| **Backend Direct** | http://localhost:8000 | Direct backend access |

## 🔍 Container Management

### View Logs
```bash
# Using scripts
./run-complete.sh --logs

# Manual
docker logs -f ai-navigation-complete
```

### Stop Container
```bash
# Using scripts
./run-complete.sh --stop

# Manual
docker stop ai-navigation-complete
docker rm ai-navigation-complete
```

### Access Container Shell
```bash
docker exec -it ai-navigation-complete /bin/bash
```

### Container Status
```bash
# View running containers
docker ps

# View images
docker images ai-navigation-complete

# View container details
docker inspect ai-navigation-complete
```

## 🐳 Docker Compose Profiles

### Production (Default)
```bash
# Basic production setup
docker-compose -f docker-compose.complete.yml up
```

### Development
```bash
# Development with source code mounting
docker-compose -f docker-compose.complete.yml --profile dev up
```

### Production with Load Balancer
```bash
# Production with Nginx load balancer
docker-compose -f docker-compose.complete.yml --profile production up
```

### With Monitoring
```bash
# Include Prometheus and Grafana
docker-compose -f docker-compose.complete.yml --profile monitoring up
```

### All Services
```bash
# Run everything
docker-compose -f docker-compose.complete.yml \
  --profile dev --profile production --profile monitoring up
```

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `production` | Environment mode |
| `SERVER_HOST` | `127.0.0.1` | Backend bind address |
| `SERVER_PORT` | `8000` | Backend port |
| `FRONTEND_PORT` | `80` | Frontend port |
| `PYTHONPATH` | `/app/backend` | Python module path |
| `YOLO_CONFIG_DIR` | `/app/backend/models` | YOLO model directory |
| `NUMBA_CACHE_DIR` | `/tmp/numba_cache` | Numba cache directory |

### Custom Environment
```bash
docker run -p 80:80 -p 8000:8000 \
  -e ENVIRONMENT=development \
  -e SERVER_PORT=8000 \
  ai-navigation-complete:latest
```

## 📁 Volume Mounts

The container supports persistent storage:

- **Backend Logs**: `/app/backend/logs`
- **AI Models**: `/app/backend/models`

### Example with Volumes
```bash
docker run -p 80:80 -p 8000:8000 \
  -v $(pwd)/backend/logs:/app/backend/logs \
  -v $(pwd)/backend/models:/app/backend/models \
  ai-navigation-complete:latest
```

## 🏥 Health Checks

The container includes comprehensive health checks:

```bash
# Frontend health
curl http://localhost:80/health

# Backend health (via proxy)
curl http://localhost:80/api/health

# Backend health (direct)
curl http://localhost:8000/health
```

Health check responses:
```json
{
  "status": "healthy",
  "service": "frontend",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🔧 Configuration

### Nginx Configuration
The Nginx configuration is located at `/etc/nginx/nginx.conf` in the container. Key features:

- Serves static frontend files
- Proxies API requests to backend
- WebSocket support for real-time communication
- WebRTC endpoint proxying
- MJPEG video streaming
- Security headers
- Gzip compression
- Error pages

### Custom Nginx Config
```bash
# Mount custom nginx config
docker run -p 80:80 -p 8000:8000 \
  -v $(pwd)/custom-nginx.conf:/etc/nginx/nginx.conf:ro \
  ai-navigation-complete:latest
```

## 🐛 Debugging

### Interactive Mode
```bash
# Run interactively
./run-complete.sh --interactive

# Or manual
docker run -it -p 80:80 -p 8000:8000 ai-navigation-complete:latest /bin/bash
```

### Service Logs
```bash
# All logs
docker logs -f ai-navigation-complete

# Nginx logs
docker exec ai-navigation-complete tail -f /var/log/nginx/access.log
docker exec ai-navigation-complete tail -f /var/log/nginx/error.log

# Backend logs
docker exec ai-navigation-complete tail -f /app/backend/logs/backend.log
```

### Common Issues

1. **Port conflicts:**
   ```bash
   # Use different ports
   ./run-complete.sh --frontend-port 3000 --backend-port 8001
   ```

2. **Permission issues:**
   ```bash
   # Check container user
   docker exec ai-navigation-complete whoami
   
   # Fix volume permissions
   sudo chown -R 1000:1000 backend/logs backend/models
   ```

3. **Build failures:**
   ```bash
   # Clean build
   ./build-complete.sh --no-cache
   
   # Check disk space
   docker system df
   docker system prune
   ```

4. **Service not starting:**
   ```bash
   # Check logs
   docker logs ai-navigation-complete
   
   # Check processes
   docker exec ai-navigation-complete ps aux
   ```

## 📊 Resource Requirements

### Minimum Requirements
- **CPU:** 2+ cores
- **RAM:** 4GB+
- **Disk:** 8GB+ (for image and models)
- **Network:** Ports 80, 8000

### Recommended Requirements
- **CPU:** 4+ cores
- **RAM:** 8GB+
- **Disk:** 16GB+ (with persistent storage)
- **Network:** Stable internet for model downloads

## 🔐 Security

### Built-in Security Features
- Non-root user (appuser, UID 1000)
- Security headers in Nginx
- No hardcoded secrets
- Health check endpoints
- Resource isolation

### Production Security
```bash
# Run with resource limits
docker run -p 80:80 -p 8000:8000 \
  --memory=4g \
  --cpus=2 \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  ai-navigation-complete:latest
```

## 🚀 Production Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.complete.yml ai-navigation
```

### Kubernetes
```bash
# Generate manifests
docker-compose -f docker-compose.complete.yml config | kompose convert -f -

# Apply to cluster
kubectl apply -f .
```

### Cloud Deployment
The complete image can be deployed to:
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- DigitalOcean App Platform
- Railway
- Heroku (with Docker support)

## 📈 Monitoring

### Built-in Monitoring
- Health check endpoints
- Nginx access/error logs
- Backend application logs
- Container resource usage

### External Monitoring
```bash
# With Prometheus and Grafana
docker-compose -f docker-compose.complete.yml --profile monitoring up
```

Access monitoring:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

## 🔄 Updates and Maintenance

### Update Application
```bash
# Rebuild with latest code
./build-complete.sh --no-cache

# Stop old container and start new one
./run-complete.sh --stop
./run-complete.sh
```

### Backup Data
```bash
# Backup logs and models
tar -czf backup-$(date +%Y%m%d).tar.gz backend/logs backend/models
```

### Clean Up
```bash
# Remove old images
docker image prune

# Remove all unused resources
docker system prune -a
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🆘 Support

If you encounter issues:

1. Check the logs: `./run-complete.sh --logs`
2. Verify health: `curl http://localhost:80/health`
3. Test individual services
4. Review resource usage: `docker stats`
5. Check network connectivity
6. Rebuild with `--no-cache` if needed

The complete Docker setup provides a production-ready deployment of the AI Navigation Assistant with all components integrated and properly configured.