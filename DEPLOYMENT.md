# AI Navigation Assistant - Deployment Guide

This guide covers deployment configurations and procedures for the AI Navigation Assistant system.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## Overview

The AI Navigation Assistant consists of three main components:

1. **Backend Server**: FastAPI application with AI processing capabilities
2. **Frontend Interface**: Web-based user interface for navigation control
3. **Client Device**: Mobile client for sensor data capture

The system supports both development and production deployment modes with different configurations for each environment.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for production)
- **Storage**: Minimum 2GB free space
- **Network**: Stable internet connection for WebRTC functionality

### Required Software

```bash
# Python and pip
python3 --version  # Should be 3.8+
pip3 --version

# Optional but recommended for production
gunicorn --version  # For production WSGI server
nginx --version     # For reverse proxy (production)
```

### Hardware Requirements

- **CPU**: Multi-core processor (4+ cores recommended for production)
- **GPU**: Optional CUDA-compatible GPU for faster YOLOv11 inference
- **Camera**: Required for client device functionality
- **Microphone**: Required for voice command functionality

## Environment Configuration

The system uses environment-specific configuration files located in each component directory:

### Backend Configuration

**Development**: `backend/.env.development`
**Production**: `backend/.env.production`

Key configuration parameters:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true/false
LOG_LEVEL=debug/info

# AI Model Configuration
YOLO_MODEL_PATH=yolov11n.pt
YOLO_CONFIDENCE_THRESHOLD=0.5
YOLO_DEVICE=cpu/cuda

# Safety Configuration
MAX_PROCESSING_LATENCY_MS=100
EMERGENCY_STOP_ENABLED=true
```

### Frontend Configuration

**Development**: `frontend/.env.development`
**Production**: `frontend/.env.production`

```bash
# Backend Connection
BACKEND_URL=http://localhost:8000
BACKEND_WS_URL=ws://localhost:8000/ws

# Audio Configuration
SPEECH_SYNTHESIS_LANG=en-US
SPEECH_SYNTHESIS_RATE=1.0
```

### Client Configuration

**Development**: `client/.env.development`
**Production**: `client/.env.production`

```bash
# WebRTC Configuration
WEBRTC_ICE_SERVERS=stun:stun.l.google.com:19302
VIDEO_WIDTH=640
VIDEO_HEIGHT=480
AUDIO_SAMPLE_RATE=16000
```

## Development Deployment

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-navigation-assistant
   ```

2. **Set up Python virtual environment**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download required models**:
   ```bash
   # YOLOv11 model (will be downloaded automatically on first run)
   # Vosk speech recognition model
   mkdir -p models
   # Download from https://alphacephei.com/vosk/models
   ```

4. **Start all components**:
   ```bash
   # Using Python script
   python scripts/start_development.py
   
   # Or using shell script (Linux/macOS)
   ./scripts/start.sh
   
   # Or using batch script (Windows)
   scripts\start.bat
   ```

### Individual Component Startup

**Backend only**:
```bash
cd backend
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend only**:
```bash
cd frontend
python -m http.server 3000
```

**Client only**:
```bash
cd client
python -m http.server 3001
```

### Development URLs

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Frontend: http://localhost:3000
- Client: http://localhost:3001

## Production Deployment

### System Preparation

1. **Create deployment user**:
   ```bash
   sudo useradd -m -s /bin/bash ai-nav
   sudo usermod -aG sudo ai-nav
   ```

2. **Set up directory structure**:
   ```bash
   sudo mkdir -p /opt/ai-navigation-assistant
   sudo chown ai-nav:ai-nav /opt/ai-navigation-assistant
   ```

3. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv nginx supervisor
   
   # CentOS/RHEL
   sudo yum install python3 python3-pip nginx supervisor
   ```

### Application Deployment

1. **Deploy application code**:
   ```bash
   cd /opt/ai-navigation-assistant
   git clone <repository-url> .
   ```

2. **Set up Python environment**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install gunicorn  # Production WSGI server
   ```

3. **Configure environment variables**:
   ```bash
   # Copy and customize production environment files
   cp backend/.env.production backend/.env
   cp frontend/.env.production frontend/.env
   cp client/.env.production client/.env
   
   # Edit files with production-specific values
   nano backend/.env
   ```

4. **Set up logging directories**:
   ```bash
   mkdir -p logs backend/logs
   chown -R ai-nav:ai-nav logs backend/logs
   ```

### Process Management with Supervisor

Create supervisor configuration files:

**Backend Service** (`/etc/supervisor/conf.d/ai-nav-backend.conf`):
```ini
[program:ai-nav-backend]
command=/opt/ai-navigation-assistant/backend/venv/bin/gunicorn main:app --bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker
directory=/opt/ai-navigation-assistant/backend
user=ai-nav
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/ai-navigation-assistant/logs/backend.log
environment=ENVIRONMENT="production"
```

**Frontend Service** (`/etc/supervisor/conf.d/ai-nav-frontend.conf`):
```ini
[program:ai-nav-frontend]
command=python3 -m http.server 3000 --bind 0.0.0.0
directory=/opt/ai-navigation-assistant/frontend
user=ai-nav
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/ai-navigation-assistant/logs/frontend.log
```

**Client Service** (`/etc/supervisor/conf.d/ai-nav-client.conf`):
```ini
[program:ai-nav-client]
command=python3 -m http.server 3001 --bind 0.0.0.0
directory=/opt/ai-navigation-assistant/client
user=ai-nav
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/ai-navigation-assistant/logs/client.log
```

Start services:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start ai-nav-backend ai-nav-frontend ai-nav-client
```

### Nginx Reverse Proxy

Create Nginx configuration (`/etc/nginx/sites-available/ai-navigation-assistant`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket connections
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Frontend
    location / {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Client interface
    location /client/ {
        proxy_pass http://localhost:3001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files and caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/ai-navigation-assistant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL/TLS Configuration

For production deployment with HTTPS (required for WebRTC):

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring and Logging

### Log Files

**Development**:
- Backend: `backend/logs/development.log`
- System: Console output

**Production**:
- Backend: `/opt/ai-navigation-assistant/logs/backend.log`
- Frontend: `/opt/ai-navigation-assistant/logs/frontend.log`
- Client: `/opt/ai-navigation-assistant/logs/client.log`
- Nginx: `/var/log/nginx/access.log`, `/var/log/nginx/error.log`

### Monitoring Endpoints

- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics` (if enabled)
- **System Status**: `GET /system/status`

### Log Rotation

Configure logrotate (`/etc/logrotate.d/ai-navigation-assistant`):

```
/opt/ai-navigation-assistant/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ai-nav ai-nav
    postrotate
        supervisorctl restart ai-nav-backend ai-nav-frontend ai-nav-client
    endscript
}
```

## Troubleshooting

### Common Issues

**1. Backend fails to start**:
```bash
# Check logs
tail -f backend/logs/development.log

# Verify Python dependencies
pip list | grep -E "(fastapi|uvicorn|opencv|ultralytics)"

# Check port availability
netstat -tlnp | grep :8000
```

**2. WebRTC connection issues**:
```bash
# Verify STUN server connectivity
curl -I https://stun.l.google.com:19302

# Check firewall settings
sudo ufw status
sudo iptables -L
```

**3. Camera/microphone access denied**:
- Ensure HTTPS is used (required for WebRTC)
- Check browser permissions
- Verify device availability

**4. High CPU/memory usage**:
```bash
# Monitor system resources
htop
# Check specific processes
ps aux | grep -E "(python|gunicorn)"
```

### Performance Optimization

**1. Backend optimization**:
- Use GPU for YOLOv11 inference: Set `YOLO_DEVICE=cuda`
- Adjust worker processes: Modify Gunicorn `--workers` parameter
- Enable model caching: Set appropriate cache sizes

**2. Network optimization**:
- Use CDN for static assets
- Enable gzip compression in Nginx
- Optimize WebRTC ICE servers

**3. Database optimization** (if applicable):
- Configure connection pooling
- Set up read replicas
- Implement caching layer

## Security Considerations

### Network Security

1. **Firewall Configuration**:
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22    # SSH
   sudo ufw allow 80    # HTTP
   sudo ufw allow 443   # HTTPS
   sudo ufw enable
   ```

2. **SSL/TLS**:
   - Use strong cipher suites
   - Enable HSTS headers
   - Regular certificate renewal

### Application Security

1. **Environment Variables**:
   - Never commit `.env` files to version control
   - Use strong secret keys
   - Rotate credentials regularly

2. **Input Validation**:
   - Validate all user inputs
   - Sanitize file uploads
   - Implement rate limiting

3. **Access Control**:
   - Implement authentication if required
   - Use CORS policies
   - Restrict API access

### Data Privacy

1. **Video/Audio Streams**:
   - No persistent storage of streams
   - Encrypted transmission (WebRTC DTLS)
   - Clear data retention policies

2. **Logging**:
   - Avoid logging sensitive data
   - Implement log sanitization
   - Secure log file access

## Backup and Recovery

### Backup Strategy

1. **Application Code**:
   - Version control (Git)
   - Regular repository backups

2. **Configuration Files**:
   ```bash
   # Backup configuration
   tar -czf config-backup-$(date +%Y%m%d).tar.gz \
       backend/.env frontend/.env client/.env \
       /etc/nginx/sites-available/ai-navigation-assistant \
       /etc/supervisor/conf.d/ai-nav-*.conf
   ```

3. **Logs and Data**:
   ```bash
   # Backup logs
   tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/
   ```

### Recovery Procedures

1. **Application Recovery**:
   ```bash
   # Stop services
   sudo supervisorctl stop ai-nav-backend ai-nav-frontend ai-nav-client
   
   # Restore code
   git pull origin main
   
   # Restart services
   sudo supervisorctl start ai-nav-backend ai-nav-frontend ai-nav-client
   ```

2. **Configuration Recovery**:
   ```bash
   # Restore configuration files
   tar -xzf config-backup-YYYYMMDD.tar.gz
   
   # Reload services
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo nginx -t && sudo systemctl reload nginx
   ```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Check log files for errors
   - Monitor system resources
   - Verify backup integrity

2. **Monthly**:
   - Update system packages
   - Review security logs
   - Performance optimization

3. **Quarterly**:
   - Update Python dependencies
   - Security audit
   - Disaster recovery testing

### Update Procedures

1. **Application Updates**:
   ```bash
   # Backup current version
   git tag backup-$(date +%Y%m%d)
   
   # Update code
   git pull origin main
   
   # Update dependencies
   source backend/venv/bin/activate
   pip install -r backend/requirements.txt
   
   # Restart services
   sudo supervisorctl restart ai-nav-backend
   ```

2. **System Updates**:
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade
   
   # Reboot if kernel updated
   sudo reboot
   ```

For additional support or questions, please refer to the project documentation or contact the development team.