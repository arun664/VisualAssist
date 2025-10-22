#!/bin/bash
# AI Navigation Assistant - Docker Startup Script
# Starts both frontend (nginx) and backend (python) services

set -e

echo "🚀 Starting AI Navigation Assistant - Complete Application"
echo "=========================================================="

# Function to handle shutdown gracefully
cleanup() {
    echo "🛑 Shutting down services..."
    
    # Stop nginx
    if [ ! -z "$NGINX_PID" ]; then
        echo "Stopping nginx (PID: $NGINX_PID)..."
        kill -TERM $NGINX_PID 2>/dev/null || true
        wait $NGINX_PID 2>/dev/null || true
    fi
    
    # Stop backend
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill -TERM $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
    fi
    
    echo "✅ Services stopped gracefully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Create necessary directories
mkdir -p /tmp/nginx /var/log/nginx
touch /var/log/nginx/access.log /var/log/nginx/error.log

# Update client configuration for containerized environment
echo "🔧 Configuring client for containerized environment..."
cat > /app/client/docker-config.js << 'EOF'
// Docker container configuration override
window.AI_NAV_CONFIG = {
  getBackendUrl: () => {
    // In Docker container, backend is available via nginx proxy
    return window.location.origin + '/api';
  },
  
  getWebSocketUrl: () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return protocol + '//' + window.location.host + '/ws';
  },
  
  getVideoStreamUrl: () => {
    return window.location.origin + '/processed_video_stream';
  },
  
  isDevelopment: () => false,
  isProduction: () => true,
  isHttpsPage: () => window.location.protocol === 'https:',
  hasMixedContentIssue: () => false
};

console.log('🐳 Docker container configuration loaded');
console.log('Backend URL:', window.AI_NAV_CONFIG.getBackendUrl());
console.log('WebSocket URL:', window.AI_NAV_CONFIG.getWebSocketUrl());
EOF

# Update client HTML to use docker configuration
if [ -f "/app/client/index.html" ]; then
    # Add docker config script before the existing config script
    sed -i 's|<script src="../config.js"></script>|<script src="docker-config.js"></script>\n    <script src="../config.js"></script>|g' /app/client/index.html
fi

# Test nginx configuration
echo "🔍 Testing nginx configuration..."
nginx -t -c /etc/nginx/nginx.conf

# Start nginx in background
echo "🌐 Starting nginx (frontend server)..."
nginx -c /etc/nginx/nginx.conf &
NGINX_PID=$!
echo "✅ Nginx started (PID: $NGINX_PID)"

# Wait a moment for nginx to start
sleep 2

# Check if nginx is running
if ! kill -0 $NGINX_PID 2>/dev/null; then
    echo "❌ Nginx failed to start"
    exit 1
fi

# Start backend in background
echo "🐍 Starting Python backend..."
cd /app/backend

# Set Python path
export PYTHONPATH=/app/backend

# Start the backend using the entrypoint script
python entrypoint.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to start
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

# Display service status
echo ""
echo "📊 Service Status:"
echo "   Frontend (nginx): Running on port 80 (PID: $NGINX_PID)"
echo "   Backend (python): Running on port 8000 (PID: $BACKEND_PID)"
echo ""
echo "🌍 Application URLs:"
echo "   Frontend: http://localhost:80"
echo "   Backend API: http://localhost:80/api"
echo "   Health Check: http://localhost:80/health"
echo "   Backend Direct: http://localhost:8000 (internal)"
echo ""

# Function to check service health
check_health() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "🏥 Checking $service health..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service is healthy"
            return 0
        fi
        
        echo "⏳ $service health check attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service health check failed after $max_attempts attempts"
    return 1
}

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check frontend health
if ! check_health "Frontend" "http://localhost:80/health"; then
    echo "⚠️ Frontend health check failed, but continuing..."
fi

# Check backend health
if ! check_health "Backend" "http://localhost:8000/health"; then
    echo "⚠️ Backend health check failed, but continuing..."
fi

echo ""
echo "🎉 AI Navigation Assistant is ready!"
echo "   Access the application at: http://localhost:80"
echo ""
echo "📋 Useful commands:"
echo "   docker logs -f <container_name>  # View logs"
echo "   docker exec -it <container_name> /bin/bash  # Access container"
echo ""
echo "🔍 Monitoring services... (Press Ctrl+C to stop)"

# Monitor both processes
while true; do
    # Check if nginx is still running
    if ! kill -0 $NGINX_PID 2>/dev/null; then
        echo "❌ Nginx process died, restarting..."
        nginx -c /etc/nginx/nginx.conf &
        NGINX_PID=$!
        sleep 2
    fi
    
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process died, restarting..."
        cd /app/backend
        python entrypoint.py &
        BACKEND_PID=$!
        sleep 5
    fi
    
    # Wait before next check
    sleep 30
done