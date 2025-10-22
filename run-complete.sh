#!/bin/bash
# AI Navigation Assistant - Complete Application Docker Run Script

set -e  # Exit on any error

echo "🚀 Running AI Navigation Assistant - Complete Application"
echo "========================================================"

# Configuration
IMAGE_NAME="ai-navigation-complete"
TAG="latest"
CONTAINER_NAME="ai-navigation-complete"
FRONTEND_PORT="80"
BACKEND_PORT="8000"
ENVIRONMENT="production"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            ENVIRONMENT="development"
            echo "🔧 Using development environment"
            shift
            ;;
        --frontend-port)
            FRONTEND_PORT="$2"
            shift 2
            ;;
        --backend-port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --detach|-d)
            DETACH=true
            shift
            ;;
        --interactive|-it)
            INTERACTIVE=true
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --stop)
            STOP_CONTAINER=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev                Use development environment"
            echo "  --frontend-port PORT Set frontend port (default: 80)"
            echo "  --backend-port PORT  Set backend port (default: 8000)"
            echo "  --name NAME          Set container name"
            echo "  --detach, -d         Run in detached mode"
            echo "  --interactive, -it   Run in interactive mode"
            echo "  --build              Build image before running"
            echo "  --logs               Show container logs"
            echo "  --stop               Stop running container"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run normally"
            echo "  $0 --dev --frontend-port 3000        # Run in dev mode on port 3000"
            echo "  $0 --build -d                        # Build and run detached"
            echo "  $0 --logs                             # Show logs"
            echo "  $0 --stop                             # Stop container"
            echo ""
            echo "The complete application includes:"
            echo "  - Frontend served by Nginx on port $FRONTEND_PORT"
            echo "  - Backend API accessible via /api proxy"
            echo "  - WebSocket support via /ws proxy"
            echo "  - WebRTC endpoints via /webrtc proxy"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Stop container if requested
if [ "$STOP_CONTAINER" = true ]; then
    echo "🛑 Stopping container: $CONTAINER_NAME"
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
        echo "✅ Container stopped and removed"
    else
        echo "ℹ️  Container $CONTAINER_NAME is not running"
    fi
    exit 0
fi

# Show logs if requested
if [ "$SHOW_LOGS" = true ]; then
    echo "📋 Showing logs for container: $CONTAINER_NAME"
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker logs -f "$CONTAINER_NAME"
    else
        echo "❌ Container $CONTAINER_NAME is not running"
        exit 1
    fi
    exit 0
fi

# Build image if requested
if [ "$BUILD" = true ]; then
    echo "🔨 Building image first..."
    ./build-complete.sh
    echo ""
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check if image exists
if ! docker images -q "$FULL_IMAGE_NAME" | grep -q .; then
    echo "❌ Docker image not found: $FULL_IMAGE_NAME"
    echo "💡 Build it first with: ./build-complete.sh"
    echo "💡 Or use --build flag to build automatically"
    exit 1
fi

# Stop existing container if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo "🔄 Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
fi

echo "📋 Run Configuration:"
echo "   Image: $FULL_IMAGE_NAME"
echo "   Container: $CONTAINER_NAME"
echo "   Frontend Port: $FRONTEND_PORT:80"
echo "   Backend Port: $BACKEND_PORT:8000"
echo "   Environment: $ENVIRONMENT"
echo ""

# Prepare docker run command
DOCKER_CMD="docker run"

# Add run options
if [ "$DETACH" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -d"
elif [ "$INTERACTIVE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -it"
else
    DOCKER_CMD="$DOCKER_CMD -it"
fi

# Add port mapping and other options
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
DOCKER_CMD="$DOCKER_CMD -p $FRONTEND_PORT:80"
DOCKER_CMD="$DOCKER_CMD -p $BACKEND_PORT:8000"
DOCKER_CMD="$DOCKER_CMD -e ENVIRONMENT=$ENVIRONMENT"
DOCKER_CMD="$DOCKER_CMD -e FRONTEND_PORT=80"
DOCKER_CMD="$DOCKER_CMD -e SERVER_HOST=127.0.0.1"
DOCKER_CMD="$DOCKER_CMD -e SERVER_PORT=8000"

# Add volume mounts for persistence
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/backend/logs:/app/backend/logs"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/backend/models:/app/backend/models"

# Add image name
DOCKER_CMD="$DOCKER_CMD $FULL_IMAGE_NAME"

echo "🚀 Starting complete application..."
echo "Command: $DOCKER_CMD"
echo ""

# Run the container
if eval $DOCKER_CMD; then
    echo ""
    if [ "$DETACH" = true ]; then
        echo "✅ Complete application started in detached mode!"
        echo "   Container: $CONTAINER_NAME"
        echo ""
        echo "🌍 Application URLs:"
        echo "   Frontend: http://localhost:$FRONTEND_PORT"
        echo "   Backend API: http://localhost:$FRONTEND_PORT/api"
        echo "   Health Check: http://localhost:$FRONTEND_PORT/health"
        echo "   Backend Direct: http://localhost:$BACKEND_PORT (internal)"
        echo ""
        echo "📋 Useful commands:"
        echo "   docker logs -f $CONTAINER_NAME         # View logs"
        echo "   docker exec -it $CONTAINER_NAME /bin/bash  # Access container"
        echo "   docker stop $CONTAINER_NAME            # Stop container"
        echo "   ./run-complete.sh --logs               # Show logs"
        echo "   ./run-complete.sh --stop               # Stop container"
        echo ""
        echo "🔍 To test the application:"
        echo "   curl http://localhost:$FRONTEND_PORT/health"
        echo "   curl http://localhost:$FRONTEND_PORT/api/health"
    else
        echo "✅ Complete application finished running"
    fi
else
    echo ""
    echo "❌ Failed to start complete application!"
    exit 1
fi