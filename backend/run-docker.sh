#!/bin/bash
# AI Navigation Assistant Backend - Docker Run Script

set -e  # Exit on any error

echo "🚀 Running AI Navigation Assistant Backend Docker Container"
echo "=========================================================="

# Configuration
IMAGE_NAME="ai-navigation-backend"
TAG="latest"
CONTAINER_NAME="ai-navigation-backend"
PORT="8000"
ENVIRONMENT="production"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --simple)
            TAG="simple"
            CONTAINER_NAME="ai-navigation-backend-simple"
            echo "📦 Using simple version"
            shift
            ;;
        --dev)
            ENVIRONMENT="development"
            echo "🔧 Using development environment"
            shift
            ;;
        --port)
            PORT="$2"
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
            echo "  --simple         Use simple version (minimal dependencies)"
            echo "  --dev            Use development environment"
            echo "  --port PORT      Set port mapping (default: 8000)"
            echo "  --name NAME      Set container name"
            echo "  --detach, -d     Run in detached mode"
            echo "  --interactive, -it  Run in interactive mode"
            echo "  --build          Build image before running"
            echo "  --logs           Show container logs"
            echo "  --stop           Stop running container"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run normally"
            echo "  $0 --dev --port 8001  # Run in dev mode on port 8001"
            echo "  $0 --simple -d        # Run simple version detached"
            echo "  $0 --build            # Build and run"
            echo "  $0 --logs             # Show logs of running container"
            echo "  $0 --stop             # Stop running container"
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
    if [ "$TAG" = "simple" ]; then
        ./build-docker.sh --simple
    else
        ./build-docker.sh
    fi
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
    echo "💡 Build it first with: ./build-docker.sh"
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
echo "   Port: $PORT:8000"
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
DOCKER_CMD="$DOCKER_CMD -p $PORT:8000"
DOCKER_CMD="$DOCKER_CMD -e ENVIRONMENT=$ENVIRONMENT"
DOCKER_CMD="$DOCKER_CMD -e SERVER_HOST=0.0.0.0"
DOCKER_CMD="$DOCKER_CMD -e SERVER_PORT=8000"

# Add volume mounts for persistence
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/logs:/app/logs"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/models:/app/models"

# Add image name
DOCKER_CMD="$DOCKER_CMD $FULL_IMAGE_NAME"

echo "🚀 Starting container..."
echo "Command: $DOCKER_CMD"
echo ""

# Run the container
if eval $DOCKER_CMD; then
    echo ""
    if [ "$DETACH" = true ]; then
        echo "✅ Container started in detached mode!"
        echo "   Container: $CONTAINER_NAME"
        echo "   URL: http://localhost:$PORT"
        echo ""
        echo "📋 Useful commands:"
        echo "   docker logs -f $CONTAINER_NAME    # View logs"
        echo "   docker exec -it $CONTAINER_NAME /bin/bash  # Access container"
        echo "   docker stop $CONTAINER_NAME       # Stop container"
        echo "   ./run-docker.sh --logs            # Show logs"
        echo "   ./run-docker.sh --stop            # Stop container"
    else
        echo "✅ Container finished running"
    fi
else
    echo ""
    echo "❌ Failed to start container!"
    exit 1
fi