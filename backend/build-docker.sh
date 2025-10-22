#!/bin/bash
# AI Navigation Assistant Backend - Docker Build Script

set -e  # Exit on any error

echo "🐳 Building AI Navigation Assistant Backend Docker Image"
echo "======================================================="

# Configuration
IMAGE_NAME="ai-navigation-backend"
TAG="latest"
DOCKERFILE="Dockerfile"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --simple)
            DOCKERFILE="Dockerfile.simple"
            TAG="simple"
            echo "📦 Using simplified Dockerfile"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --simple     Use Dockerfile.simple (minimal dependencies)"
            echo "  --tag TAG    Set image tag (default: latest)"
            echo "  --name NAME  Set image name (default: ai-navigation-backend)"
            echo "  --help       Show this help message"
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

echo "📋 Build Configuration:"
echo "   Image Name: ${FULL_IMAGE_NAME}"
echo "   Dockerfile: ${DOCKERFILE}"
echo "   Context: $(pwd)"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "❌ Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
echo "Command: docker build -f $DOCKERFILE -t $FULL_IMAGE_NAME ."
echo ""

if docker build -f "$DOCKERFILE" -t "$FULL_IMAGE_NAME" .; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo "   Image: $FULL_IMAGE_NAME"
    
    # Show image info
    echo ""
    echo "📊 Image Information:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    echo "🚀 To run the container:"
    echo "   docker run -p 8000:8000 $FULL_IMAGE_NAME"
    echo ""
    echo "🔍 To run with environment variables:"
    echo "   docker run -p 8000:8000 -e ENVIRONMENT=development $FULL_IMAGE_NAME"
    echo ""
    echo "🐛 To run interactively for debugging:"
    echo "   docker run -it -p 8000:8000 $FULL_IMAGE_NAME /bin/bash"
    
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi