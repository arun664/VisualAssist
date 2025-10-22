#!/bin/bash
# AI Navigation Assistant - Complete Application Docker Build Script

set -e  # Exit on any error

echo "🐳 Building AI Navigation Assistant - Complete Application Docker Image"
echo "======================================================================"

# Configuration
IMAGE_NAME="ai-navigation-complete"
TAG="latest"
DOCKERFILE="Dockerfile"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tag TAG        Set image tag (default: latest)"
            echo "  --name NAME      Set image name (default: ai-navigation-complete)"
            echo "  --no-cache       Build without using cache"
            echo "  --help           Show this help message"
            echo ""
            echo "This builds a complete Docker image containing:"
            echo "  - Frontend (HTML/CSS/JS client)"
            echo "  - Backend (Python FastAPI server)"
            echo "  - Nginx (web server and reverse proxy)"
            echo "  - All dependencies and configurations"
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
echo "   Cache: $([ -n "$NO_CACHE" ] && echo "Disabled" || echo "Enabled")"
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

# Check if required directories exist
if [ ! -d "client" ]; then
    echo "❌ Client directory not found"
    exit 1
fi

if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found"
    exit 1
fi

# Create docker directory if it doesn't exist
mkdir -p docker

echo "🔍 Pre-build checks:"
echo "   ✓ Docker available"
echo "   ✓ Dockerfile exists"
echo "   ✓ Client directory exists"
echo "   ✓ Backend directory exists"
echo "   ✓ Docker config directory ready"
echo ""

# Build the Docker image
echo "🔨 Building complete Docker image..."
echo "Command: docker build $NO_CACHE -f $DOCKERFILE -t $FULL_IMAGE_NAME ."
echo ""

BUILD_START_TIME=$(date +%s)

if docker build $NO_CACHE -f "$DOCKERFILE" -t "$FULL_IMAGE_NAME" .; then
    BUILD_END_TIME=$(date +%s)
    BUILD_DURATION=$((BUILD_END_TIME - BUILD_START_TIME))
    
    echo ""
    echo "✅ Docker image built successfully!"
    echo "   Image: $FULL_IMAGE_NAME"
    echo "   Build time: ${BUILD_DURATION}s"
    
    # Show image info
    echo ""
    echo "📊 Image Information:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # Show image layers (top 10)
    echo ""
    echo "📦 Image Layers (top 10):"
    docker history "$FULL_IMAGE_NAME" --format "table {{.CreatedBy}}\t{{.Size}}" --no-trunc | head -11
    
    echo ""
    echo "🚀 To run the complete application:"
    echo "   docker run -p 80:80 -p 8000:8000 $FULL_IMAGE_NAME"
    echo ""
    echo "🌐 To run on custom port:"
    echo "   docker run -p 3000:80 -p 8001:8000 $FULL_IMAGE_NAME"
    echo ""
    echo "🔍 To run with environment variables:"
    echo "   docker run -p 80:80 -p 8000:8000 -e ENVIRONMENT=development $FULL_IMAGE_NAME"
    echo ""
    echo "🐛 To run interactively for debugging:"
    echo "   docker run -it -p 80:80 -p 8000:8000 $FULL_IMAGE_NAME /bin/bash"
    echo ""
    echo "📋 Application will be available at:"
    echo "   Frontend: http://localhost:80"
    echo "   Backend API: http://localhost:80/api"
    echo "   Health Check: http://localhost:80/health"
    
else
    echo ""
    echo "❌ Docker build failed!"
    echo ""
    echo "🔍 Troubleshooting tips:"
    echo "   1. Check Docker daemon is running"
    echo "   2. Ensure sufficient disk space"
    echo "   3. Try building with --no-cache flag"
    echo "   4. Check network connectivity for package downloads"
    echo "   5. Review build logs above for specific errors"
    exit 1
fi