#!/bin/bash

# AI Navigation Assistant Startup Script
# Supports both development and production modes

set -e

# Default values
ENVIRONMENT="development"
COMPONENT="all"
VERBOSE=false
HELP=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show help
show_help() {
    cat << EOF
AI Navigation Assistant Startup Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Set environment (development|production) [default: development]
    -c, --component COMP     Start specific component (all|backend|frontend|client) [default: all]
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                                    # Start all components in development mode
    $0 -e production                      # Start all components in production mode
    $0 -c backend                         # Start only backend in development mode
    $0 -e production -c backend -v        # Start only backend in production mode with verbose output

COMPONENTS:
    all         Start all components (backend, frontend, client)
    backend     Start only the backend server
    frontend    Start only the frontend server
    client      Start only the client server

ENVIRONMENTS:
    development Use development configuration with hot reload
    production  Use production configuration with process management

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--component)
            COMPONENT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            print_color $RED "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    show_help
    exit 0
fi

# Validate environment
if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "production" ]]; then
    print_color $RED "Error: Environment must be 'development' or 'production'"
    exit 1
fi

# Validate component
if [[ "$COMPONENT" != "all" && "$COMPONENT" != "backend" && "$COMPONENT" != "frontend" && "$COMPONENT" != "client" ]]; then
    print_color $RED "Error: Component must be 'all', 'backend', 'frontend', or 'client'"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set verbose output
if [ "$VERBOSE" = true ]; then
    set -x
fi

print_color $BLUE "AI Navigation Assistant Startup"
print_color $BLUE "==============================="
print_color $YELLOW "Environment: $ENVIRONMENT"
print_color $YELLOW "Component: $COMPONENT"
print_color $YELLOW "Project Root: $PROJECT_ROOT"
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    print_color $RED "Error: Python 3 is required but not installed"
    exit 1
fi

# Check if virtual environment exists for backend
BACKEND_VENV="$PROJECT_ROOT/backend/venv"
if [ ! -d "$BACKEND_VENV" ]; then
    print_color $YELLOW "Warning: Backend virtual environment not found at $BACKEND_VENV"
    print_color $YELLOW "Consider running: python3 -m venv $BACKEND_VENV && source $BACKEND_VENV/bin/activate && pip install -r backend/requirements.txt"
fi

# Function to start backend
start_backend() {
    print_color $GREEN "Starting backend server..."
    
    cd "$PROJECT_ROOT/backend"
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    
    if [ "$ENVIRONMENT" = "development" ]; then
        # Development mode with hot reload
        python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
    else
        # Production mode with Gunicorn
        if command -v gunicorn &> /dev/null; then
            gunicorn main:app \
                --bind 0.0.0.0:8000 \
                --workers 4 \
                --worker-class uvicorn.workers.UvicornWorker \
                --max-requests 1000 \
                --max-requests-jitter 100 \
                --timeout 30 \
                --keepalive 10 \
                --log-level info \
                --access-logfile logs/access.log \
                --error-logfile logs/error.log \
                --pid logs/backend.pid
        else
            print_color $YELLOW "Warning: Gunicorn not found, using uvicorn for production"
            python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
        fi
    fi
}

# Function to start frontend
start_frontend() {
    print_color $GREEN "Starting frontend server..."
    
    cd "$PROJECT_ROOT/frontend"
    
    if [ "$ENVIRONMENT" = "development" ]; then
        python3 -m http.server 3000
    else
        python3 -m http.server 3000 --bind 0.0.0.0
    fi
}

# Function to start client
start_client() {
    print_color $GREEN "Starting client server..."
    
    cd "$PROJECT_ROOT/client"
    
    if [ "$ENVIRONMENT" = "development" ]; then
        python3 -m http.server 3001
    else
        python3 -m http.server 3001 --bind 0.0.0.0
    fi
}

# Function to start all components
start_all() {
    print_color $GREEN "Starting all components..."
    
    if [ "$ENVIRONMENT" = "development" ]; then
        python3 "$SCRIPT_DIR/start_development.py"
    else
        python3 "$SCRIPT_DIR/start_production.py"
    fi
}

# Create necessary directories
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/backend/logs"
mkdir -p "$PROJECT_ROOT/backend/models"

# Start the requested component(s)
case $COMPONENT in
    "all")
        start_all
        ;;
    "backend")
        start_backend
        ;;
    "frontend")
        start_frontend
        ;;
    "client")
        start_client
        ;;
esac