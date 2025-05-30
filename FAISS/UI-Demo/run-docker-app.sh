#!/bin/bash

# FAISS Performance Tracker Docker Runner
# This script builds and runs the FAISS Streamlit application in Docker

set -e  # Exit on any error

echo "======================================"
echo "FAISS Docker Application Setup"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Docker is installed
if ! command_exists docker; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
COMPOSE_CMD=""
if command_exists docker-compose; then
    COMPOSE_CMD="docker-compose"
    echo "Using docker-compose (v1)"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
    echo "Using docker compose (v2)"
else
    echo "Warning: Docker Compose not found. Will use docker run instead."
fi

# Check Docker version
DOCKER_VERSION=$(docker --version)
echo "Docker version: $DOCKER_VERSION"

# Verify required files exist
echo "Checking required files..."

if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please ensure requirements.txt exists."
    exit 1
fi

APP_FILE="pipeline+ui.py"
if [ ! -f "$APP_FILE" ]; then
    echo "Error: Main app file $APP_FILE not found. Please ensure it exists."
    exit 1
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data logs

# Create sample dataset if it doesn't exist
SAMPLE_DATASET="ip_flow_dataset.csv"
if [ ! -f "$SAMPLE_DATASET" ]; then
    echo "Creating sample network dataset..."
    cat > "$SAMPLE_DATASET" << EOF
frame.number,frame.time,ip.src,ip.dst,tcp.srcport,tcp.dstport,_ws.col.protocol,frame.len
1,0.000000,192.168.1.1,192.168.1.2,80,443,TCP,1460
2,0.001000,10.0.0.1,10.0.0.2,53,8080,UDP,512
3,0.002000,172.16.0.1,172.16.0.2,80,8080,HTTP,1024
4,0.003000,192.168.0.1,192.168.0.2,443,443,HTTPS,1200
5,0.004000,10.10.10.1,10.10.10.2,22,22,SSH,64
6,0.005000,192.168.1.10,192.168.1.20,3389,3389,RDP,1400
7,0.006000,172.16.1.1,172.16.1.2,21,21,FTP,256
8,0.007000,10.0.1.1,10.0.1.2,25,25,SMTP,800
9,0.008000,192.168.2.1,192.168.2.2,110,110,POP3,400
10,0.009000,172.16.2.1,172.16.2.2,143,143,IMAP,600
EOF
fi

# Create .dockerignore if it doesn't exist
if [ ! -f ".dockerignore" ]; then
    echo "Creating .dockerignore..."
    cat > .dockerignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Logs
logs/
*.log

# Other
README.md
*.md
EOF
fi

# Print system info
echo "======================================"
echo "System Information:"
echo "======================================"
echo "Docker executable: $(which docker)"
echo "Available disk space: $(df -h . | tail -1 | awk '{print $4}' 2>/dev/null || echo 'N/A')"
echo "CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'N/A')"
echo ""

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker stop faiss-performance-tracker 2>/dev/null || true
docker rm faiss-performance-tracker 2>/dev/null || true

# Build Docker image
echo "======================================"
echo "Building Docker Image..."
echo "======================================"
echo "This may take several minutes for the first build..."
echo ""

docker build -t faiss-performance-tracker .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully!"
else
    echo "✗ Docker build failed!"
    exit 1
fi

# Run the container
echo ""
echo "======================================"
echo "Starting Docker Container..."
echo "======================================"

if [ ! -z "$COMPOSE_CMD" ] && [ -f "docker-compose.yml" ]; then
    echo "Using Docker Compose to start the application..."
    $COMPOSE_CMD up -d
    
    # Check container status
    sleep 3
    if $COMPOSE_CMD ps | grep -q "Up"; then
        echo "Container started successfully with Docker Compose!"
        CONTAINER_NAME=$($COMPOSE_CMD ps --services)
    else
        echo "Failed to start container with Docker Compose!"
        $COMPOSE_CMD logs
        exit 1
    fi
else
    echo "Using docker run to start the application..."
    docker run -d \
        --name faiss-performance-tracker \
        -p 8501:8501 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        --restart unless-stopped \
        faiss-performance-tracker
    
    # Check container status
    sleep 3
    if docker ps | grep -q faiss-performance-tracker; then
        echo "✓ Container started successfully!"
        CONTAINER_NAME="faiss-performance-tracker"
    else
        echo "✗ Failed to start container!"
        docker logs faiss-performance-tracker
        exit 1
    fi
fi

echo ""
echo "======================================"
echo "Application Started Successfully!"
echo "======================================"
echo "Access the application at: http://localhost:8501"
echo "Container name: $CONTAINER_NAME"
echo ""
echo "Useful commands:"
echo "  View logs: docker logs -f $CONTAINER_NAME"
echo "  Stop app:  docker stop $CONTAINER_NAME"
echo "  Restart:   docker restart $CONTAINER_NAME"
echo ""

# Wait for application to be ready
echo "Waiting for application to start..."
sleep 10

# Check if application is responding
if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    echo "✓ Application is responding and ready!"
else
    echo "⚠ Application may still be starting up..."
    echo "  Check logs with: docker logs $CONTAINER_NAME"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo "The FAISS Performance Tracker is now running in Docker."
echo "Open your browser and navigate to: http://localhost:8501"
echo ""
echo "To stop the application:"
if [ ! -z "$COMPOSE_CMD" ] && [ -f "docker-compose.yml" ]; then
    echo "  $COMPOSE_CMD down"
else
    echo "  docker stop faiss-performance-tracker"
fi
echo ""
echo "Container will automatically restart unless stopped manually."