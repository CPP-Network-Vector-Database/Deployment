#!/bin/bash

# FAISS Performance Tracker Streamlit App Runner
# This script sets up the environment and runs the Streamlit application

set -e  # Exit on any error

echo "======================================"
echo "FAISS Streamlit App Setup"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Error: Python3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found. Please ensure requirements.txt exists in the current directory."
    exit 1
fi

# Check if the main app file exists
APP_FILE="pipeline+ui.py"
if [ ! -f "$APP_FILE" ]; then
    echo "Error: Main app file $APP_FILE not found. Please ensure it exists in the current directory."
    exit 1
fi

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
11,0.010000,192.168.3.1,192.168.3.2,80,8080,HTTP,900
12,0.011000,10.1.1.1,10.1.1.2,443,443,HTTPS,1300
13,0.012000,172.17.0.1,172.17.0.2,22,2222,SSH,128
14,0.013000,192.168.4.1,192.168.4.2,53,53,DNS,64
15,0.014000,10.2.2.1,10.2.2.2,80,80,HTTP,512
EOF
fi

# Check if custom dataset path is provided via environment variable
if [ ! -z "$CSV_PATH" ]; then
    if [ -f "$CSV_PATH" ]; then
        echo "Custom dataset found at: $CSV_PATH"
        echo "Note: Update the csv_path in the Streamlit app to use this file."
    else
        echo "Warning: Custom CSV_PATH specified but file not found: $CSV_PATH"
        echo "Using sample dataset instead."
    fi
fi

# Setting memory limit warnings
export PYTHONWARNINGS="ignore::UserWarning"

echo "======================================"
echo "System Information:"
echo "======================================"
echo "Python executable: $(which python)"
echo "Available memory: $(free -h | grep Mem | awk '{print $2}' 2>/dev/null || echo 'N/A')"
echo "CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'N/A')"
echo "Streamlit version: $(streamlit --version 2>/dev/null || echo 'Not installed')"
echo ""

# Verify key packages are installed
echo "Verifying package installations..."
python3 -c "
try:
    import streamlit
    import pandas
    import faiss
    import numpy
    import sentence_transformers
    import plotly
    print('All core packages verified successfully')
except ImportError as e:
    print(f'Package import failed: {e}')
    exit(1)
"

echo ""
echo "======================================"
echo "Starting Streamlit Application..."
echo "======================================"
echo "Application will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

# Set Streamlit configuration
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the Streamlit app
streamlit run "$APP_FILE" --server.port 8501 --server.headless true

# Check if script completed successfully (IMPT: this will only run if streamlit is stopped)
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Application Stopped Successfully!"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "Error: Application encountered an error!"
    echo "======================================"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
echo "Script completed successfully!"
echo "Reactivate with: source venv/bin/activate"/home/pes1ug22am100/Deployment/FAISS/UI-Demo/pipeline+ui.py