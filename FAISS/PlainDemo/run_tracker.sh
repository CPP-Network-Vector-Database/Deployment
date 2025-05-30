#!/bin/bash

# FAISS Performance Tracker Runner Script
# This script sets up the environment and runs the FAISS performance tracking

set -e  # Exit on any error

echo "======================================"
echo "FAISS Performance Tracker Setup"
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
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing core dependencies..."
    pip install pandas faiss-cpu numpy psutil matplotlib sentence-transformers torch transformers scikit-learn scipy
fi

# Check if the dataset file exists
DATASET_PATH="ip_flow_dataset.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "Warning: Dataset file not found at $DATASET_PATH"
    echo "Please update the csv_path variable in the script to point to your dataset file."
    echo "You can also set the CSV_PATH environment variable:"
    echo "export CSV_PATH='/path/to/your/dataset.csv'"
    echo ""
fi

# Create results directory
echo "Creating results directory..."
mkdir -p PipelineResults

# Set memory limit warnings (optional)
export PYTHONWARNINGS="ignore::UserWarning"

# Print system info
echo "======================================"
echo "System Information:"
echo "======================================"
echo "Python executable: $(which python)"
echo "Available memory: $(free -h | grep Mem | awk '{print $2}' 2>/dev/null || echo 'N/A')"
echo "CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'N/A')"
echo ""

# Run the performance tracker
echo "======================================"
echo "Starting FAISS Performance Tracking..."
echo "======================================"
echo "This may take a while depending on your system..."
echo "Results will be saved in the PipelineResults directory"
echo ""

# Check if custom CSV path is provided
if [ ! -z "$CSV_PATH" ]; then
    echo "Using custom dataset path: $CSV_PATH"
    # For now, we'll just run the original script
fi

# Run the main script
python3 faiss_performance_tracker.py

# Check if script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Performance Tracking Completed!"
    echo "======================================"
    echo "Results saved in:"
    echo "  - Individual model CSVs: PipelineResults/*_metrics.csv"
    echo "  - Combined CSV: PipelineResults/all_models_performance_metrics.csv" 
    echo "  - Performance plots: PipelineResults/*.png"
    echo ""
    echo "Files created:"
    ls -la PipelineResults/
else
    echo ""
    echo "======================================"
    echo "Error: Script execution failed!"
    echo "======================================"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
echo "Script completed successfully!"
echo "Virtual environment can be reactivated with: source venv/bin/activate"