#!/bin/bash

# Check if pip3 is available
if command -v pip3 &>/dev/null; then
    echo "Installing dependencies with pip3..."
    pip3 install -r requirements.txt
elif command -v pip &>/dev/null; then
    echo "Installing dependencies with pip..."
    pip install -r requirements.txt
else
    echo "Error: pip not found. Please install pip first."
    exit 1
fi

echo "Dependencies installed successfully!"