#!/bin/bash
# Script to set up the environment for the MLP Language Model

# Create virtual environment if it doesn't exist
if [ ! -d "mlp_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv mlp_env
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source mlp_env/bin/activate

# Install required dependencies
echo "Installing required dependencies..."
pip install -r requirements.txt

# Ask if user wants to install optional dependencies
read -p "Do you want to install optional dependencies for enhanced features? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing optional dependencies..."
    pip install gensim transformers torch tokenizers
fi

# Create screenshots directory if it doesn't exist
if [ ! -d "screenshots" ]; then
    echo "Creating screenshots directory..."
    mkdir -p screenshots
fi

# Print success message
echo "Setup complete! You can now run the UI applications:"
echo "  python complete_mlp_ui.py    # Complete UI with all features"
echo "  python basic_mlp_ui.py       # Basic UI with core functionality"
echo "  python run_standard_mlp.py   # Standard MLP UI"
echo
echo "See README.md and UI_GUIDE.md for more information."