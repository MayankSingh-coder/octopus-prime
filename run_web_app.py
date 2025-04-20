#!/usr/bin/env python3
"""
Web Application Launcher for Language Model UI

This script launches the Flask web application for the Language Model UI.
It provides a web-based interface for training, testing, and using language models
with both standard MLP and attention-enhanced architectures.
"""

import os
import sys
import logging
import argparse
import webbrowser
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('language_model_web_ui')

def setup_model_directory():
    """
    Set up the directory for saving models and logs.
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_output')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"Created model directory at {model_dir}")
    return model_dir

def open_browser(port):
    """
    Open the web browser after a short delay to ensure the server is running.
    """
    def _open_browser():
        time.sleep(1.5)  # Wait for the server to start
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        logger.info(f"Opening browser at {url}")
    
    browser_thread = threading.Thread(target=_open_browser)
    browser_thread.daemon = True
    browser_thread.start()

def main():
    """
    Main function to run the web application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Language Model Web UI')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--no-browser', action='store_true', help='Do not automatically open browser')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Set up the model directory
    model_dir = setup_model_directory()
    
    try:
        # Try to import required modules
        from flask import Flask
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Import the app
        from app import app
        
        # Set model folder in app config
        app.config['MODEL_FOLDER'] = model_dir
        
        # Open browser if not disabled
        if not args.no_browser:
            open_browser(args.port)
        
        # Start the Flask app
        logger.info(f"Starting Language Model Web UI on port {args.port}")
        app.run(debug=args.debug, host='0.0.0.0', port=args.port)
        
    except ImportError as e:
        print("\n" + "=" * 80)
        
        if "flask" in str(e):
            print("ERROR: Flask is not installed.")
            print("=" * 80)
            print("\nTo install Flask:")
            print("pip install flask flask-cors")
        elif "numpy" in str(e):
            print("ERROR: NumPy is not installed.")
            print("=" * 80)
            print("\nTo install NumPy:")
            print("pip install numpy")
        elif "matplotlib" in str(e):
            print("ERROR: Matplotlib is not installed.")
            print("=" * 80)
            print("\nTo install Matplotlib:")
            print("pip install matplotlib")
        else:
            print(f"ERROR: Missing required module: {e}")
            print("=" * 80)
            print("\nPlease install all required dependencies:")
            print("pip install flask flask-cors numpy matplotlib")
        
        print("\nAlternatively, you can use the existing command-line example:")
        print("python3 run_mlp_ui.py")
        print("\nThis example will launch the desktop UI version of the application.")
        print("=" * 80 + "\n")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()