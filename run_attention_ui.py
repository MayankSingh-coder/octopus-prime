#!/usr/bin/env python3
"""
Enhanced launcher for the Attention-Enhanced Language Model UI application.
This script imports and runs the AttentionPerceptronUI which extends the standard
MultiLayerPerceptronUI with support for attention mechanisms.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('attention_ui')

def setup_model_directory():
    """
    Set up the directory for saving models and logs.
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_output')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"Created model directory at {model_dir}")
    return model_dir

def main():
    """
    Main function to run the application with enhanced configuration.
    """
    # Set up the model directory
    model_dir = setup_model_directory()
    
    # Try to import tkinter
    try:
        import tkinter as tk
        from attention_perceptron_ui import AttentionPerceptronUI
        
        # Configure the UI
        root = tk.Tk()
        root.title("Attention-Enhanced Language Model")
        
        # Set a minimum window size for better usability
        root.minsize(1200, 800)
        
        # Initialize the UI with the model directory
        app = AttentionPerceptronUI(root)
        
        # Configure default paths for saving/loading models
        app.default_model_path = model_dir
        
        # Set some reasonable defaults for the model parameters
        app.context_size_var.set(3)  # Increase context size for better predictions
        app.hidden_layers_var.set("128,64")  # Larger hidden layers for better learning
        app.learning_rate_var.set(0.05)  # Adjusted learning rate
        app.iterations_var.set(1500)  # More iterations for better training
        
        # Set attention-specific defaults
        app.model_type_var.set("attention")  # Default to attention model
        app.attention_dim_var.set(40)  # Dimension of attention space
        app.num_heads_var.set(2)  # Number of attention heads
        app.attention_dropout_var.set(0.1)  # Dropout rate for attention
        
        # Start the main loop
        logger.info("Starting Attention-Enhanced Language Model UI application")
        root.mainloop()
    
    except ImportError as e:
        print("\n" + "=" * 80)
        
        if "_tkinter" in str(e):
            print("ERROR: Tkinter is not installed or not properly configured.")
            print("=" * 80)
            print("\nTo install tkinter:")
            print("- On macOS: Install Python with Homebrew: 'brew install python-tk'")
            print("- On Ubuntu/Debian: 'sudo apt-get install python3-tk'")
            print("- On Windows: Reinstall Python and select the tcl/tk option")
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
            print("pip install numpy matplotlib")
            print("pip install tkinter  # For the UI")
        
        print("\nAlternatively, you can use the existing command-line example:")
        print("python3 attention_example.py")
        print("\nThis example will train both standard and attention-enhanced models")
        print("and compare their performance on text generation tasks.")
        print("=" * 80 + "\n")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()