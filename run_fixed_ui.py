#!/usr/bin/env python3
"""
Run the Neural Network Language Model UI with the fixed model.

This script launches the complete UI for the neural network language model
using the fixed version of the MultiLayerPerceptron model.
"""

import tkinter as tk
import sys
import os
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to run the UI with the fixed model.
    """
    # First, backup the original model file
    original_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "neural_network_lm", "models", "multi_layer_perceptron.py"
    )
    backup_file = original_file + ".backup"
    
    # Create backup if it doesn't exist
    if not os.path.exists(backup_file):
        shutil.copy2(original_file, backup_file)
        print(f"Created backup of original model at {backup_file}")
    
    # Replace the original model with the fixed version
    fixed_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "neural_network_lm", "models", "multi_layer_perceptron_fixed_final2.py"
    )
    shutil.copy2(fixed_file, original_file)
    print(f"Replaced model with fixed version")
    
    # Import the UI
    from neural_network_lm.ui.complete_mlp_ui import CompleteMlpUI
    
    # Create and run the UI
    root = tk.Tk()
    app = CompleteMlpUI(root)
    root.mainloop()
    
    # Restore the original model file when done
    shutil.copy2(backup_file, original_file)
    print(f"Restored original model file")

if __name__ == "__main__":
    main()