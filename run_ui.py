#!/usr/bin/env python3
"""
Run the Neural Network Language Model UI.

This script launches the complete UI for the neural network language model.
"""

import tkinter as tk
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the UI
from neural_network_lm.ui.complete_mlp_ui import CompleteMlpUI

def main():
    """
    Main function to run the UI.
    """
    root = tk.Tk()
    app = CompleteMlpUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()