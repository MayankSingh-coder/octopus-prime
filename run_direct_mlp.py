#!/usr/bin/env python3
"""
Direct launcher for the Multi-Layer Perceptron UI.
"""

import tkinter as tk
from multi_layer_perceptron_ui import MultiLayerPerceptronUI

def main():
    """
    Main function to run the application.
    """
    # Create the root window
    root = tk.Tk()
    root.title("Multi-Layer Perceptron Language Model")
    
    # Create the UI
    app = MultiLayerPerceptronUI(root)
    
    # Set a minimum window size for better usability
    root.minsize(1200, 800)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()