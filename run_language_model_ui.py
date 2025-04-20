#!/usr/bin/env python3
"""
Launcher for the Language Model UI application.
This script simply imports and runs the LanguageModelUI from language_model_ui.py
"""

import tkinter as tk
from language_model_ui import LanguageModelUI

def main():
    """
    Main function to run the application.
    """
    root = tk.Tk()
    app = LanguageModelUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()