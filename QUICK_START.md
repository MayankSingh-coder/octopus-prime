# Quick Start Guide

This guide will help you quickly get started with the MLP Language Model UI.

## Setup

### On macOS/Linux:

1. Open a terminal in the project directory
2. Run the setup script:
   ```bash
   ./setup_environment.sh
   ```
3. Follow the prompts to install dependencies

### On Windows:

1. Open Command Prompt in the project directory
2. Run the setup script:
   ```
   setup_environment.bat
   ```
3. Follow the prompts to install dependencies

## Running the UI

After setup, you can run any of the UI applications:

### Complete MLP UI (Recommended)

```bash
# On macOS/Linux
source mlp_env/bin/activate
python complete_mlp_ui.py

# On Windows
mlp_env\Scripts\activate
python complete_mlp_ui.py
```

This UI includes:
- Training with visualization
- Next word prediction with probabilities
- Text generation with temperature control
- Model saving and loading

### Basic MLP UI

```bash
python basic_mlp_ui.py
```

A simpler UI with basic functionality.

### Standard MLP UI

```bash
python run_standard_mlp.py
```

A launcher for the standard MLP UI.

## Documentation

For more detailed information, see:

- `README.md` - Overview of the project
- `UI_GUIDE.md` - Detailed guide for using the UI
- `take_screenshots.md` - Instructions for taking screenshots for documentation

## Troubleshooting

If you encounter issues:

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Check that you're using Python 3.6 or higher:
   ```bash
   python --version
   ```

3. Ensure tkinter is installed (required for the UI):
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On macOS: It should be included with Python
   - On Windows: It should be included with Python

4. If you get errors about missing modules when running the UI, try installing the optional dependencies:
   ```bash
   pip install gensim transformers torch tokenizers
   ```

## Next Steps

1. Load a text file for training
2. Experiment with different model parameters
3. Try generating text with different temperature settings
4. Save your trained model for future use