#!/usr/bin/env python3
"""
Enhanced launcher for the Language Model UI application.
This script supports both standard Multi-Layer Perceptron and Attention-Enhanced models.
It automatically detects available dependencies and provides appropriate UI options.
"""

import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('language_model_ui')

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Language Model UI')
    parser.add_argument('--model', choices=['standard', 'attention', 'single'], default='auto',
                        help='Model type to use (standard, attention, single, or auto-detect)')
    args = parser.parse_args()
    
    # Set up the model directory
    model_dir = setup_model_directory()
    
    try:
        # Try to import required modules
        import tkinter as tk
        import numpy as np
        
        # Try to import matplotlib, but continue if not available
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
            print("Warning: Matplotlib not available. Some visualization features will be disabled.")
        
        from multi_layer_perceptron_ui import MultiLayerPerceptronUI
        
        # Check if attention model is available
        attention_available = False
        try:
            from attention_perceptron import AttentionPerceptron
            from attention_perceptron_ui import AttentionPerceptronUI
            attention_available = True
            logger.info("Attention model support detected")
        except ImportError as e:
            logger.warning(f"Attention model support not available: {e}")
        
        # Check if single-layer perceptron model is available
        single_layer_available = False
        try:
            from single_layer_perceptron import SingleLayerPerceptron
            from single_layer_perceptron_ui import SingleLayerPerceptronUI
            single_layer_available = True
            logger.info("Single-layer perceptron model support detected")
        except ImportError as e:
            logger.warning(f"Single-layer perceptron model support not available: {e}")
        
        # Configure the UI
        root = tk.Tk()
        
        # Determine which UI to use based on arguments and availability
        use_attention = False
        use_single_layer = False
        
        if args.model == 'auto':
            # Prioritize single-layer perceptron if available
            if single_layer_available:
                use_single_layer = True
            elif attention_available:
                use_attention = True
        elif args.model == 'attention':
            if not attention_available:
                print("Warning: Attention model requested but not available. Using standard model.")
            else:
                use_attention = True
        elif args.model == 'single':
            if not single_layer_available:
                print("Warning: Single-layer perceptron model requested but not available. Using standard model.")
            else:
                use_single_layer = True
        
        # Initialize the appropriate UI
        if use_single_layer:
            root.title("Single-Layer Perceptron Language Model")
            app = SingleLayerPerceptronUI(root)
            logger.info("Using Single-Layer Perceptron UI")
        elif use_attention:
            root.title("Enhanced Language Model with Attention")
            app = AttentionPerceptronUI(root)
            logger.info("Using Attention-Enhanced UI")
        else:
            root.title("Enhanced Multi-Layer Perceptron Language Model")
            app = MultiLayerPerceptronUI(root)
            logger.info("Using Standard MLP UI")
        
        # Set a minimum window size for better usability
        root.minsize(1200, 800)
        
        # Configure default paths for saving/loading models
        app.default_model_path = model_dir
        
        # Set some reasonable defaults for the model parameters
        app.context_size_var.set(3)  # Increase context size for better predictions
        if hasattr(app, 'hidden_layers_var'):
            app.hidden_layers_var.set("128,64")  # Larger hidden layers for better learning
        app.learning_rate_var.set(0.05)  # Adjusted learning rate
        app.iterations_var.set(1500)  # More iterations for better training
        
        # Set single-layer specific defaults if available
        if use_single_layer and hasattr(app, 'hidden_size_var'):
            app.hidden_size_var.set(128)  # Default hidden layer size
        
        # Set attention-specific defaults if available
        if use_attention and hasattr(app, 'model_type_var'):
            app.model_type_var.set("attention")  # Default to attention model
            app.attention_dim_var.set(40)  # Default attention dimension
            app.num_heads_var.set(2)  # Default number of attention heads
            app.attention_dropout_var.set(0.1)  # Default attention dropout
        
        # Start the main loop
        logger.info("Starting Language Model UI application")
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
            # Continue without matplotlib
            print("WARNING: Matplotlib is not installed. Using simplified UI without visualization features.")
            print("=" * 80)
            print("\nTo enable visualization features, install Matplotlib:")
            print("pip install matplotlib")
            print("\nContinuing with simplified UI...")
            
            # Try to import required modules again, but without matplotlib
            try:
                import tkinter as tk
                import numpy as np
                from no_matplotlib_ui import SimplePerceptronUI
                
                # Check if single-layer perceptron model is available
                single_layer_available = False
                try:
                    from simple_perceptron import SimpleSingleLayerPerceptron
                    single_layer_available = True
                    print("Single-layer perceptron model support detected")
                except ImportError as e:
                    print(f"Single-layer perceptron model support not available: {e}")
                
                # Configure the UI
                root = tk.Tk()
                
                # Initialize the UI
                if single_layer_available and args.model == 'single':
                    root.title("Single-Layer Perceptron Language Model (Simplified UI)")
                    app = SimplePerceptronUI(root, model_class=SimpleSingleLayerPerceptron, is_single_layer=True)
                    print("Using Single-Layer Perceptron with simplified UI")
                else:
                    from simple_perceptron import SimpleMultiLayerPerceptron
                    root.title("Multi-Layer Perceptron Language Model (Simplified UI)")
                    app = SimplePerceptronUI(root, model_class=SimpleMultiLayerPerceptron, is_single_layer=False)
                    print("Using Standard MLP with simplified UI")
                
                # Set a minimum window size for better usability
                root.minsize(1200, 800)
                
                # Configure default paths for saving/loading models
                app.default_model_path = model_dir
                
                # Set some reasonable defaults for the model parameters
                app.context_size_var.set(3)
                if hasattr(app, 'hidden_layers_var'):
                    app.hidden_layers_var.set("128,64")
                app.learning_rate_var.set(0.05)
                app.iterations_var.set(1500)
                
                # Set single-layer specific defaults if available
                if args.model == 'single' and hasattr(app, 'hidden_size_var'):
                    app.hidden_size_var.set(128)
                
                # Start the main loop
                print("Starting Language Model UI application with simplified UI")
                root.mainloop()
                return
            except Exception as inner_e:
                print(f"Error starting the application with simplified UI: {inner_e}")
                import traceback
                traceback.print_exc()
                # Continue to the error message below
        else:
            print(f"ERROR: Missing required module: {e}")
            print("=" * 80)
            print("\nPlease install all required dependencies:")
            print("pip install numpy matplotlib")
            print("pip install tkinter  # For the UI")
        
        print("\nAlternatively, you can use the existing command-line example:")
        print("python3 attention_example.py")
        print("\nOr try the single-layer perceptron model:")
        print("python3 run_mlp_ui.py --model single")
        print("\nThese examples will train different types of models")
        print("and compare their performance on text generation tasks.")
        print("=" * 80 + "\n")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()