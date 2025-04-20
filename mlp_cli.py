#!/usr/bin/env python3
"""
Command-line interface for the Multi-Layer Perceptron Language Model.
This script provides a text-based interface for training and using the model.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multi_layer_perceptron import MultiLayerPerceptron

def print_header(text):
    """Print a header with the given text."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_section(text):
    """Print a section header with the given text."""
    print("\n" + "-" * 40)
    print(f" {text}")
    print("-" * 40)

def train_model(args):
    """Train a new model with the given parameters."""
    print_header("TRAINING NEW MODEL")
    
    # Parse hidden layers
    try:
        hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    except ValueError:
        print("Error: Invalid hidden layers format. Use comma-separated integers (e.g., 64,32).")
        return None
    
    # Create model
    model = MultiLayerPerceptron(
        context_size=args.context_size,
        hidden_layers=hidden_layers,
        learning_rate=args.learning_rate,
        n_iterations=args.iterations,
        random_state=42
    )
    
    # Load training text
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                training_text = f.read()
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return None
    else:
        print("Please enter training text (press Ctrl+D when finished):")
        training_text = sys.stdin.read()
    
    if not training_text.strip():
        print("Error: No training text provided.")
        return None
    
    print(f"\nTraining on {len(training_text.split())} words...")
    print(f"Model parameters: context_size={args.context_size}, hidden_layers={hidden_layers}, learning_rate={args.learning_rate}, iterations={args.iterations}")
    
    # Define progress callback
    def progress_callback(iteration, total_iterations, train_loss, val_loss, message=None):
        """
        Callback function to report training progress.

        Args:
            iteration (int): The current iteration number.
            total_iterations (int): The total number of iterations.
            train_loss (float): The training loss at the current iteration.
            val_loss (float): The validation loss at the current iteration.
            message (str, optional): Additional message to display.

        This function prints the training and validation loss at every 100th iteration or when a message is provided.
        """
        if iteration % 100 == 0 or message:
            status = f"Iteration {iteration}/{total_iterations} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            if message:
                status += f" - {message}"
            print(status)
    
    # Train the model
    try:
        model.fit(training_text, progress_callback=progress_callback)
        
        # Print model info
        print_section("MODEL INFORMATION")
        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"Architecture: Input({model.input_size}) → ", end="")
        for layer_size in model.hidden_layers:
            print(f"Hidden({layer_size}) → ", end="")
        print(f"Output({model.output_size})")
        
        # Plot training loss if requested
        if args.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(model.iteration_count, model.training_loss, label='Training Loss')
            plt.plot(model.iteration_count, model.validation_loss, label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Cross-Entropy Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_loss.png')
            print("\nTraining loss plot saved to 'training_loss.png'")
        
        # Save the model if requested
        if args.save:
            model.save_model(args.save)
            print(f"\nModel saved to '{args.save}'")
        
        return model
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def load_model(args):
    """Load a model from a file."""
    print_header("LOADING MODEL")
    
    try:
        model = MultiLayerPerceptron.load_model(args.load)
        print(f"Model loaded from '{args.load}'")
        
        # Print model info
        print_section("MODEL INFORMATION")
        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"Context size: {model.context_size}")
        print(f"Architecture: Input({model.input_size}) → ", end="")
        for layer_size in model.hidden_layers:
            print(f"Hidden({layer_size}) → ", end="")
        print(f"Output({model.output_size})")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_next_word(model, context, top_n=5):
    """Predict the next word given a context."""
    print_section("PREDICTING NEXT WORD")
    print(f"Context: '{context}'")
    
    try:
        # Get predictions
        predictions, info = model.get_top_predictions(context, top_n=top_n)
        
        # Show context adjustment info if any
        if info["adjustment_made"]:
            print("\nContext was adjusted:")
            if "adjustment_type" in info:
                if "padded_beginning" in info["adjustment_type"]:
                    print("- Context was too short. Added padding at the beginning.")
                if "truncated_beginning" in info["adjustment_type"]:
                    print("- Context was too long. Used only the most recent words.")
                if "replaced_unknown" in info["adjustment_type"]:
                    print("- Unknown words were replaced with known vocabulary.")
            
            if "unknown_words" in info:
                print(f"- Unknown words: {', '.join(info['unknown_words'])}")
            
            print(f"- Adjusted context: '{' '.join(info['adjusted_context'])}'")
        
        # Show predictions
        print("\nTop predictions:")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"{i}. '{word}' (probability: {prob:.6f})")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

def generate_text(model, context, n_words=20):
    """Generate text starting from the given context."""
    print_section("GENERATING TEXT")
    print(f"Initial context: '{context}'")
    print(f"Number of words to generate: {n_words}")
    
    try:
        # Generate text
        generated_words, info = model.predict_next_n_words(context, n=n_words)
        
        # Show context adjustment info if any
        if info["adjustment_made"]:
            print("\nContext was adjusted:")
            if info["adjusted_context"]:
                print(f"- Original context: '{' '.join(info['original_context'])}'")
                print(f"- Adjusted context: '{' '.join(info['adjusted_context'])}'")
            
            # Add information about the first prediction step
            first_step = info["prediction_steps"][0]
            if "adjustment_type" in first_step:
                if "padded_beginning" in first_step["adjustment_type"]:
                    print("- Context was too short. Added padding at the beginning.")
                if "truncated_beginning" in first_step["adjustment_type"]:
                    print("- Context was too long. Used only the most recent words.")
                if "replaced_unknown" in first_step["adjustment_type"]:
                    print("- Unknown words were replaced with known vocabulary.")
            
            if "unknown_words" in first_step:
                print(f"- Unknown words: {', '.join(first_step['unknown_words'])}")
        
        # Show generated text
        print("\nGenerated text:")
        print(info["full_text"])
        
    except Exception as e:
        print(f"Error during text generation: {str(e)}")

def interactive_mode(model):
    """Run an interactive session with the model."""
    print_header("INTERACTIVE MODE")
    print("Type 'exit' or 'quit' to end the session.")
    print("Commands:")
    print("  predict <context>     - Predict the next word")
    print("  generate <context>    - Generate text")
    print("  top <n> <context>     - Show top N predictions")
    print("  info                  - Show model information")
    print("  help                  - Show this help message")
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            if not parts:
                continue
                
            command = parts[0].lower()
            
            if command == 'help':
                print("Commands:")
                print("  predict <context>     - Predict the next word")
                print("  generate <context>    - Generate text")
                print("  top <n> <context>     - Show top N predictions")
                print("  info                  - Show model information")
                print("  help                  - Show this help message")
                
            elif command == 'info':
                print_section("MODEL INFORMATION")
                print(f"Vocabulary size: {len(model.vocabulary)}")
                print(f"Context size: {model.context_size}")
                print(f"Architecture: Input({model.input_size}) → ", end="")
                for layer_size in model.hidden_layers:
                    print(f"Hidden({layer_size}) → ", end="")
                print(f"Output({model.output_size})")
                
            elif command == 'predict':
                if len(parts) < 2:
                    print("Error: Missing context. Usage: predict <context>")
                    continue
                    
                context = parts[1]
                predict_next_word(model, context)
                
            elif command == 'generate':
                if len(parts) < 2:
                    print("Error: Missing context. Usage: generate <context>")
                    continue
                    
                context = parts[1]
                n_words = 20  # Default
                
                # Check if the first word is a number (number of words to generate)
                context_parts = context.split(maxsplit=1)
                if len(context_parts) > 1:
                    try:
                        n = int(context_parts[0])
                        context = context_parts[1]
                        n_words = n
                    except ValueError:
                        pass
                        
                generate_text(model, context, n_words)
                
            elif command == 'top':
                if len(parts) < 2:
                    print("Error: Missing arguments. Usage: top <n> <context>")
                    continue
                    
                args = parts[1].split(maxsplit=1)
                if len(args) < 2:
                    print("Error: Missing context. Usage: top <n> <context>")
                    continue
                    
                try:
                    top_n = int(args[0])
                    context = args[1]
                    predict_next_word(model, context, top_n)
                except ValueError:
                    print("Error: Invalid number. Usage: top <n> <context>")
                    
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for a list of commands.")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main function to parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description="Command-line interface for the Multi-Layer Perceptron Language Model")
    
    # Model parameters
    parser.add_argument('--context-size', type=int, default=2, help="Number of previous words to use as context")
    parser.add_argument('--hidden-layers', type=str, default="64,32", help="Comma-separated list of hidden layer sizes")
    parser.add_argument('--learning-rate', type=float, default=0.1, help="Learning rate for weight updates")
    parser.add_argument('--iterations', type=int, default=1000, help="Number of training iterations")
    
    # File I/O
    parser.add_argument('--input-file', type=str, help="Input file for training text")
    parser.add_argument('--save', type=str, help="Save model to file")
    parser.add_argument('--load', type=str, help="Load model from file")
    
    # Actions
    parser.add_argument('--predict', type=str, help="Predict next word for the given context")
    parser.add_argument('--generate', type=str, help="Generate text starting from the given context")
    parser.add_argument('--n-words', type=int, default=20, help="Number of words to generate")
    parser.add_argument('--top-n', type=int, default=5, help="Number of top predictions to show")
    parser.add_argument('--plot', action='store_true', help="Plot training loss")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Load or train model
    model = None
    if args.load:
        model = load_model(args)
    else:
        model = train_model(args)
    
    if not model:
        return
    
    # Perform requested actions
    if args.predict:
        predict_next_word(model, args.predict, args.top_n)
    
    if args.generate:
        generate_text(model, args.generate, args.n_words)
    
    # Run interactive mode if requested or if no other action was specified
    if args.interactive or (not args.predict and not args.generate and not args.save):
        interactive_mode(model)

if __name__ == "__main__":
    main()