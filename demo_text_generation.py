#!/usr/bin/env python3
"""
Demo script to show improved text generation with the fixed model.
"""

import sys
import os
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to demonstrate text generation with the fixed model.
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
        "neural_network_lm", "models", "multi_layer_perceptron_fixed2.py"
    )
    shutil.copy2(fixed_file, original_file)
    print(f"Replaced model with fixed version")
    
    # Import the model
    from neural_network_lm.models.multi_layer_perceptron import MultiLayerPerceptron
    
    # Create a model
    model = MultiLayerPerceptron(
        context_size=2,
        embedding_dim=20,
        hidden_layers=[32, 16],
        learning_rate=0.01,
        n_iterations=10,  # Small number for testing
        random_state=42
    )
    
    # Test text generation with an untrained model
    print("\n=== Text Generation with Untrained Model ===")
    print("This should now generate reasonable text even without training")
    
    # Try with the problematic input
    context = "hii smartcoin"
    n_words = 10
    temperature = 1.0
    
    print(f"\nContext: '{context}'")
    print(f"Generating {n_words} words with temperature {temperature}...")
    
    # Generate text
    generated_words, info = model.predict_next_n_words(context, n=n_words, temperature=temperature)
    
    # Print generated text
    print("\nGenerated text:")
    print(f"{context} {' '.join(generated_words)}")
    
    # Print generation steps
    print("\nGeneration steps:")
    for i, step in enumerate(info["predictions"]):
        print(f"Step {i+1}:")
        print(f"  Context: {' '.join(step['context'])}")
        print(f"  Predicted: {step['predicted_word']}")
        if "note" in step:
            print(f"  Note: {step['note']}")
        if "fallback" in step and step["fallback"]:
            print(f"  Fallback: True")
            if "fallback_source" in step:
                print(f"  Fallback source: {step['fallback_source']}")
        print()
    
    # Try with another input
    context = "loan application"
    print(f"\nContext: '{context}'")
    print(f"Generating {n_words} words with temperature {temperature}...")
    
    # Generate text
    generated_words, info = model.predict_next_n_words(context, n=n_words, temperature=temperature)
    
    # Print generated text
    print("\nGenerated text:")
    print(f"{context} {' '.join(generated_words)}")
    
    # Restore the original model file when done
    shutil.copy2(backup_file, original_file)
    print(f"\nRestored original model file")

if __name__ == "__main__":
    main()