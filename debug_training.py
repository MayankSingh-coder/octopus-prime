#!/usr/bin/env python3
"""
Debug script to identify and fix training issues.
"""

import sys
import os
import shutil
import traceback

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to debug training issues.
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
        "neural_network_lm", "models", "multi_layer_perceptron_final.py"
    )
    shutil.copy2(fixed_file, original_file)
    print(f"Replaced model with fixed version")
    
    # Import the model
    from neural_network_lm.models.multi_layer_perceptron import MultiLayerPerceptron
    
    # Create a model with verbose debugging
    model = MultiLayerPerceptron(
        context_size=2,
        embedding_dim=20,
        hidden_layers=[32, 16],
        learning_rate=0.01,
        n_iterations=10,  # Small number for testing
        random_state=42
    )
    
    # Create a simple training text
    text = """
    The quick brown fox jumps over the lazy dog.
    A neural network learns from examples.
    Language models predict the next word in a sequence.
    Machine learning algorithms improve with more data.
    Deep learning models can recognize patterns in text.
    Natural language processing helps computers understand human language.
    """
    
    # Debug training process
    print("\n=== Debugging Training Process ===")
    print("Text length:", len(text))
    print("Word count:", len(text.split()))
    
    try:
        # Preprocess the text
        words = model._preprocess_text(text)
        print("Preprocessed word count:", len(words))
        print("First 10 words:", words[:10])
        
        # Build vocabulary
        model._build_vocabulary(words)
        print("Vocabulary size:", len(model.vocabulary))
        print("First 10 vocabulary items:", model.vocabulary[:10])
        
        # Create training data
        X, y = model._create_training_data(words)
        print("Training data shapes:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        
        # Try training
        print("\nAttempting to train the model...")
        model.fit(text)
        print("✓ Training successful!")
        
        # Test prediction
        print("\n=== Testing Prediction ===")
        context = "neural network"
        print(f"Context: '{context}'")
        
        # Predict next word
        next_word, info = model.predict_next_word(context)
        print(f"Predicted next word: '{next_word}'")
        
        # Predict multiple words
        n_words = 5
        generated_words, gen_info = model.predict_next_n_words(context, n=n_words)
        print(f"Generated text: '{context} {' '.join(generated_words)}'")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Restore the original model file when done
    shutil.copy2(backup_file, original_file)
    print(f"\nRestored original model file")

if __name__ == "__main__":
    main()