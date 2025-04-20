#!/usr/bin/env python3
"""
Test script to verify the dimension mismatch fixes.
This script loads a trained model and tests prediction with various contexts.
"""

import os
import sys
import numpy as np
from simple_language_model import SimpleLanguageModel

def main():
    """
    Test loading a model and making predictions with various contexts.
    """
    # Check if model file is provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default model path
        model_path = "simple_language_model.pkl"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    print(f"Loading model from {model_path}...")
    try:
        model = SimpleLanguageModel.load_model(model_path)
        print("Model loaded successfully!")
        
        # Print model information
        print(f"Context size: {model.context_size}")
        print(f"Embedding dimension: {model.embedding_dim}")
        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"Input size: {model.input_size}")
        print(f"Output size: {model.output_size}")
        print(f"Weights shape: {model.weights.shape}")
        
        # Test with a valid context
        if len(model.vocabulary) >= model.context_size:
            valid_context = model.vocabulary[:model.context_size]
            print(f"\nTesting with valid context: {valid_context}")
            try:
                next_word = model.predict_next_word(valid_context)
                print(f"Predicted next word: '{next_word}'")
                
                # Get top 3 predictions
                top_predictions = model.get_top_predictions(valid_context, top_n=3)
                print("Top 3 predictions:")
                for word, prob in top_predictions:
                    print(f"  '{word}': {prob:.4f}")
            except Exception as e:
                print(f"Error with valid context: {e}")
        
        # Test with an unknown word in context
        print("\nTesting with unknown word in context...")
        try:
            unknown_context = ["unknown_word"] * model.context_size
            next_word = model.predict_next_word(unknown_context)
            print(f"Predicted next word: '{next_word}'")
        except Exception as e:
            print(f"Error with unknown context: {e}")
        
        # Test with a mixed context (known and unknown words)
        if len(model.vocabulary) >= 1:
            mixed_context = [model.vocabulary[0]] + ["unknown_word"] * (model.context_size - 1)
            print(f"\nTesting with mixed context: {mixed_context}")
            try:
                next_word = model.predict_next_word(mixed_context)
                print(f"Predicted next word: '{next_word}'")
            except Exception as e:
                print(f"Error with mixed context: {e}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()