#!/usr/bin/env python3
"""
Test script to verify model loading and prediction.
This script loads a trained model and tests prediction to ensure dimensions match correctly.
"""

import os
import sys
import numpy as np
from simple_language_model import SimpleLanguageModel

def main():
    """
    Test loading a model and making predictions.
    """
    # Check if model file is provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default model path
        model_path = "model_output/standard_model.pkl"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Available models:")
        for file in os.listdir("model_output"):
            if file.endswith(".pkl"):
                print(f"  - model_output/{file}")
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
        
        # Test prediction with a simple context
        if len(model.vocabulary) >= model.context_size:
            # Get some words from vocabulary for testing
            test_context = model.vocabulary[:model.context_size]
            print(f"\nTesting prediction with context: {test_context}")
            
            try:
                # Get context vector
                context_vector = model.embeddings.get_embeddings_for_context(test_context)
                print(f"Context vector shape: {context_vector.shape}")
                
                # Test forward pass
                print("Testing forward pass...")
                y_pred = model._forward(context_vector.reshape(1, -1))
                print(f"Prediction shape: {y_pred.shape}")
                
                # Get top prediction
                predicted_idx = np.argmax(y_pred[0])
                predicted_word = model.idx_to_word[predicted_idx]
                print(f"Predicted next word: '{predicted_word}'")
                
                # Get top 3 predictions
                top_indices = np.argsort(y_pred[0])[-3:][::-1]
                top_probs = y_pred[0][top_indices]
                top_words = [model.idx_to_word[idx] for idx in top_indices]
                
                print("\nTop 3 predictions:")
                for word, prob in zip(top_words, top_probs):
                    print(f"  '{word}': {prob:.4f}")
                
                print("\nModel loading and prediction test successful!")
            except Exception as e:
                print(f"Error during prediction: {e}")
        else:
            print("Vocabulary is too small to create test context.")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()