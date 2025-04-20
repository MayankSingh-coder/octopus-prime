#!/usr/bin/env python3
"""
Test script to verify the fixes to the MultiLayerPerceptron class.
"""

import os
import numpy as np
from multi_layer_perceptron import MultiLayerPerceptron

def main():
    """
    Test the MultiLayerPerceptron class with a simple example.
    """
    # Sample text for training
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    A watched pot never boils.
    Actions speak louder than words.
    All that glitters is not gold.
    Better late than never.
    Birds of a feather flock together.
    Cleanliness is next to godliness.
    Don't count your chickens before they hatch.
    """
    
    print("Creating and training the model...")
    model = MultiLayerPerceptron(
        context_size=2,
        embedding_dim=50,
        hidden_layers=[32, 16],
        learning_rate=0.1,
        n_iterations=200,
        random_state=42
    )
    
    # Train the model
    model.fit(sample_text)
    
    # Test prediction
    print("\nTesting next word prediction:")
    test_contexts = [
        "the quick",
        "brown fox",
        "never boils",
        "actions speak"
    ]
    
    for context in test_contexts:
        next_word, info = model.predict_next_word(context)
        print(f"Context: '{context}' → Next word: '{next_word}'")
        
        # Show top 3 predictions
        top_predictions, _ = model.get_top_predictions(context, top_n=3)
        print("Top 3 predictions:")
        for word, prob in top_predictions:
            print(f"  '{word}': {prob:.4f}")
        print()
    
    # Test text generation
    print("\nTesting text generation:")
    for context in test_contexts[:2]:
        predicted_words, info = model.predict_next_n_words(context, n=5)
        print(f"Initial context: '{context}' → Generated: '{' '.join(predicted_words)}'")
        print(f"Full sequence: '{context} {' '.join(predicted_words)}'")
        print()
    
    # Save and load the model
    model_path = "test_model.pkl"
    print(f"\nSaving model to {model_path}...")
    model.save_model(model_path)
    
    print(f"Loading model from {model_path}...")
    loaded_model = MultiLayerPerceptron.load_model(model_path)
    
    # Test the loaded model
    print("\nTesting loaded model:")
    context = "the quick"
    next_word, _ = loaded_model.predict_next_word(context)
    print(f"Context: '{context}' → Next word: '{next_word}'")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Removed {model_path}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()