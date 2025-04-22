#!/usr/bin/env python3
"""
Test script to verify that model creation and training works correctly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model():
    """Test model creation and training."""
    print("Testing model creation and training...")
    
    # Import the model
    from neural_network_lm.models.multi_layer_perceptron import MultiLayerPerceptron
    
    # Create a simple model
    model = MultiLayerPerceptron(
        context_size=2,
        embedding_dim=20,
        hidden_layers=[32, 16],
        learning_rate=0.01,
        n_iterations=10,  # Small number for testing
        random_state=42
    )
    
    print("✓ Model created successfully")
    
    # Create a simple training text
    text = """
    The quick brown fox jumps over the lazy dog.
    A neural network learns from examples.
    Language models predict the next word in a sequence.
    """
    
    # Train the model
    print("Training model (this will take a few seconds)...")
    model.fit(text)
    
    print("✓ Model trained successfully")
    
    # Test prediction
    context = "neural network"
    next_word, info = model.predict_next_word(context)
    
    print(f"Context: '{context}'")
    print(f"Predicted next word: '{next_word}'")
    
    # Print top predictions
    print("\nTop predictions:")
    for word, prob in sorted(info["probabilities"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {word}: {prob:.4f}")
    
    print("\nAll tests passed! The model is working correctly.")

if __name__ == "__main__":
    test_model()