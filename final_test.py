#!/usr/bin/env python3
"""
Final test script to verify all fixes.
"""

import sys
import os
import shutil
import traceback

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to test all fixes.
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
        "neural_network_lm", "models", "multi_layer_perceptron_fixed_final2.py"
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
    
    # Test 1: Untrained model prediction
    print("\n=== Test 1: Untrained Model Prediction ===")
    print("Testing prediction with untrained model...")
    
    try:
        # Test with various contexts
        contexts = [
            "hii smartcoin",
            "loan application",
            "neural network",
            "this is a very long context that should be truncated"
        ]
        
        for context in contexts:
            print(f"\nContext: '{context}'")
            next_word, info = model.predict_next_word(context)
            print(f"Predicted next word: '{next_word}'")
            print(f"Fallback used: {info.get('fallback', False)}")
            
            # Generate multiple words
            n_words = 5
            generated_words, gen_info = model.predict_next_n_words(context, n=n_words)
            print(f"Generated text: '{context} {' '.join(generated_words)}'")
        
        print("\n✓ Untrained model prediction works correctly")
    except Exception as e:
        print(f"\n❌ Error in untrained prediction: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Test 2: Training
    print("\n=== Test 2: Model Training ===")
    print("Training model (this will take a few seconds)...")
    
    try:
        model.fit(text)
        print("✓ Training successful!")
        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"First 10 vocabulary items: {model.vocabulary[:10]}")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Test 3: Trained model prediction
    print("\n=== Test 3: Trained Model Prediction ===")
    print("Testing prediction with trained model...")
    
    try:
        # Test with various contexts
        contexts = [
            "neural network",
            "language models",
            "hii smartcoin",  # Unknown words
            "this is a very long context that should be truncated"
        ]
        
        for context in contexts:
            print(f"\nContext: '{context}'")
            next_word, info = model.predict_next_word(context)
            print(f"Predicted next word: '{next_word}'")
            print(f"Fallback used: {info.get('fallback', False)}")
            
            # Generate multiple words
            n_words = 5
            generated_words, gen_info = model.predict_next_n_words(context, n=n_words)
            print(f"Generated text: '{context} {' '.join(generated_words)}'")
        
        print("\n✓ Trained model prediction works correctly")
    except Exception as e:
        print(f"\n❌ Error in trained prediction: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Test 4: Hugging Face integration
    print("\n=== Test 4: Hugging Face Integration ===")
    print("Testing Hugging Face integration...")
    
    try:
        # Save model in Hugging Face format
        hf_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hf_model"
        )
        
        print(f"Saving model to {hf_dir}...")
        model.save_for_huggingface(hf_dir)
        
        # Load model from Hugging Face format
        print(f"Loading model from {hf_dir}...")
        loaded_model = MultiLayerPerceptron.from_huggingface(hf_dir)
        
        # Test prediction with loaded model
        context = "neural network"
        print(f"\nContext: '{context}'")
        next_word, info = loaded_model.predict_next_word(context)
        print(f"Predicted next word: '{next_word}'")
        
        # Generate multiple words
        n_words = 5
        generated_words, gen_info = loaded_model.predict_next_n_words(context, n=n_words)
        print(f"Generated text: '{context} {' '.join(generated_words)}'")
        
        print("\n✓ Hugging Face integration works correctly")
    except Exception as e:
        print(f"\n❌ Error in Hugging Face integration: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Restore the original model file when done
    shutil.copy2(backup_file, original_file)
    print(f"\nRestored original model file")
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    main()