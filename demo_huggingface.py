#!/usr/bin/env python3
"""
Demo script to show Hugging Face integration with the fixed model.
"""

import sys
import os
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to demonstrate Hugging Face integration with the fixed model.
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
    
    # Create a model
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
    """
    
    # Train the model
    print("\n=== Training Model ===")
    print("Training model (this will take a few seconds)...")
    try:
        model.fit(text)
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Continuing with demo using untrained model...")
    
    # Save the model in Hugging Face format
    hf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "hf_model"
    )
    
    print(f"\n=== Saving Model in Hugging Face Format ===")
    print(f"Saving model to {hf_dir}...")
    
    try:
        model.save_for_huggingface(hf_dir)
        print(f"✓ Model saved successfully to {hf_dir}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    # Load the model from Hugging Face format
    print("\n=== Loading Model from Hugging Face Format ===")
    print(f"Loading model from {hf_dir}...")
    
    try:
        loaded_model = MultiLayerPerceptron.from_huggingface(hf_dir)
        print("✓ Model loaded successfully")
        
        # Test text generation with the loaded model
        context = "neural network"
        n_words = 5
        
        print(f"\nGenerating text with loaded model...")
        print(f"Context: '{context}'")
        
        generated_words, info = loaded_model.predict_next_n_words(context, n=n_words)
        
        print(f"Generated text: '{context} {' '.join(generated_words)}'")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    
    # Restore the original model file when done
    shutil.copy2(backup_file, original_file)
    print(f"\nRestored original model file")
    
    print("\n=== Hugging Face Integration Demo Complete ===")
    print("You can now use your model with Hugging Face libraries!")
    print("To upload to Hugging Face Hub, use the following commands:")
    print("\n```bash")
    print("# Install huggingface_hub")
    print("pip install huggingface_hub")
    print("\n# Login to Hugging Face")
    print("huggingface-cli login")
    print("\n# Upload your model")
    print("python -c \"from huggingface_hub import HfApi; api = HfApi(); api.create_repo('your-username/your-model-name', repo_type='model')\"")
    print("python -c \"from huggingface_hub import HfApi; api = HfApi(); api.upload_folder('hf_model', 'your-username/your-model-name')\"")
    print("```")

if __name__ == "__main__":
    main()