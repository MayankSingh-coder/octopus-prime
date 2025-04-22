#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    # Test importing the package
    import neural_network_lm
    print(f"✓ neural_network_lm package imported successfully (version: {neural_network_lm.__version__})")
    
    # Test importing models
    from neural_network_lm.models.perceptron import Perceptron
    from neural_network_lm.models.multi_layer_perceptron import MultiLayerPerceptron
    from neural_network_lm.models.attention_perceptron import AttentionPerceptron
    from neural_network_lm.models.self_attention import SelfAttention
    print("✓ All model classes imported successfully")
    
    # Test importing utils
    from neural_network_lm.utils.embeddings import WordEmbeddings
    from neural_network_lm.utils.visualization import plot_training_history, plot_attention_weights
    print("✓ All utility classes imported successfully")
    
    # Test importing tokenizers
    from neural_network_lm.tokenizers.custom_tokenizers import BPETokenizer, WordPieceTokenizer
    print("✓ All tokenizer classes imported successfully")
    
    # Test importing UI
    from neural_network_lm.ui.complete_mlp_ui import CompleteMlpUI
    print("✓ UI class imported successfully")
    
    print("\nAll imports successful! The package is correctly set up.")

if __name__ == "__main__":
    test_imports()