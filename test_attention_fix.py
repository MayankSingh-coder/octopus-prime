#!/usr/bin/env python3
"""
Test script to verify the dimension mismatch fixes for the attention model.
This script loads a trained attention model and tests prediction with various contexts.
"""

import os
import sys
import numpy as np
import pickle
from attention_perceptron import AttentionPerceptron

def main():
    """
    Test loading an attention model and making predictions with various contexts.
    """
    # Check if model file is provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default model path
        model_path = "model_output/attention_model.pkl"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    print(f"Loading model from {model_path}...")
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new model with the same parameters
        model = AttentionPerceptron(
            context_size=model_data['context_size'],
            embedding_dim=model_data['embedding_dim'],
            hidden_layers=model_data['hidden_layers'],
            attention_dim=model_data['attention_dim'],
            num_attention_heads=model_data['num_attention_heads'],
            attention_dropout=model_data.get('attention_dropout', 0.1),
            learning_rate=model_data['learning_rate'],
            n_iterations=model_data['n_iterations'],
            random_state=model_data['random_state']
        )
        
        # Restore model attributes
        model.vocabulary = model_data['vocabulary']
        model.word_to_idx = model_data['word_to_idx']
        model.idx_to_word = model_data['idx_to_word']
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        model.input_size = model_data['input_size']
        model.output_size = model_data['output_size']
        
        # Initialize embeddings
        from embeddings import WordEmbeddings
        model.embeddings = WordEmbeddings(
            embedding_dim=model.embedding_dim,
            random_state=model.random_state
        )
        model.embeddings.vocabulary = model.vocabulary
        model.embeddings.word_to_idx = model.word_to_idx
        model.embeddings.idx_to_word = model.idx_to_word
        model.embeddings.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        # Initialize random embeddings for each word
        np.random.seed(model.random_state)
        model.embeddings.embeddings = {}
        for word, idx in model.word_to_idx.items():
            model.embeddings.embeddings[idx] = np.random.randn(model.embedding_dim)
        
        # Initialize special token embeddings
        for token, idx in model.embeddings.special_tokens.items():
            model.embeddings.embeddings[idx] = np.random.randn(model.embedding_dim)
        
        # Initialize encoder
        from sklearn.preprocessing import OneHotEncoder
        model.encoder = OneHotEncoder(sparse_output=False)
        model.encoder.fit(np.array(range(model.output_size)).reshape(-1, 1))
        
        # Restore attention layer
        from self_attention import SelfAttention
        if 'attention_W_query' in model_data and model_data['attention_W_query'] is not None:
            model.attention_layer = SelfAttention(
                input_dim=model.embedding_dim,
                attention_dim=model.attention_dim,
                num_heads=model.num_attention_heads,
                dropout_rate=model.attention_dropout,
                random_state=model.random_state
            )
            
            # Restore attention weights
            model.attention_layer.W_query = model_data['attention_W_query']
            model.attention_layer.W_key = model_data['attention_W_key']
            model.attention_layer.W_value = model_data['attention_W_value']
            model.attention_layer.W_output = model_data['attention_W_output']
            model.attention_layer.b_query = model_data['attention_b_query']
            model.attention_layer.b_key = model_data['attention_b_key']
            model.attention_layer.b_value = model_data['attention_b_value']
            model.attention_layer.b_output = model_data['attention_b_output']
        
        print("Model loaded successfully!")
        
        # Print model information
        print(f"Context size: {model.context_size}")
        print(f"Embedding dimension: {model.embedding_dim}")
        print(f"Attention dimension: {model.attention_dim}")
        print(f"Number of attention heads: {model.num_attention_heads}")
        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"Input size: {model.input_size}")
        print(f"Output size: {model.output_size}")
        
        # Test with a valid context
        if len(model.vocabulary) >= model.context_size:
            valid_context = model.vocabulary[:model.context_size]
            print(f"\nTesting with valid context: {valid_context}")
            try:
                next_word, info = model.predict_next_word(valid_context)
                print(f"Predicted next word: '{next_word}'")
                print(f"Attention weights shape: {np.array(info['attention_weights']).shape}")
            except Exception as e:
                print(f"Error with valid context: {e}")
        
        # Test with an unknown word in context
        print("\nTesting with unknown word in context...")
        try:
            unknown_context = ["unknown_word"] * model.context_size
            next_word, info = model.predict_next_word(unknown_context)
            print(f"Predicted next word: '{next_word}'")
            print(f"Attention weights shape: {np.array(info['attention_weights']).shape}")
        except Exception as e:
            print(f"Error with unknown context: {e}")
        
        # Test with a mixed context (known and unknown words)
        if len(model.vocabulary) >= 1:
            mixed_context = [model.vocabulary[0]] + ["unknown_word"] * (model.context_size - 1)
            print(f"\nTesting with mixed context: {mixed_context}")
            try:
                next_word, info = model.predict_next_word(mixed_context)
                print(f"Predicted next word: '{next_word}'")
                print(f"Attention weights shape: {np.array(info['attention_weights']).shape}")
            except Exception as e:
                print(f"Error with mixed context: {e}")
        
        # Test with a context of wrong length
        print("\nTesting with context of wrong length...")
        try:
            short_context = ["short"]
            next_word, info = model.predict_next_word(short_context)
            print(f"Predicted next word: '{next_word}'")
            print(f"Attention weights shape: {np.array(info['attention_weights']).shape}")
        except Exception as e:
            print(f"Error with short context: {e}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()