#!/usr/bin/env python3
"""
Simple demonstration of self-attention mechanism.
This script shows how self-attention works without the complexity of a full model.
"""

import numpy as np
import matplotlib.pyplot as plt

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Parameters:
    -----------
    Q : numpy.ndarray
        Query matrix of shape (batch_size, seq_length, d_k)
    K : numpy.ndarray
        Key matrix of shape (batch_size, seq_length, d_k)
    V : numpy.ndarray
        Value matrix of shape (batch_size, seq_length, d_v)
    mask : numpy.ndarray, optional
        Mask of shape (batch_size, seq_length, seq_length)
        
    Returns:
    --------
    tuple
        (output, attention_weights)
    """
    # Compute attention scores
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores)
    
    # Apply attention weights to values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input array
        
    Returns:
    --------
    numpy.ndarray
        Softmax of input array
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def create_positional_encoding(max_length, d_model):
    """
    Create positional encoding for transformer models.
    
    Parameters:
    -----------
    max_length : int
        Maximum sequence length
    d_model : int
        Dimensionality of the model
        
    Returns:
    --------
    numpy.ndarray
        Positional encoding of shape (1, max_length, d_model)
    """
    # Initialize positional encoding
    pos_enc = np.zeros((max_length, d_model))
    
    # Calculate positional encoding
    for pos in range(max_length):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    # Add batch dimension
    pos_enc = pos_enc[np.newaxis, ...]
    
    return pos_enc

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    batch_size = 1
    seq_length = 5
    d_model = 8
    
    # Create a simple sequence
    # In a real model, these would be word embeddings
    X = np.random.randn(batch_size, seq_length, d_model)
    
    # Add positional encoding
    pos_enc = create_positional_encoding(seq_length, d_model)
    X = X + pos_enc
    
    # Create linear projections for Q, K, V
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    
    # Project input to Q, K, V
    Q = np.matmul(X, W_q)
    K = np.matmul(X, W_k)
    V = np.matmul(X, W_v)
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    # Print shapes
    print(f"Input shape: {X.shape}")
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0], cmap='viridis')
    plt.colorbar()
    plt.title('Self-Attention Weights')
    plt.xlabel('Key position')
    plt.ylabel('Query position')
    plt.xticks(range(seq_length))
    plt.yticks(range(seq_length))
    
    # Add values to the heatmap
    for i in range(seq_length):
        for j in range(seq_length):
            plt.text(j, i, f'{attention_weights[0, i, j]:.2f}',
                     ha='center', va='center', color='white' if attention_weights[0, i, j] < 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig('attention_weights.png')
    print("Attention weights visualization saved to 'attention_weights.png'")
    
    # Create a causal mask (for autoregressive models like language models)
    print("\nDemonstrating causal attention (for autoregressive models):")
    mask = np.triu(np.ones((seq_length, seq_length)), k=1)[np.newaxis, ...]
    
    # Compute causal attention
    causal_output, causal_attention_weights = scaled_dot_product_attention(Q, K, V, mask)
    
    # Visualize causal attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(causal_attention_weights[0], cmap='viridis')
    plt.colorbar()
    plt.title('Causal Self-Attention Weights')
    plt.xlabel('Key position')
    plt.ylabel('Query position')
    plt.xticks(range(seq_length))
    plt.yticks(range(seq_length))
    
    # Add values to the heatmap
    for i in range(seq_length):
        for j in range(seq_length):
            plt.text(j, i, f'{causal_attention_weights[0, i, j]:.2f}',
                     ha='center', va='center', color='white' if causal_attention_weights[0, i, j] < 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig('causal_attention_weights.png')
    print("Causal attention weights visualization saved to 'causal_attention_weights.png'")
    
    # Show mathematical formulation
    print("\nMathematical Formulation of Self-Attention:")
    print("Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V")
    print("Where:")
    print("- Q (query), K (key), and V (value) are linear projections of the input")
    print("- d_k is the dimensionality of the key vectors")
    print("- softmax normalizes the attention scores to sum to 1")
    
    print("\nIn multi-head attention, we have:")
    print("MultiHead(X) = Concat(head_1, ..., head_h) * W_o")
    print("where head_i = Attention(XW_q_i, XW_k_i, XW_v_i)")
    
    print("\nSelf-attention allows the model to weigh the importance of different")
    print("positions in a sequence when computing a representation for a position.")

if __name__ == "__main__":
    main()