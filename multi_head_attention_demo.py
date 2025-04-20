#!/usr/bin/env python3
"""
Demonstration of multi-head self-attention mechanism.
This script shows how multi-head self-attention works in transformer models.
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

def multi_head_attention(X, num_heads, mask=None):
    """
    Compute multi-head self-attention.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input tensor of shape (batch_size, seq_length, d_model)
    num_heads : int
        Number of attention heads
    mask : numpy.ndarray, optional
        Mask of shape (batch_size, seq_length, seq_length)
        
    Returns:
    --------
    tuple
        (output, attention_weights_per_head)
    """
    batch_size, seq_length, d_model = X.shape
    
    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    # Dimension per head
    d_k = d_model // num_heads
    
    # Initialize weights for each head
    W_q = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    W_k = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    W_v = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    
    # Output projection
    W_o = np.random.randn(num_heads * d_k, d_model)
    
    # Process each head
    head_outputs = []
    attention_weights_per_head = []
    
    for h in range(num_heads):
        # Project input to Q, K, V for this head
        Q_h = np.matmul(X, W_q[h])  # (batch_size, seq_length, d_k)
        K_h = np.matmul(X, W_k[h])  # (batch_size, seq_length, d_k)
        V_h = np.matmul(X, W_v[h])  # (batch_size, seq_length, d_k)
        
        # Compute attention
        head_output, attention_weights = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
        
        # Store outputs and weights
        head_outputs.append(head_output)
        attention_weights_per_head.append(attention_weights)
    
    # Concatenate all head outputs
    concat_output = np.concatenate(head_outputs, axis=-1)  # (batch_size, seq_length, num_heads * d_k)
    
    # Apply output projection
    output = np.matmul(concat_output, W_o)  # (batch_size, seq_length, d_model)
    
    return output, attention_weights_per_head

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    batch_size = 1
    seq_length = 5
    d_model = 64
    num_heads = 4
    
    # Create a simple sequence
    # In a real model, these would be word embeddings
    X = np.random.randn(batch_size, seq_length, d_model)
    
    # Add positional encoding
    pos_enc = create_positional_encoding(seq_length, d_model)
    X = X + pos_enc
    
    # Compute multi-head attention
    output, attention_weights_per_head = multi_head_attention(X, num_heads)
    
    # Print shapes
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention heads: {num_heads}")
    
    # Visualize attention weights for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))
    
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attention_weights_per_head[h][0], cmap='viridis')
        ax.set_title(f'Head {h+1}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        ax.set_xticks(range(seq_length))
        ax.set_yticks(range(seq_length))
        
        # Add values to the heatmap
        for i in range(seq_length):
            for j in range(seq_length):
                ax.text(j, i, f'{attention_weights_per_head[h][0, i, j]:.2f}',
                         ha='center', va='center', 
                         color='white' if attention_weights_per_head[h][0, i, j] < 0.5 else 'black',
                         fontsize=8)
    
    plt.tight_layout()
    plt.savefig('multi_head_attention_weights.png')
    print("Multi-head attention weights visualization saved to 'multi_head_attention_weights.png'")
    
    # Create a causal mask (for autoregressive models like language models)
    print("\nDemonstrating causal multi-head attention (for autoregressive models):")
    mask = np.triu(np.ones((seq_length, seq_length)), k=1)[np.newaxis, ...]
    
    # Compute causal multi-head attention
    causal_output, causal_attention_weights_per_head = multi_head_attention(X, num_heads, mask)
    
    # Visualize causal attention weights for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))
    
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(causal_attention_weights_per_head[h][0], cmap='viridis')
        ax.set_title(f'Causal Head {h+1}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        ax.set_xticks(range(seq_length))
        ax.set_yticks(range(seq_length))
        
        # Add values to the heatmap
        for i in range(seq_length):
            for j in range(seq_length):
                ax.text(j, i, f'{causal_attention_weights_per_head[h][0, i, j]:.2f}',
                         ha='center', va='center', 
                         color='white' if causal_attention_weights_per_head[h][0, i, j] < 0.5 else 'black',
                         fontsize=8)
    
    plt.tight_layout()
    plt.savefig('causal_multi_head_attention_weights.png')
    print("Causal multi-head attention weights visualization saved to 'causal_multi_head_attention_weights.png'")
    
    # Show attention pattern differences between heads
    print("\nAttention Pattern Analysis:")
    print("Different heads can learn to attend to different patterns:")
    
    # Calculate average attention position for each head
    for h in range(num_heads):
        weights = attention_weights_per_head[h][0]
        
        # Calculate the average position each query attends to
        positions = np.arange(seq_length)
        avg_attended_positions = np.zeros(seq_length)
        
        for i in range(seq_length):
            avg_attended_positions[i] = np.sum(weights[i] * positions) / np.sum(weights[i])
        
        print(f"Head {h+1} average attended positions: {avg_attended_positions}")
    
    # Show mathematical formulation
    print("\nMathematical Formulation of Multi-Head Attention:")
    print("MultiHead(X) = Concat(head_1, ..., head_h) * W_o")
    print("where head_i = Attention(XW_q_i, XW_k_i, XW_v_i)")
    print("and Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V")
    
    print("\nAdvantages of Multi-Head Attention:")
    print("1. Different heads can attend to different parts of the sequence")
    print("2. Some heads may focus on local patterns, others on global relationships")
    print("3. Increases the model's representation power")
    print("4. Allows the model to jointly attend to information from different subspaces")

if __name__ == "__main__":
    main()