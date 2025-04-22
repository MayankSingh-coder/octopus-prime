"""
Self-Attention Mechanism Implementation

This module provides an implementation of the self-attention mechanism
as described in "Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np

class SelfAttention:
    """
    Self-attention mechanism implementation for sequence modeling.
    
    This class implements the scaled dot-product attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017). It allows the model to weigh
    the importance of different words in a context when making predictions.
    
    Mathematical formulation:
    - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    - Where Q (query), K (key), and V (value) are derived from the input
    - d_k is the dimensionality of the key vectors
    """
    
    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0.1, random_state=42):
        """
        Initialize the self-attention mechanism.
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of the input vectors
        attention_dim : int, optional
            Dimensionality of the attention space. If None, uses input_dim
        num_heads : int
            Number of attention heads for multi-head attention
        dropout_rate : float
            Dropout rate for attention weights (0.0 to 1.0)
        random_state : int
            Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.attention_dim = attention_dim if attention_dim is not None else input_dim
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)
        
        # Initialize weights for query, key, value projections
        # Xavier/Glorot initialization for better gradient flow
        scale = np.sqrt(2.0 / (input_dim + self.attention_dim))
        
        # For multi-head attention, we create separate projections for each head
        self.W_query = self.rng.normal(0, scale, (num_heads, input_dim, self.head_dim))
        self.W_key = self.rng.normal(0, scale, (num_heads, input_dim, self.head_dim))
        self.W_value = self.rng.normal(0, scale, (num_heads, input_dim, self.head_dim))
        
        # Output projection to combine heads
        self.W_output = self.rng.normal(0, scale, (num_heads * self.head_dim, self.attention_dim))
        
        # Bias terms
        self.b_query = np.zeros((num_heads, self.head_dim))
        self.b_key = np.zeros((num_heads, self.head_dim))
        self.b_value = np.zeros((num_heads, self.head_dim))
        self.b_output = np.zeros(self.attention_dim)
        
        # For storing intermediate values during forward pass (used in backprop)
        self.cache = {}
        
    def forward(self, X, mask=None, training=True):
        """
        Forward pass through the self-attention mechanism.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input sequence of shape (batch_size, seq_length, input_dim)
        mask : numpy.ndarray, optional
            Attention mask of shape (batch_size, seq_length, seq_length)
            Used to mask out certain positions (e.g., future positions in causal attention)
        training : bool
            Whether the model is in training mode (affects dropout)
            
        Returns:
        --------
        numpy.ndarray
            Attention output of shape (batch_size, seq_length, attention_dim)
        """
        try:
            batch_size, seq_length, input_dim = X.shape
            
            # Handle dimension mismatch
            if input_dim != self.input_dim:
                # Resize input to match expected dimension
                if input_dim > self.input_dim:
                    X = X[:, :, :self.input_dim]
                else:
                    padding = np.zeros((batch_size, seq_length, self.input_dim - input_dim))
                    X = np.concatenate([X, padding], axis=2)
            
            # Initialize containers for multi-head attention
            all_head_outputs = []
            
            # Process each attention head
            for h in range(self.num_heads):
                # Project inputs to query, key, value with proper broadcasting
                Q = np.einsum('bsi,ih->bsh', X, self.W_query[h]) + self.b_query[h]
                K = np.einsum('bsi,ih->bsh', X, self.W_key[h]) + self.b_key[h]
                V = np.einsum('bsi,ih->bsh', X, self.W_value[h]) + self.b_value[h]
                
                # Compute attention scores with stable softmax
                scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
                
                # Apply mask if provided
                if mask is not None:
                    mask = np.broadcast_to(mask, scores.shape)
                    scores = np.where(mask, scores, -1e9)
                
                # Compute attention weights with numerical stability
                scores_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
                
                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    dropout_mask = self.rng.binomial(1, 1 - self.dropout_rate, attention_weights.shape)
                    attention_weights = attention_weights * dropout_mask
                    attention_weights = attention_weights / (np.sum(attention_weights, axis=-1, keepdims=True) + 1e-12)
                
                # Apply attention weights to values
                head_output = np.matmul(attention_weights, V)
                
                # Store for multi-head concatenation
                all_head_outputs.append(head_output)
                
                # Cache values for backpropagation
                self.cache[f'head_{h}'] = {
                    'Q': Q, 'K': K, 'V': V,
                    'scores': scores,
                    'attention_weights': attention_weights
                }
            
            # Concatenate and project all head outputs
            concat_output = np.concatenate(all_head_outputs, axis=2)
            output = np.dot(concat_output.reshape(batch_size * seq_length, -1), 
                           self.W_output).reshape(batch_size, seq_length, self.attention_dim) + self.b_output
            
            # Cache for backpropagation
            self.cache['concat_output'] = concat_output
            
            return output
            
        except Exception as e:
            raise ValueError(f"Error in attention forward pass: {str(e)}. Input shape: {X.shape}, Expected input_dim: {self.input_dim}")
    
    def backward(self, dY, learning_rate=0.01):
        """
        Backward pass through the self-attention mechanism.
        
        Parameters:
        -----------
        dY : numpy.ndarray
            Gradient of the loss with respect to the output of the attention layer
        learning_rate : float
            Learning rate for weight updates
            
        Returns:
        --------
        numpy.ndarray
            Gradient of the loss with respect to the input of the attention layer
        """
        # This is a simplified backward pass implementation
        # In a real implementation, we would compute gradients for all parameters
        # and update them using backpropagation
        
        # For now, we'll just return a placeholder gradient
        # In a real implementation, this would be the actual gradient
        return np.zeros_like(dY)