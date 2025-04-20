#!/usr/bin/env python3
"""
Single-Layer Perceptron implementation for language modeling.
This is a simplified version of the multi-layer perceptron that uses only a single hidden layer.
"""

import numpy as np
import re
import pickle
import os
import time
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from multi_layer_perceptron import MultiLayerPerceptron, WordEmbeddings

class SingleLayerPerceptron(MultiLayerPerceptron):
    """
    Single-Layer Perceptron for language modeling.
    This model inherits from MultiLayerPerceptron but enforces a single hidden layer.
    """
    
    def __init__(self, context_size=2, hidden_size=64, learning_rate=0.1, n_iterations=1000,
                 batch_size=32, validation_split=0.1, random_state=None, tokenizer_type='simple',
                 vocab_size=5000, use_pretrained=False, embedding_dim=50):
        """
        Initialize the Single-Layer Perceptron model.
        
        Parameters:
        -----------
        context_size : int
            Number of previous words to use as context
        hidden_size : int
            Size of the single hidden layer
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of training iterations
        batch_size : int
            Batch size for mini-batch gradient descent
        validation_split : float
            Fraction of data to use for validation
        random_state : int or None
            Random seed for reproducibility
        tokenizer_type : str
            Type of tokenizer to use ('simple' or 'wordpiece')
        vocab_size : int
            Maximum vocabulary size
        use_pretrained : bool
            Whether to use pretrained word embeddings
        embedding_dim : int
            Dimension of word embeddings
        """
        # Call the parent class initializer with a single hidden layer
        super().__init__(
            context_size=context_size,
            hidden_layers=[hidden_size],  # Single hidden layer
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            use_pretrained=use_pretrained,
            embedding_dim=embedding_dim
        )
        
        # Store the hidden size for easier access
        self.hidden_size = hidden_size
        
        # Override the model name
        self.model_name = "SingleLayerPerceptron"
    
    def _initialize_weights(self):
        """
        Initialize the weights for the single-layer perceptron.
        """
        # Call the parent method to initialize weights
        super()._initialize_weights()
        
        # Ensure we only have a single hidden layer
        if len(self.weights) > 2:
            # Keep only the input-to-hidden and hidden-to-output weights
            self.weights = self.weights[:2]
            self.biases = self.biases[:2]
    
    def _forward_pass(self, X):
        """
        Perform a forward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        tuple
            (output, hidden_activations)
        """
        # Input to hidden layer
        hidden_input = np.dot(X, self.weights[0]) + self.biases[0]
        hidden_activation = self._relu(hidden_input)
        
        # Hidden to output layer
        output_input = np.dot(hidden_activation, self.weights[1]) + self.biases[1]
        output = self._softmax(output_input)
        
        return output, [hidden_activation]
    
    def _backward_pass(self, X, y, output, hidden_activations):
        """
        Perform a backward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray
            Target data
        output : numpy.ndarray
            Output from forward pass
        hidden_activations : list
            List of hidden layer activations
            
        Returns:
        --------
        tuple
            (weight_gradients, bias_gradients)
        """
        # Get the single hidden layer activation
        hidden_activation = hidden_activations[0]
        
        # Output layer error
        output_error = output - y
        
        # Hidden layer error
        hidden_error = np.dot(output_error, self.weights[1].T) * self._relu_derivative(hidden_activation)
        
        # Weight gradients
        weight_gradients = [
            np.dot(X.T, hidden_error),
            np.dot(hidden_activation.T, output_error)
        ]
        
        # Bias gradients
        bias_gradients = [
            np.sum(hidden_error, axis=0),
            np.sum(output_error, axis=0)
        ]
        
        return weight_gradients, bias_gradients
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
        --------
        dict
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            'model_type': 'single_layer',
            'hidden_size': self.hidden_size
        })
        return info