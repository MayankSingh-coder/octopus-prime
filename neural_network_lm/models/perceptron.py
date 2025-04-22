"""
Single Layer Perceptron Implementation

This module provides a basic implementation of a single layer perceptron
for binary classification tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..utils.visualization import plot_decision_boundary

class Perceptron:
    """
    A single layer perceptron implementation for binary classification.
    
    The perceptron is a simple neural network with a single layer that can
    learn to classify linearly separable data.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        """
        Initialize the perceptron.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for weight updates
        n_iterations : int
            Number of training iterations
        random_state : int
            Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors = []
        
    def fit(self, X, y):
        """
        Train the perceptron on the given data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features)
        y : numpy.ndarray
            Target values of shape (n_samples,)
            
        Returns:
        --------
        self : Perceptron
            The trained perceptron
        """
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        
        # Training loop
        self.errors = []
        
        for _ in range(self.n_iterations):
            errors = 0
            
            # Process each sample
            for xi, target in zip(X, y):
                # Calculate prediction
                update = self.learning_rate * (target - self.predict(xi))
                
                # Update weights and bias
                self.weights += update * xi
                self.bias += update
                
                # Count errors
                errors += int(update != 0.0)
                
            # Store error count for this iteration
            self.errors.append(errors)
            
            # If no errors, we've converged
            if errors == 0:
                break
                
        return self
    
    def net_input(self, X):
        """
        Calculate the net input.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Net input values
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def plot_training_errors(self):
        """
        Plot the number of misclassifications per epoch.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Number of Misclassifications')
        ax.set_title('Perceptron Training: Misclassifications per Epoch')
        ax.grid(True)
        return fig
    
    def visualize_decision_boundary(self, X, y):
        """
        Visualize the decision boundary of the perceptron.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, 2)
        y : numpy.ndarray
            Target values of shape (n_samples,)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        return plot_decision_boundary(X, y, self)