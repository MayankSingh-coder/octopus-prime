"""
Visualization Utilities

This module provides functions for visualizing model performance and behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model, resolution=0.02):
    """
    Plot the decision boundary of a model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data of shape (n_samples, 2)
    y : numpy.ndarray
        Target values of shape (n_samples,)
    model : object
        Model with predict method
    resolution : float
        Resolution of the grid
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    # Check if X has 2 features
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for decision boundary visualization")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define the grid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    
    # Make predictions on the grid
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    
    # Plot the decision boundary
    ax.contourf(xx1, xx2, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    
    # Plot the data points
    for label in np.unique(y):
        ax.scatter(X[y == label, 0], X[y == label, 1], 
                  label=f'Class {label}', edgecolors='k')
    
    # Set labels and title
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')
    ax.legend()
    
    return fig

def plot_training_history(history):
    """
    Plot training history.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training and validation loss
    if 'loss' in history:
        ax.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Validation Loss')
    
    # Plot training and validation accuracy
    if 'accuracy' in history:
        ax2 = ax.twinx()
        ax2.plot(history['accuracy'], label='Training Accuracy', color='green')
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax2.set_ylabel('Accuracy')
    
    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend(loc='upper left')
    
    if 'accuracy' in history:
        ax2.legend(loc='upper right')
    
    return fig

def plot_attention_weights(attention_weights, tokens, ax=None):
    """
    Plot attention weights as a heatmap.
    
    Parameters:
    -----------
    attention_weights : numpy.ndarray
        Attention weights of shape (n_heads, seq_length, seq_length)
    tokens : list of str
        Tokens corresponding to the sequence
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # If attention_weights is 3D (multi-head), average over heads
    if len(attention_weights.shape) == 3:
        attention_weights = np.mean(attention_weights, axis=0)
    
    # Plot heatmap
    im = ax.imshow(attention_weights, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title
    ax.set_title("Attention Weights")
    
    # Add text annotations
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                   ha="center", va="center", color="white" if attention_weights[i, j] > 0.5 else "black")
    
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    
    return ax

def plot_word_embeddings_2d(embeddings, words, method='pca'):
    """
    Plot word embeddings in 2D.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Word embeddings of shape (n_words, embedding_dim)
    words : list of str
        Words corresponding to the embeddings
    method : str
        Dimensionality reduction method ('pca' or 'tsne')
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Reduce dimensionality
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot embeddings
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Add word labels
    for i, word in enumerate(words):
        ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)
    
    # Set title
    ax.set_title(f"Word Embeddings (2D {method.upper()})")
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig