import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_linearly_separable_data(n_samples=100, n_features=2, random_state=1):
    """
    Generate linearly separable data for binary classification
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : array, shape = [n_samples, n_features]
        The generated samples
    y : array, shape = [n_samples]
        The binary labels (1 or -1) for class membership
    """
    X, y = make_blobs(n_samples=n_samples, 
                      n_features=n_features, 
                      centers=2, 
                      cluster_std=1.0,
                      random_state=random_state)
    
    # Convert the labels to be 1 or -1
    y = np.where(y == 0, -1, 1)
    
    return X, y

def visualize_data(X, y, title="Binary Classification Data"):
    """
    Visualize the data points for binary classification
    
    Parameters:
    -----------
    X : array-like, shape = [n_samples, 2]
        The input samples (must be 2D for visualization)
    y : array-like, shape = [n_samples]
        The binary labels (1 or -1)
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Class 1')
    plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x', label='Class -1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.title(title)
    
    return plt

def preprocess_data(X, y, test_size=0.3, random_state=1):
    """
    Preprocess the data: split into train/test sets and standardize
    
    Parameters:
    -----------
    X : array-like, shape = [n_samples, n_features]
        The input samples
    y : array-like, shape = [n_samples]
        The binary labels
    test_size : float
        The proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        The preprocessed and split data
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test