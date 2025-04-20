import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Single Layer Perceptron for binary classification
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=1):
        """
        Initialize the perceptron
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (between 0.0 and 1.0)
        n_iterations : int
            Number of passes over the training dataset
        random_state : int
            Random number generator seed for random weight initialization
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors_ = []
        
    def fit(self, X, y):
        """
        Fit training data
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors
        y : array-like, shape = [n_samples]
            Target values (1 or -1)
            
        Returns:
        --------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = 0.0
        
        # Training loop
        for _ in range(self.n_iterations):
            errors = 0
            
            # Update weights based on each training sample
            for xi, target in zip(X, y):
                # print(f"xi: {xi}, target: {target}")
                # Calculate prediction error
                update = self.learning_rate * (target - self.predict(xi))
                
                # Update weights and bias
                self.weights += update * xi
                self.bias += update
                
                # Count misclassifications
                errors += int(update != 0.0)
                
            # Store error for each epoch
            self.errors_.append(errors)
            
            # If no errors, stop training
            if errors == 0:
                break
                
        return self
    
    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def plot_decision_boundary(self, X, y):
        """
        Plot the decision boundary
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, 2]
            Training vectors (must be 2D for visualization)
        y : array-like, shape = [n_samples]
            Target values
        """
        # Plot the data points
        plt.figure(figsize=(10, 6))
        plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Class 1')
        plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x', label='Class -1')
        
        # Plot the decision boundary
        # We need only two points to plot the line
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        
        # Calculate the decision boundary line
        # w1*x + w2*y + bias = 0 => y = (-w1*x - bias) / w2
        slope = -self.weights[0] / self.weights[1]
        intercept = -self.bias / self.weights[1]
        
        y_min = slope * x_min + intercept
        y_max = slope * x_max + intercept
        
        plt.plot([x_min, x_max], [y_min, y_max], 'k-')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='upper left')
        plt.title('Perceptron Decision Boundary')
        
        return plt
    
    def plot_training_errors(self):
        """
        Plot the number of misclassifications per epoch
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of misclassifications')
        plt.title('Perceptron Training Errors')
        
        return plt