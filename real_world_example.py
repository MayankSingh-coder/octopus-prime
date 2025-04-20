import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from perceptron import Perceptron

def load_iris_binary():
    """
    Load the Iris dataset and convert it to a binary classification problem
    (setosa vs. non-setosa)
    
    Returns:
    --------
    X : array, shape = [n_samples, n_features]
        The input features
    y : array, shape = [n_samples]
        The binary labels (1 for setosa, -1 for non-setosa)
    """
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]  # Use sepal length and petal length
    y = iris.target
    
    # Convert to binary classification: Setosa (0) vs. non-Setosa (1, 2)
    y = np.where(y == 0, 1, -1)
    
    return X, y

def main():
    print("=== Real-world Example: Iris Dataset (Binary Classification) ===")
    
    # Load the Iris dataset (binary classification)
    X, y = load_iris_binary()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of class 1 (Setosa): {np.sum(y == 1)}")
    print(f"Number of class -1 (non-Setosa): {np.sum(y == -1)}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    
    # Standardize the features
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # Train the perceptron
    print("\nTraining the perceptron...")
    ppn = Perceptron(learning_rate=0.01, n_iterations=100, random_state=1)
    ppn.fit(X_train_std, y_train)
    
    # Plot the decision boundary
    print("Plotting the decision boundary...")
    plt.figure(figsize=(10, 6))
    
    # Plot the training data points
    plt.scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], 
                color='red', marker='o', label='Setosa')
    plt.scatter(X_train_std[y_train==-1, 0], X_train_std[y_train==-1, 1], 
                color='blue', marker='x', label='Non-Setosa')
    
    # Plot the decision boundary
    # Create a mesh grid
    x_min, x_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
    y_min, y_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict for each point in the mesh grid
    Z = np.array([ppn.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.xlabel('Standardized Sepal Length')
    plt.ylabel('Standardized Petal Length')
    plt.legend(loc='upper left')
    plt.title('Perceptron Decision Boundary (Iris Dataset)')
    plt.savefig('iris_decision_boundary.png')
    
    # Plot the training errors
    plt_errors = ppn.plot_training_errors()
    plt_errors.savefig('iris_training_errors.png')
    
    # Evaluate the model
    print("\nEvaluating the model...")
    y_pred_train = ppn.predict(X_train_std)
    y_pred_test = ppn.predict(X_test_std)
    
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Count misclassifications
    train_misclassified = np.sum(y_pred_train != y_train)
    test_misclassified = np.sum(y_pred_test != y_test)
    
    print(f"Training misclassifications: {train_misclassified} out of {len(y_train)}")
    print(f"Test misclassifications: {test_misclassified} out of {len(y_test)}")
    
    plt.show()

if __name__ == "__main__":
    main()