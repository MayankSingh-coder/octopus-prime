import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from data_utils import generate_linearly_separable_data, visualize_data, preprocess_data

def main():
    # Generate linearly separable data
    print("Generating linearly separable data...")
    X, y = generate_linearly_separable_data(n_samples=100, random_state=42)
    
    # Visualize the data
    print("Visualizing the data...")
    plt_data = visualize_data(X, y, title="Generated Binary Classification Data")
    plt_data.savefig('data_visualization.png')
    
    # Preprocess the data
    print("Preprocessing the data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.3, random_state=42)
    
    # Create and train the perceptron
    print("Training the perceptron...")
    ppn = Perceptron(learning_rate=0.01, n_iterations=1000, random_state=42)
    ppn.fit(X_train, y_train)
    
    # Plot the decision boundary
    print("Plotting the decision boundary...")
    plt_boundary = ppn.plot_decision_boundary(X_train, y_train)
    plt_boundary.savefig('decision_boundary.png')
    
    # Plot the training errors
    print("Plotting the training errors...")
    plt_errors = ppn.plot_training_errors()
    plt_errors.savefig('training_errors.png')
    
    # Evaluate the model
    print("Evaluating the model...")
    y_pred = ppn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Count misclassifications
    n_misclassified = np.sum(y_pred != y_test)
    print(f"Number of misclassified samples: {n_misclassified} out of {len(y_test)}")
    
    plt.show()

if __name__ == "__main__":
    main()