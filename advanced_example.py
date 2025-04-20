import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from perceptron import Perceptron
from data_utils import visualize_data, preprocess_data

def generate_xor_data(n_samples=100, noise=0.1, random_state=1):
    """
    Generate XOR-like data (not linearly separable)
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Standard deviation of Gaussian noise
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : array, shape = [n_samples, 2]
        The generated samples
    y : array, shape = [n_samples]
        The binary labels (1 or -1) for class membership
    """
    np.random.seed(random_state)
    
    # Generate points in all four quadrants
    n_per_quadrant = n_samples // 4
    
    # First quadrant (x > 0, y > 0) - Class 1
    X1 = np.random.normal(loc=0.5, scale=noise, size=(n_per_quadrant, 2))
    y1 = np.ones(n_per_quadrant)
    
    # Second quadrant (x < 0, y > 0) - Class -1
    X2 = np.random.normal(loc=[-0.5, 0.5], scale=noise, size=(n_per_quadrant, 2))
    y2 = -np.ones(n_per_quadrant)
    
    # Third quadrant (x < 0, y < 0) - Class 1
    X3 = np.random.normal(loc=[-0.5, -0.5], scale=noise, size=(n_per_quadrant, 2))
    y3 = np.ones(n_per_quadrant)
    
    # Fourth quadrant (x > 0, y < 0) - Class -1
    X4 = np.random.normal(loc=[0.5, -0.5], scale=noise, size=(n_per_quadrant, 2))
    y4 = -np.ones(n_per_quadrant)
    
    # Combine all quadrants
    X = np.vstack((X1, X2, X3, X4))
    y = np.hstack((y1, y2, y3, y4))
    
    # Shuffle the data
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    return X, y

def main():
    # Part 1: Linearly separable data (should work well)
    print("\n=== Part 1: Linearly Separable Data ===")
    
    # Generate linearly separable data
    X_linear = np.array([
        [2.7810836, 2.550537003],
        [1.465489372, 2.362125076],
        [3.396561688, 4.400293529],
        [1.38807019, 1.850220317],
        [3.06407232, 3.005305973],
        [7.627531214, 2.759262235],
        [5.332441248, 2.088626775],
        [6.922596716, 1.77106367],
        [8.675418651, -0.242068655],
        [7.673756466, 3.508563011]
    ])
    y_linear = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    
    # Visualize the data
    plt_linear = visualize_data(X_linear, y_linear, title="Linearly Separable Data")
    plt_linear.savefig('linear_data.png')
    
    # Train the perceptron
    ppn_linear = Perceptron(learning_rate=0.1, n_iterations=10, random_state=1)
    ppn_linear.fit(X_linear, y_linear)
    
    # Plot the decision boundary
    plt_boundary_linear = ppn_linear.plot_decision_boundary(X_linear, y_linear)
    plt_boundary_linear.savefig('linear_boundary.png')
    
    # Plot the training errors
    plt_errors_linear = ppn_linear.plot_training_errors()
    plt_errors_linear.savefig('linear_errors.png')
    
    # Evaluate the model
    y_pred_linear = ppn_linear.predict(X_linear)
    accuracy_linear = np.mean(y_pred_linear == y_linear)
    print(f"Accuracy on linearly separable data: {accuracy_linear:.4f}")
    
    # Part 2: XOR data (should fail)
    print("\n=== Part 2: XOR Data (Not Linearly Separable) ===")
    
    # Generate XOR data
    X_xor, y_xor = generate_xor_data(n_samples=200, noise=0.1, random_state=1)
    
    # Visualize the data
    plt_xor = visualize_data(X_xor, y_xor, title="XOR Data (Not Linearly Separable)")
    plt_xor.savefig('xor_data.png')
    
    # Train the perceptron
    ppn_xor = Perceptron(learning_rate=0.1, n_iterations=100, random_state=1)
    ppn_xor.fit(X_xor, y_xor)
    
    # Plot the decision boundary
    plt_boundary_xor = ppn_xor.plot_decision_boundary(X_xor, y_xor)
    plt_boundary_xor.savefig('xor_boundary.png')
    
    # Plot the training errors
    plt_errors_xor = ppn_xor.plot_training_errors()
    plt_errors_xor.savefig('xor_errors.png')
    
    # Evaluate the model
    y_pred_xor = ppn_xor.predict(X_xor)
    accuracy_xor = np.mean(y_pred_xor == y_xor)
    print(f"Accuracy on XOR data: {accuracy_xor:.4f}")
    print("Note: The perceptron cannot learn XOR patterns because they are not linearly separable.")
    
    # Part 3: Moon-shaped data (should fail)
    print("\n=== Part 3: Moon-shaped Data (Not Linearly Separable) ===")
    
    # Generate moon-shaped data
    X_moons, y_moons_orig = make_moons(n_samples=200, noise=0.1, random_state=1)
    y_moons = np.where(y_moons_orig == 0, -1, 1)  # Convert to -1 and 1
    
    # Visualize the data
    plt_moons = visualize_data(X_moons, y_moons, title="Moon-shaped Data (Not Linearly Separable)")
    plt_moons.savefig('moons_data.png')
    
    # Train the perceptron
    ppn_moons = Perceptron(learning_rate=0.1, n_iterations=100, random_state=1)
    ppn_moons.fit(X_moons, y_moons)
    
    # Plot the decision boundary
    plt_boundary_moons = ppn_moons.plot_decision_boundary(X_moons, y_moons)
    plt_boundary_moons.savefig('moons_boundary.png')
    
    # Plot the training errors
    plt_errors_moons = ppn_moons.plot_training_errors()
    plt_errors_moons.savefig('moons_errors.png')
    
    # Evaluate the model
    y_pred_moons = ppn_moons.predict(X_moons)
    accuracy_moons = np.mean(y_pred_moons == y_moons)
    print(f"Accuracy on moon-shaped data: {accuracy_moons:.4f}")
    print("Note: The perceptron cannot learn moon-shaped patterns because they are not linearly separable.")
    
    plt.show()

if __name__ == "__main__":
    main()