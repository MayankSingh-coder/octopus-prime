import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from data_utils import preprocess_data, visualize_data

def create_gender_dataset(n_samples=100, random_state=42):
    """
    Create a synthetic dataset for gender classification based on height and weight
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (will be split evenly between men and women)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : array, shape = [n_samples, 2]
        Features (height in cm, weight in kg)
    y : array, shape = [n_samples]
        Labels (1 for men, -1 for women)
    """
    np.random.seed(random_state)
    
    # Number of samples per class
    n_per_class = n_samples // 2
    
    # Generate data for men (class 1)
    # Average height for men: ~180cm with std dev of 5cm
    # Average weight for men: ~80kg with std dev of 8kg
    men_heights = np.random.normal(180, 5, n_per_class)
    men_weights = np.random.normal(80, 8, n_per_class)
    men_features = np.column_stack((men_heights, men_weights))
    men_labels = np.ones(n_per_class)
    
    # Generate data for women (class -1)
    # Average height for women: ~160cm with std dev of 5cm
    # Average weight for women: ~55kg with std dev of 6kg
    women_heights = np.random.normal(160, 5, n_per_class)
    women_weights = np.random.normal(55, 6, n_per_class)
    women_features = np.column_stack((women_heights, women_weights))
    women_labels = np.ones(n_per_class) * -1
    
    # Combine the data
    X = np.vstack((men_features, women_features))
    y = np.hstack((men_labels, women_labels))
    
    # Shuffle the data
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    return X, y

def main():
    # Create gender dataset
    print("Creating gender classification dataset...")
    X, y = create_gender_dataset(n_samples=200, random_state=42)
    
    # Visualize the data
    print("Visualizing the data...")
    plt_data = visualize_data(X, y, title="Gender Classification Data (Height vs Weight)")
    plt_data.xlabel('Height (cm)')
    plt_data.ylabel('Weight (kg)')
    plt_data.savefig('gender_data_visualization.png')
    
    # Preprocess the data
    print("Preprocessing the data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.3, random_state=42)
    
    # Create and train the perceptron
    print("Training the perceptron...")
    # Increase learning rate and iterations for better convergence
    ppn = Perceptron(learning_rate=0.1, n_iterations=10000, random_state=42)
    ppn.fit(X_train, y_train)
    
    # Plot the decision boundary
    print("Plotting the decision boundary...")
    plt_boundary = ppn.plot_decision_boundary(X_train, y_train)
    plt_boundary.xlabel('Height (standardized)')
    plt_boundary.ylabel('Weight (standardized)')
    plt_boundary.title('Perceptron Decision Boundary for Gender Classification')
    plt_boundary.savefig('gender_decision_boundary.png')
    
    # Plot the training errors
    print("Plotting the training errors...")
    plt_errors = ppn.plot_training_errors()
    plt_errors.savefig('gender_training_errors.png')
    
    # Evaluate the model
    print("Evaluating the model...")
    y_pred = ppn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Count misclassifications
    n_misclassified = np.sum(y_pred != y_test)
    print(f"Number of misclassified samples: {n_misclassified} out of {len(y_test)}")
    
    # Show some example predictions
    print("\nExample predictions:")
    # Convert standardized features back to original scale for better interpretability
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)  # Fit on the entire dataset to get the same scaling
    
    # Create a few test samples
    test_samples = np.array([
        [180, 80],  # Tall, heavy - likely male
        [160, 55],  # Short, light - likely female
        [170, 65],  # Medium height and weight - could be either
    ])
    
    print("Original features (Height cm, Weight kg) -> Predicted gender")
    for sample in test_samples:
        # Standardize the sample
        sample_std = sc.transform([sample])
        # Make prediction
        prediction = ppn.predict(sample_std)[0]
        gender = "Male" if prediction == 1 else "Female"
        print(f"{sample[0]:.1f}cm, {sample[1]:.1f}kg -> {gender}")
    
    plt.show()

if __name__ == "__main__":
    main()