import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from perceptron import Perceptron
from sklearn.preprocessing import StandardScaler

class InteractivePerceptron:
    def __init__(self):
        # Create a figure for the interactive visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.95, wspace=0.3, hspace=0.3)
        
        # Create a perceptron and train it on gender data
        self.create_and_train_perceptron()
        
        # Set up the main visualization areas
        self.setup_visualization()
        
        # Create sliders for height and weight
        self.setup_sliders()
        
        # Create a button to make predictions
        self.setup_buttons()
        
        # Initial update of the visualization
        self.update_visualization(None)
        
    def create_and_train_perceptron(self):
        """Create and train the perceptron on gender classification data"""
        # Generate synthetic data for men and women
        np.random.seed(42)
        
        # Number of samples per class
        n_per_class = 100
        
        # Generate data for men (class 1)
        men_heights = np.random.normal(180, 5, n_per_class)
        men_weights = np.random.normal(80, 8, n_per_class)
        men_features = np.column_stack((men_heights, men_weights))
        men_labels = np.ones(n_per_class)
        
        # Generate data for women (class -1)
        women_heights = np.random.normal(160, 5, n_per_class)
        women_weights = np.random.normal(55, 6, n_per_class)
        women_features = np.column_stack((women_heights, women_weights))
        women_labels = np.ones(n_per_class) * -1
        
        # Combine the data
        self.X = np.vstack((men_features, women_features))
        self.y = np.hstack((men_labels, women_labels))
        
        # Standardize the features
        self.scaler = StandardScaler()
        self.X_std = self.scaler.fit_transform(self.X)
        print(f"Standardized features:\n{self.X_std[:5]}")
        # Train the perceptron
        self.ppn = Perceptron(learning_rate=0.1, n_iterations=10000, random_state=42)
        self.ppn.fit(self.X_std, self.y)
        
        # Store the original data for visualization
        self.men_features = men_features
        self.women_features = women_features
        
    def setup_visualization(self):
        """Set up the visualization areas"""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Set up the data visualization (top left)
        self.ax_data = self.axes[0, 0]
        self.ax_data.set_title('Training Data and Decision Boundary')
        self.ax_data.set_xlabel('Height (cm)')
        self.ax_data.set_ylabel('Weight (kg)')
        
        # Plot the training data
        self.ax_data.scatter(self.men_features[:, 0], self.men_features[:, 1], 
                            color='blue', marker='o', label='Men')
        self.ax_data.scatter(self.women_features[:, 0], self.women_features[:, 1], 
                            color='red', marker='x', label='Women')
        
        # Plot the decision boundary
        self.plot_decision_boundary()
        
        # Set up the computation flow visualization (top right)
        self.ax_flow = self.axes[0, 1]
        self.ax_flow.set_title('Perceptron Computation Flow')
        self.ax_flow.axis('off')
        
        # Set up the weight visualization (bottom left)
        self.ax_weights = self.axes[1, 0]
        self.ax_weights.set_title('Perceptron Weights')
        self.ax_weights.set_xlabel('Feature')
        self.ax_weights.set_ylabel('Weight Value')
        
        # Plot the weights
        features = ['Height', 'Weight', 'Bias']
        weights = np.append(self.ppn.weights, self.ppn.bias)
        self.ax_weights.bar(features, weights)
        
        # Set up the prediction result visualization (bottom right)
        self.ax_result = self.axes[1, 1]
        self.ax_result.set_title('Prediction Result')
        self.ax_result.axis('off')
        
    def plot_decision_boundary(self):
        """Plot the decision boundary on the data visualization"""
        # Get the min and max values for the original features
        x_min, x_max = self.X[:, 0].min() - 10, self.X[:, 0].max() + 10
        y_min, y_max = self.X[:, 1].min() - 10, self.X[:, 1].max() + 10
        
        # Create a mesh grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                            np.arange(y_min, y_max, 1))
        
        # Flatten the mesh grid and create feature pairs
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Standardize the mesh points
        mesh_points_std = self.scaler.transform(mesh_points)
        
        # Predict for each point in the mesh grid
        Z = np.array([self.ppn.predict(x) for x in mesh_points_std])
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        self.ax_data.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        self.ax_data.legend()
        
    def setup_sliders(self):
        """Set up sliders for height and weight"""
        # Create axes for the sliders
        ax_height = plt.axes([0.1, 0.15, 0.8, 0.03])
        ax_weight = plt.axes([0.1, 0.1, 0.8, 0.03])
        
        # Create the sliders
        self.height_slider = Slider(
            ax=ax_height,
            label='Height (cm)',
            valmin=140,
            valmax=200,
            valinit=170,
            valstep=1
        )
        
        self.weight_slider = Slider(
            ax=ax_weight,
            label='Weight (kg)',
            valmin=40,
            valmax=100,
            valinit=65,
            valstep=1
        )
        
        # Register the update function with each slider
        self.height_slider.on_changed(self.update_visualization)
        self.weight_slider.on_changed(self.update_visualization)
        
    def setup_buttons(self):
        """Set up buttons for interaction"""
        # Create axes for the button
        ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
        
        # Create the button
        self.predict_button = Button(
            ax=ax_button,
            label='Predict',
            color='lightgoldenrodyellow',
            hovercolor='0.975'
        )
        
        # Register the click function with the button
        self.predict_button.on_clicked(self.make_prediction)
        
    def update_visualization(self, val):
        """Update the visualization based on slider values"""
        # Get the current height and weight from the sliders
        height = self.height_slider.val
        weight = self.weight_slider.val
        
        # Update the data visualization
        # Clear the current point if it exists
        if hasattr(self, 'current_point'):
            self.current_point.remove()
        
        # Plot the current point
        self.current_point = self.ax_data.scatter(
            height, weight, color='green', marker='*', s=200, 
            label='Current Input', zorder=5
        )
        
        # Update the computation flow visualization
        self.ax_flow.clear()
        self.ax_flow.axis('off')
        
        # Add text explaining the computation flow
        self.ax_flow.text(0.5, 0.9, 'Perceptron Computation Flow', 
                         ha='center', va='center', fontsize=12, fontweight='bold')
        
        self.ax_flow.text(0.1, 0.8, f'Input Features:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_flow.text(0.15, 0.75, f'Height = {height} cm', 
                         ha='left', va='center', fontsize=10)
        self.ax_flow.text(0.15, 0.7, f'Weight = {weight} kg', 
                         ha='left', va='center', fontsize=10)
        
        # Standardize the input
        input_features = np.array([[height, weight]])
        input_std = self.scaler.transform(input_features)[0]
        
        self.ax_flow.text(0.1, 0.6, f'Standardized Features:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_flow.text(0.15, 0.55, f'Height_std = {input_std[0]:.4f}', 
                         ha='left', va='center', fontsize=10)
        self.ax_flow.text(0.15, 0.5, f'Weight_std = {input_std[1]:.4f}', 
                         ha='left', va='center', fontsize=10)
        
        # Calculate the net input
        net_input = self.ppn.net_input(input_std)
        
        self.ax_flow.text(0.1, 0.4, f'Net Input Calculation:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_flow.text(0.15, 0.35, f'w1 * Height_std + w2 * Weight_std + bias', 
                         ha='left', va='center', fontsize=10)
        self.ax_flow.text(0.15, 0.3, 
                         f'{self.ppn.weights[0]:.4f} * {input_std[0]:.4f} + ' +
                         f'{self.ppn.weights[1]:.4f} * {input_std[1]:.4f} + ' +
                         f'{self.ppn.bias:.4f}', 
                         ha='left', va='center', fontsize=10)
        self.ax_flow.text(0.15, 0.25, f'= {net_input:.4f}', 
                         ha='left', va='center', fontsize=10)
        
        # Make the prediction
        prediction = self.ppn.predict(input_std)
        
        self.ax_flow.text(0.1, 0.15, f'Activation Function:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_flow.text(0.15, 0.1, f'If net_input >= 0, predict 1 (Male)', 
                         ha='left', va='center', fontsize=10)
        self.ax_flow.text(0.15, 0.05, f'If net_input < 0, predict -1 (Female)', 
                         ha='left', va='center', fontsize=10)
        
        # Update the result visualization
        self.ax_result.clear()
        self.ax_result.axis('off')
        
        # Display the prediction result
        gender = "Male" if prediction == 1 else "Female"
        color = "blue" if prediction == 1 else "red"
        
        self.ax_result.text(0.5, 0.6, f'Prediction: {gender}', 
                           ha='center', va='center', fontsize=16, 
                           fontweight='bold', color=color)
        
        confidence = abs(net_input)  # Simple measure of confidence
        self.ax_result.text(0.5, 0.4, f'Confidence: {confidence:.4f}', 
                           ha='center', va='center', fontsize=14)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        
    def make_prediction(self, event):
        """Make a prediction when the button is clicked"""
        # This just calls the update visualization function
        self.update_visualization(None)
        
    def show(self):
        """Show the interactive visualization"""
        plt.show()

if __name__ == "__main__":
    print("Starting Interactive Perceptron Visualization...")
    print("This will show how the perceptron processes input features to make predictions.")
    print("Use the sliders to adjust height and weight, and see how the prediction changes.")
    
    # Create and show the interactive visualization
    interactive_ppn = InteractivePerceptron()
    interactive_ppn.show()