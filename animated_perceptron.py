import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Rectangle, Arrow, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from perceptron import Perceptron
from sklearn.preprocessing import StandardScaler
import time

class AnimatedPerceptron:
    def __init__(self):
        # Create a figure for the interactive visualization
        self.fig = plt.figure(figsize=(15, 10))
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95, wspace=0.3, hspace=0.3)
        
        # Create a grid for the layout
        gs = self.fig.add_gridspec(2, 3)
        
        # Create axes for different parts of the visualization
        self.ax_data = self.fig.add_subplot(gs[0, 0])  # Data and decision boundary
        self.ax_neuron = self.fig.add_subplot(gs[0, 1:])  # Neuron visualization
        self.ax_weights = self.fig.add_subplot(gs[1, 0])  # Weights visualization
        self.ax_result = self.fig.add_subplot(gs[1, 1])  # Result visualization
        self.ax_steps = self.fig.add_subplot(gs[1, 2])  # Step-by-step calculation
        
        # Create a perceptron and train it on gender data
        self.create_and_train_perceptron()
        
        # Set up the main visualization areas
        self.setup_visualization()
        
        # Create sliders for height and weight
        self.setup_sliders()
        
        # Create a button to make predictions
        self.setup_buttons()
        
        # Animation state
        self.animation_running = False
        self.anim = None
        
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
        
        # Train the perceptron
        self.ppn = Perceptron(learning_rate=0.1, n_iterations=10000, random_state=42)
        self.ppn.fit(self.X_std, self.y)
        
        # Store the original data for visualization
        self.men_features = men_features
        self.women_features = women_features
        
    def setup_visualization(self):
        """Set up the visualization areas"""
        # Set up the data visualization
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
        
        # Set up the neuron visualization
        self.ax_neuron.set_title('Perceptron Neuron Visualization')
        self.ax_neuron.axis('off')
        
        # Set up the weight visualization
        self.ax_weights.set_title('Perceptron Weights')
        self.ax_weights.set_xlabel('Feature')
        self.ax_weights.set_ylabel('Weight Value')
        
        # Plot the weights
        features = ['Height', 'Weight', 'Bias']
        weights = np.append(self.ppn.weights, self.ppn.bias)
        self.ax_weights.bar(features, weights, color=['skyblue', 'lightgreen', 'salmon'])
        
        # Set up the result visualization
        self.ax_result.set_title('Prediction Result')
        self.ax_result.axis('off')
        
        # Set up the steps visualization
        self.ax_steps.set_title('Calculation Steps')
        self.ax_steps.axis('off')
        
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
        ax_height = plt.axes([0.1, 0.12, 0.8, 0.03])
        ax_weight = plt.axes([0.1, 0.07, 0.8, 0.03])
        
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
        # Create axes for the buttons
        ax_predict = plt.axes([0.3, 0.02, 0.2, 0.04])
        ax_animate = plt.axes([0.55, 0.02, 0.2, 0.04])
        
        # Create the buttons
        self.predict_button = Button(
            ax=ax_predict,
            label='Predict',
            color='lightgoldenrodyellow',
            hovercolor='0.975'
        )
        
        self.animate_button = Button(
            ax=ax_animate,
            label='Animate Calculation',
            color='lightblue',
            hovercolor='0.8'
        )
        
        # Register the click functions with the buttons
        self.predict_button.on_clicked(self.make_prediction)
        self.animate_button.on_clicked(self.animate_calculation)
        
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
        
        # Standardize the input
        input_features = np.array([[height, weight]])
        input_std = self.scaler.transform(input_features)[0]
        
        # Calculate the net input
        net_input = self.ppn.net_input(input_std)
        
        # Make the prediction
        prediction = self.ppn.predict(input_std)
        
        # Update the steps visualization
        self.ax_steps.clear()
        self.ax_steps.axis('off')
        
        # Add text explaining the calculation steps
        self.ax_steps.text(0.5, 0.95, 'Calculation Steps', 
                         ha='center', va='center', fontsize=12, fontweight='bold')
        
        self.ax_steps.text(0.05, 0.85, f'1. Input Features:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_steps.text(0.1, 0.8, f'Height = {height} cm', 
                         ha='left', va='center', fontsize=10)
        self.ax_steps.text(0.1, 0.75, f'Weight = {weight} kg', 
                         ha='left', va='center', fontsize=10)
        
        self.ax_steps.text(0.05, 0.65, f'2. Standardize Features:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_steps.text(0.1, 0.6, f'Height_std = {input_std[0]:.4f}', 
                         ha='left', va='center', fontsize=10)
        self.ax_steps.text(0.1, 0.55, f'Weight_std = {input_std[1]:.4f}', 
                         ha='left', va='center', fontsize=10)
        
        self.ax_steps.text(0.05, 0.45, f'3. Calculate Net Input:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_steps.text(0.1, 0.4, f'w1 * Height_std + w2 * Weight_std + bias', 
                         ha='left', va='center', fontsize=10)
        self.ax_steps.text(0.1, 0.35, 
                         f'{self.ppn.weights[0]:.4f} * {input_std[0]:.4f} + ' +
                         f'{self.ppn.weights[1]:.4f} * {input_std[1]:.4f} + ' +
                         f'{self.ppn.bias:.4f}', 
                         ha='left', va='center', fontsize=10)
        self.ax_steps.text(0.1, 0.3, f'= {net_input:.4f}', 
                         ha='left', va='center', fontsize=10)
        
        self.ax_steps.text(0.05, 0.2, f'4. Apply Activation Function:', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        self.ax_steps.text(0.1, 0.15, f'If net_input >= 0, predict 1 (Male)', 
                         ha='left', va='center', fontsize=10)
        self.ax_steps.text(0.1, 0.1, f'If net_input < 0, predict -1 (Female)', 
                         ha='left', va='center', fontsize=10)
        self.ax_steps.text(0.1, 0.05, f'Prediction: {prediction} ({prediction == 1 and "Male" or "Female"})', 
                         ha='left', va='center', fontsize=10, fontweight='bold')
        
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
        
        # Update the neuron visualization (static version)
        self.draw_neuron(height, weight, input_std, net_input, prediction)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        
    def draw_neuron(self, height, weight, input_std, net_input, prediction):
        """Draw the neuron visualization"""
        self.ax_neuron.clear()
        self.ax_neuron.axis('off')
        self.ax_neuron.set_title('Perceptron Neuron Visualization')
        
        # Set up the coordinates
        neuron_x, neuron_y = 0.5, 0.5
        input1_x, input1_y = 0.1, 0.7
        input2_x, input2_y = 0.1, 0.3
        output_x, output_y = 0.9, 0.5
        
        # Draw the neuron
        neuron_circle = Circle((neuron_x, neuron_y), 0.15, fill=True, 
                              color='lightgray', alpha=0.8, zorder=2)
        self.ax_neuron.add_patch(neuron_circle)
        
        # Draw the input nodes
        input1_circle = Circle((input1_x, input1_y), 0.08, fill=True, 
                              color='skyblue', alpha=0.8, zorder=2)
        input2_circle = Circle((input2_x, input2_y), 0.08, fill=True, 
                              color='lightgreen', alpha=0.8, zorder=2)
        self.ax_neuron.add_patch(input1_circle)
        self.ax_neuron.add_patch(input2_circle)
        
        # Draw the output node
        output_color = 'blue' if prediction == 1 else 'red'
        output_circle = Circle((output_x, output_y), 0.08, fill=True, 
                              color=output_color, alpha=0.8, zorder=2)
        self.ax_neuron.add_patch(output_circle)
        
        # Draw the connections
        arrow1 = FancyArrowPatch((input1_x + 0.08, input1_y), (neuron_x - 0.15, neuron_y + 0.05),
                                connectionstyle="arc3,rad=0.2", 
                                arrowstyle="-|>", color='gray', lw=2, zorder=1)
        arrow2 = FancyArrowPatch((input2_x + 0.08, input2_y), (neuron_x - 0.15, neuron_y - 0.05),
                                connectionstyle="arc3,rad=-0.2", 
                                arrowstyle="-|>", color='gray', lw=2, zorder=1)
        arrow3 = FancyArrowPatch((neuron_x + 0.15, neuron_y), (output_x - 0.08, output_y),
                                connectionstyle="arc3,rad=0", 
                                arrowstyle="-|>", color='gray', lw=2, zorder=1)
        
        self.ax_neuron.add_patch(arrow1)
        self.ax_neuron.add_patch(arrow2)
        self.ax_neuron.add_patch(arrow3)
        
        # Add labels
        self.ax_neuron.text(input1_x, input1_y + 0.12, f'Height: {height} cm\n(std: {input_std[0]:.2f})', 
                           ha='center', va='center', fontsize=9)
        self.ax_neuron.text(input2_x, input2_y - 0.12, f'Weight: {weight} kg\n(std: {input_std[1]:.2f})', 
                           ha='center', va='center', fontsize=9)
        
        self.ax_neuron.text(neuron_x, neuron_y, f'Net Input:\n{net_input:.2f}', 
                           ha='center', va='center', fontsize=9)
        
        gender = "Male" if prediction == 1 else "Female"
        self.ax_neuron.text(output_x, output_y + 0.12, f'Prediction:\n{gender}', 
                           ha='center', va='center', fontsize=9)
        
        # Add weight labels on the connections
        self.ax_neuron.text(neuron_x - 0.25, neuron_y + 0.1, f'w1: {self.ppn.weights[0]:.2f}', 
                           ha='center', va='center', fontsize=8, color='blue')
        self.ax_neuron.text(neuron_x - 0.25, neuron_y - 0.1, f'w2: {self.ppn.weights[1]:.2f}', 
                           ha='center', va='center', fontsize=8, color='green')
        
        # Add bias
        self.ax_neuron.text(neuron_x, neuron_y - 0.2, f'Bias: {self.ppn.bias:.2f}', 
                           ha='center', va='center', fontsize=8, color='red')
        
    def animate_calculation(self, event):
        """Animate the calculation process"""
        if self.animation_running:
            # Stop the animation if it's already running
            if self.anim is not None:
                self.anim.event_source.stop()
                self.anim = None
            self.animation_running = False
            return
        
        self.animation_running = True
        
        # Get the current height and weight from the sliders
        height = self.height_slider.val
        weight = self.weight_slider.val
        
        # Standardize the input
        input_features = np.array([[height, weight]])
        input_std = self.scaler.transform(input_features)[0]
        
        # Calculate the net input
        net_input = self.ppn.net_input(input_std)
        
        # Make the prediction
        prediction = self.ppn.predict(input_std)
        
        # Animation frames
        frames = 60  # Total number of frames
        
        # Animation function
        def animate(i):
            self.ax_neuron.clear()
            self.ax_neuron.axis('off')
            self.ax_neuron.set_title('Perceptron Neuron Visualization')
            
            # Set up the coordinates
            neuron_x, neuron_y = 0.5, 0.5
            input1_x, input1_y = 0.1, 0.7
            input2_x, input2_y = 0.1, 0.3
            output_x, output_y = 0.9, 0.5
            
            # Animation phases
            if i < 15:  # Phase 1: Show inputs
                # Draw the neuron
                neuron_circle = Circle((neuron_x, neuron_y), 0.15, fill=True, 
                                      color='lightgray', alpha=0.8, zorder=2)
                self.ax_neuron.add_patch(neuron_circle)
                
                # Draw the input nodes with animation
                alpha = min(1.0, i / 10)
                input1_circle = Circle((input1_x, input1_y), 0.08, fill=True, 
                                      color='skyblue', alpha=alpha, zorder=2)
                input2_circle = Circle((input2_x, input2_y), 0.08, fill=True, 
                                      color='lightgreen', alpha=alpha, zorder=2)
                self.ax_neuron.add_patch(input1_circle)
                self.ax_neuron.add_patch(input2_circle)
                
                # Add labels with animation
                if i > 5:
                    self.ax_neuron.text(input1_x, input1_y + 0.12, f'Height: {height} cm\n(std: {input_std[0]:.2f})', 
                                       ha='center', va='center', fontsize=9, alpha=alpha)
                    self.ax_neuron.text(input2_x, input2_y - 0.12, f'Weight: {weight} kg\n(std: {input_std[1]:.2f})', 
                                       ha='center', va='center', fontsize=9, alpha=alpha)
            
            elif i < 30:  # Phase 2: Show connections and weights
                # Draw all nodes
                neuron_circle = Circle((neuron_x, neuron_y), 0.15, fill=True, 
                                      color='lightgray', alpha=0.8, zorder=2)
                input1_circle = Circle((input1_x, input1_y), 0.08, fill=True, 
                                      color='skyblue', alpha=0.8, zorder=2)
                input2_circle = Circle((input2_x, input2_y), 0.08, fill=True, 
                                      color='lightgreen', alpha=0.8, zorder=2)
                self.ax_neuron.add_patch(neuron_circle)
                self.ax_neuron.add_patch(input1_circle)
                self.ax_neuron.add_patch(input2_circle)
                
                # Add input labels
                self.ax_neuron.text(input1_x, input1_y + 0.12, f'Height: {height} cm\n(std: {input_std[0]:.2f})', 
                                   ha='center', va='center', fontsize=9)
                self.ax_neuron.text(input2_x, input2_y - 0.12, f'Weight: {weight} kg\n(std: {input_std[1]:.2f})', 
                                   ha='center', va='center', fontsize=9)
                
                # Animate connections
                progress = (i - 15) / 15
                
                # First connection
                if progress > 0:
                    end_x1 = neuron_x - 0.15
                    end_y1 = neuron_y + 0.05
                    start_x1 = input1_x + 0.08
                    start_y1 = input1_y
                    
                    current_x1 = start_x1 + progress * (end_x1 - start_x1)
                    current_y1 = start_y1 + progress * (end_y1 - start_y1) + 0.1 * np.sin(progress * np.pi)
                    
                    arrow1 = FancyArrowPatch((start_x1, start_y1), (current_x1, current_y1),
                                            connectionstyle="arc3,rad=0.2", 
                                            arrowstyle="-|>", color='blue', lw=2, zorder=1)
                    self.ax_neuron.add_patch(arrow1)
                    
                    # Add weight label
                    if progress > 0.5:
                        weight_alpha = min(1.0, (progress - 0.5) * 2)
                        self.ax_neuron.text(neuron_x - 0.25, neuron_y + 0.1, f'w1: {self.ppn.weights[0]:.2f}', 
                                           ha='center', va='center', fontsize=8, color='blue', alpha=weight_alpha)
                
                # Second connection
                if progress > 0.3:
                    adjusted_progress = (progress - 0.3) / 0.7
                    end_x2 = neuron_x - 0.15
                    end_y2 = neuron_y - 0.05
                    start_x2 = input2_x + 0.08
                    start_y2 = input2_y
                    
                    current_x2 = start_x2 + adjusted_progress * (end_x2 - start_x2)
                    current_y2 = start_y2 + adjusted_progress * (end_y2 - start_y2) - 0.1 * np.sin(adjusted_progress * np.pi)
                    
                    arrow2 = FancyArrowPatch((start_x2, start_y2), (current_x2, current_y2),
                                            connectionstyle="arc3,rad=-0.2", 
                                            arrowstyle="-|>", color='green', lw=2, zorder=1)
                    self.ax_neuron.add_patch(arrow2)
                    
                    # Add weight label
                    if adjusted_progress > 0.5:
                        weight_alpha = min(1.0, (adjusted_progress - 0.5) * 2)
                        self.ax_neuron.text(neuron_x - 0.25, neuron_y - 0.1, f'w2: {self.ppn.weights[1]:.2f}', 
                                           ha='center', va='center', fontsize=8, color='green', alpha=weight_alpha)
                
                # Add bias with animation
                if progress > 0.7:
                    bias_alpha = min(1.0, (progress - 0.7) * 3)
                    self.ax_neuron.text(neuron_x, neuron_y - 0.2, f'Bias: {self.ppn.bias:.2f}', 
                                       ha='center', va='center', fontsize=8, color='red', alpha=bias_alpha)
            
            elif i < 45:  # Phase 3: Show calculation in neuron
                # Draw all nodes
                input1_circle = Circle((input1_x, input1_y), 0.08, fill=True, 
                                      color='skyblue', alpha=0.8, zorder=2)
                input2_circle = Circle((input2_x, input2_y), 0.08, fill=True, 
                                      color='lightgreen', alpha=0.8, zorder=2)
                self.ax_neuron.add_patch(input1_circle)
                self.ax_neuron.add_patch(input2_circle)
                
                # Neuron "processing" animation
                progress = (i - 30) / 15
                pulse_size = 0.15 + 0.02 * np.sin(progress * 10 * np.pi)
                pulse_color = (1 - progress, 0.5, progress)  # Color transition
                
                neuron_circle = Circle((neuron_x, neuron_y), pulse_size, fill=True, 
                                      color=pulse_color, alpha=0.8, zorder=2)
                self.ax_neuron.add_patch(neuron_circle)
                
                # Draw the connections
                arrow1 = FancyArrowPatch((input1_x + 0.08, input1_y), (neuron_x - 0.15, neuron_y + 0.05),
                                        connectionstyle="arc3,rad=0.2", 
                                        arrowstyle="-|>", color='blue', lw=2, zorder=1)
                arrow2 = FancyArrowPatch((input2_x + 0.08, input2_y), (neuron_x - 0.15, neuron_y - 0.05),
                                        connectionstyle="arc3,rad=-0.2", 
                                        arrowstyle="-|>", color='green', lw=2, zorder=1)
                self.ax_neuron.add_patch(arrow1)
                self.ax_neuron.add_patch(arrow2)
                
                # Add all labels
                self.ax_neuron.text(input1_x, input1_y + 0.12, f'Height: {height} cm\n(std: {input_std[0]:.2f})', 
                                   ha='center', va='center', fontsize=9)
                self.ax_neuron.text(input2_x, input2_y - 0.12, f'Weight: {weight} kg\n(std: {input_std[1]:.2f})', 
                                   ha='center', va='center', fontsize=9)
                
                self.ax_neuron.text(neuron_x - 0.25, neuron_y + 0.1, f'w1: {self.ppn.weights[0]:.2f}', 
                                   ha='center', va='center', fontsize=8, color='blue')
                self.ax_neuron.text(neuron_x - 0.25, neuron_y - 0.1, f'w2: {self.ppn.weights[1]:.2f}', 
                                   ha='center', va='center', fontsize=8, color='green')
                self.ax_neuron.text(neuron_x, neuron_y - 0.2, f'Bias: {self.ppn.bias:.2f}', 
                                   ha='center', va='center', fontsize=8, color='red')
                
                # Show calculation in neuron
                calc_progress = min(1.0, progress * 2)
                if calc_progress > 0:
                    calc_text = f'Calculating...\n'
                    if calc_progress > 0.3:
                        calc_text += f'{self.ppn.weights[0]:.2f} * {input_std[0]:.2f} +\n'
                    if calc_progress > 0.6:
                        calc_text += f'{self.ppn.weights[1]:.2f} * {input_std[1]:.2f} +\n'
                    if calc_progress > 0.9:
                        calc_text += f'{self.ppn.bias:.2f} =\n{net_input:.2f}'
                    
                    self.ax_neuron.text(neuron_x, neuron_y, calc_text, 
                                       ha='center', va='center', fontsize=8)
            
            else:  # Phase 4: Show output and prediction
                # Draw all nodes
                neuron_circle = Circle((neuron_x, neuron_y), 0.15, fill=True, 
                                      color='lightgray', alpha=0.8, zorder=2)
                input1_circle = Circle((input1_x, input1_y), 0.08, fill=True, 
                                      color='skyblue', alpha=0.8, zorder=2)
                input2_circle = Circle((input2_x, input2_y), 0.08, fill=True, 
                                      color='lightgreen', alpha=0.8, zorder=2)
                self.ax_neuron.add_patch(neuron_circle)
                self.ax_neuron.add_patch(input1_circle)
                self.ax_neuron.add_patch(input2_circle)
                
                # Draw the connections
                arrow1 = FancyArrowPatch((input1_x + 0.08, input1_y), (neuron_x - 0.15, neuron_y + 0.05),
                                        connectionstyle="arc3,rad=0.2", 
                                        arrowstyle="-|>", color='blue', lw=2, zorder=1)
                arrow2 = FancyArrowPatch((input2_x + 0.08, input2_y), (neuron_x - 0.15, neuron_y - 0.05),
                                        connectionstyle="arc3,rad=-0.2", 
                                        arrowstyle="-|>", color='green', lw=2, zorder=1)
                self.ax_neuron.add_patch(arrow1)
                self.ax_neuron.add_patch(arrow2)
                
                # Add all labels
                self.ax_neuron.text(input1_x, input1_y + 0.12, f'Height: {height} cm\n(std: {input_std[0]:.2f})', 
                                   ha='center', va='center', fontsize=9)
                self.ax_neuron.text(input2_x, input2_y - 0.12, f'Weight: {weight} kg\n(std: {input_std[1]:.2f})', 
                                   ha='center', va='center', fontsize=9)
                
                self.ax_neuron.text(neuron_x - 0.25, neuron_y + 0.1, f'w1: {self.ppn.weights[0]:.2f}', 
                                   ha='center', va='center', fontsize=8, color='blue')
                self.ax_neuron.text(neuron_x - 0.25, neuron_y - 0.1, f'w2: {self.ppn.weights[1]:.2f}', 
                                   ha='center', va='center', fontsize=8, color='green')
                self.ax_neuron.text(neuron_x, neuron_y - 0.2, f'Bias: {self.ppn.bias:.2f}', 
                                   ha='center', va='center', fontsize=8, color='red')
                
                # Show net input in neuron
                self.ax_neuron.text(neuron_x, neuron_y, f'Net Input:\n{net_input:.2f}', 
                                   ha='center', va='center', fontsize=9)
                
                # Animate output connection and node
                progress = (i - 45) / 15
                
                # Output connection animation
                end_x3 = output_x - 0.08
                end_y3 = output_y
                start_x3 = neuron_x + 0.15
                start_y3 = neuron_y
                
                current_x3 = start_x3 + progress * (end_x3 - start_x3)
                current_y3 = start_y3 + progress * (end_y3 - start_y3)
                
                arrow3 = FancyArrowPatch((start_x3, start_y3), (current_x3, current_y3),
                                        connectionstyle="arc3,rad=0", 
                                        arrowstyle="-|>", color='gray', lw=2, zorder=1)
                self.ax_neuron.add_patch(arrow3)
                
                # Output node animation
                if progress > 0.7:
                    output_alpha = min(1.0, (progress - 0.7) * 3)
                    output_color = 'blue' if prediction == 1 else 'red'
                    output_circle = Circle((output_x, output_y), 0.08, fill=True, 
                                          color=output_color, alpha=output_alpha, zorder=2)
                    self.ax_neuron.add_patch(output_circle)
                    
                    # Add output label
                    gender = "Male" if prediction == 1 else "Female"
                    self.ax_neuron.text(output_x, output_y + 0.12, f'Prediction:\n{gender}', 
                                       ha='center', va='center', fontsize=9, alpha=output_alpha)
                    
                    # Add activation function explanation
                    self.ax_neuron.text(neuron_x + 0.3, neuron_y + 0.15, 
                                       f'Activation Function:\nIf net_input ≥ 0 → Male\nIf net_input < 0 → Female', 
                                       ha='center', va='center', fontsize=8, alpha=output_alpha)
            
            return []
        
        # Create the animation
        self.anim = FuncAnimation(self.fig, animate, frames=frames, interval=50, blit=True)
        
        # Start the animation
        self.fig.canvas.draw_idle()
        
    def make_prediction(self, event):

        """Make a prediction when the button is clicked"""
        # This just calls the update visualization function
        self.update_visualization(None)
        
    def show(self):
        """Show the interactive visualization"""
        plt.show()

if __name__ == "__main__":
    print("Starting Animated Perceptron Visualization...")
    print("This will show how the perceptron processes input features to make predictions.")
    print("Use the sliders to adjust height and weight, and see how the prediction changes.")
    print("Click 'Animate Calculation' to see a step-by-step animation of the perceptron's computation.")
    
    # Create and show the interactive visualization
    animated_ppn = AnimatedPerceptron()
    animated_ppn.show()