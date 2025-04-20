#!/usr/bin/env python3
"""
Complete UI for the Multi-Layer Perceptron model with all features:
- Training with graph visualization
- Next word prediction with probabilities
- Sentence generation
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CompleteMlpUI:
    """
    A complete UI for the Multi-Layer Perceptron model with all features.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        """
        self.root = root
        self.root.title("Complete MLP UI")
        self.root.geometry("1200x800")
        
        # Create model
        self.model = None
        self.training_thread = None
        self.stop_training_event = threading.Event()
        self.progress_queue = queue.Queue()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.training_tab = ttk.Frame(self.notebook, padding=10)
        self.prediction_tab = ttk.Frame(self.notebook, padding=10)
        self.generation_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.training_tab, text="Train Model")
        self.notebook.add(self.prediction_tab, text="Predict Next Word")
        self.notebook.add(self.generation_tab, text="Generate Text")
        
        # Set up training tab
        self.setup_training_tab()
        
        # Set up prediction tab
        self.setup_prediction_tab()
        
        # Set up generation tab
        self.setup_generation_tab()
        
        # Set up status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start checking the progress queue
        self.check_progress_queue()
    
    def setup_training_tab(self):
        """
        Set up the training tab.
        """
        # Create frames
        left_frame = ttk.Frame(self.training_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(self.training_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model Information frame
        self.info_frame = ttk.LabelFrame(right_frame, text="Model Information")
        self.info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model Status
        status_frame = ttk.Frame(self.info_frame)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="Model Status:").pack(side=tk.LEFT)
        self.model_status_var = tk.StringVar(value="Not trained")
        ttk.Label(status_frame, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Vocabulary Size
        vocab_frame = ttk.Frame(self.info_frame)
        vocab_frame.pack(fill=tk.X, pady=5)
        ttk.Label(vocab_frame, text="Vocabulary Size:").pack(side=tk.LEFT)
        self.vocab_size_var = tk.StringVar(value="N/A")
        ttk.Label(vocab_frame, textvariable=self.vocab_size_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Architecture
        arch_frame = ttk.Frame(self.info_frame)
        arch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(arch_frame, text="Architecture:").pack(side=tk.LEFT)
        self.arch_var = tk.StringVar(value="N/A")
        ttk.Label(arch_frame, textvariable=self.arch_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Left frame - Training data and parameters
        ttk.Label(left_frame, text="Training Data", style="Header.TLabel").pack(pady=(0, 5), anchor=tk.W)
        
        # Text input frame
        text_input_frame = ttk.Frame(left_frame)
        text_input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Training text
        self.training_text = scrolledtext.ScrolledText(text_input_frame, wrap=tk.WORD, height=15)
        self.training_text.pack(fill=tk.BOTH, expand=True)
        
        # Text buttons
        text_buttons_frame = ttk.Frame(text_input_frame)
        text_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        load_file_button = ttk.Button(text_buttons_frame, text="Load from File", command=self.load_text_from_file)
        load_file_button.pack(side=tk.LEFT)
        
        clear_text_button = ttk.Button(text_buttons_frame, text="Clear Text", command=self.clear_training_text)
        clear_text_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(left_frame, text="Model Parameters")
        self.params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Context Size
        context_frame = ttk.Frame(self.params_frame)
        context_frame.pack(fill=tk.X, pady=5)
        ttk.Label(context_frame, text="Context Size:").pack(side=tk.LEFT)
        self.context_size_var = tk.IntVar(value=3)
        context_size_entry = ttk.Spinbox(context_frame, from_=1, to=5, textvariable=self.context_size_var, width=5)
        context_size_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(context_frame, text="(Number of previous words to use as context)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Hidden Layers
        hidden_frame = ttk.Frame(self.params_frame)
        hidden_frame.pack(fill=tk.X, pady=5)
        ttk.Label(hidden_frame, text="Hidden Layers:").pack(side=tk.LEFT)
        self.hidden_layers_var = tk.StringVar(value="64,32")
        hidden_layers_entry = ttk.Entry(hidden_frame, textvariable=self.hidden_layers_var, width=15)
        hidden_layers_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(hidden_frame, text="(Comma-separated list of layer sizes, e.g., 64,32)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Learning rate
        lr_frame = ttk.Frame(self.params_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        learning_rate_entry = ttk.Spinbox(lr_frame, from_=0.001, to=1.0, increment=0.01, textvariable=self.learning_rate_var, width=5)
        learning_rate_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(lr_frame, text="(Step size for weight updates)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Iterations
        iter_frame = ttk.Frame(self.params_frame)
        iter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iterations_var = tk.IntVar(value=1000)
        iterations_entry = ttk.Spinbox(iter_frame, from_=100, to=10000, increment=100, textvariable=self.iterations_var, width=7)
        iterations_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(iter_frame, text="(Number of training iterations)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Training buttons
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.train_button = ttk.Button(buttons_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(side=tk.LEFT)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Model operations
        model_ops_frame = ttk.Frame(left_frame)
        model_ops_frame.pack(fill=tk.X)
        
        self.save_model_button = ttk.Button(model_ops_frame, text="Save Model", command=self.save_model, state=tk.DISABLED)
        self.save_model_button.pack(side=tk.LEFT)
        
        self.load_model_button = ttk.Button(model_ops_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Right frame - Training progress
        ttk.Label(right_frame, text="Training Progress", style="Header.TLabel").pack(pady=(0, 5), anchor=tk.W)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Progress label
        self.progress_label = ttk.Label(right_frame, text="Not started")
        self.progress_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Training plot
        plot_frame = ttk.LabelFrame(right_frame, text="Training Loss")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure for the plot
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Training and Validation Loss')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True)
        
        # Create canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_prediction_tab(self):
        """
        Set up the prediction tab.
        """
        # Top frame - Input
        top_frame = ttk.Frame(self.prediction_tab)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(top_frame, text="Enter Context:", style="Bold.TLabel").pack(anchor=tk.W)
        
        # Context entry
        self.context_entry = ttk.Entry(top_frame, width=50)
        self.context_entry.pack(side=tk.LEFT, pady=(5, 0))
        
        # Predict button
        self.predict_button = ttk.Button(top_frame, text="Predict Next Word", command=self.predict_next_word, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Number of predictions
        ttk.Label(top_frame, text="Top N:").pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        self.top_n_var = tk.IntVar(value=5)
        top_n_entry = ttk.Spinbox(top_frame, from_=1, to=20, textvariable=self.top_n_var, width=3)
        top_n_entry.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Middle frame - Results
        middle_frame = ttk.LabelFrame(self.prediction_tab, text="Prediction Results")
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Results table
        columns = ("word", "probability")
        self.results_tree = ttk.Treeview(middle_frame, columns=columns, show="headings")
        self.results_tree.heading("word", text="Word")
        self.results_tree.heading("probability", text="Probability")
        self.results_tree.column("word", width=150)
        self.results_tree.column("probability", width=150)
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame - Instructions
        bottom_frame = ttk.Frame(self.prediction_tab)
        bottom_frame.pack(fill=tk.X)
        
        ttk.Label(bottom_frame, text="Instructions:", style="Bold.TLabel").pack(anchor=tk.W)
        instructions = (
            "1. Enter any text as context - the model will automatically adjust if needed.\n"
            "2. Click 'Predict Next Word' to see the most likely next words according to the model.\n"
            "3. The table shows the top N predicted words and their probabilities.\n"
            "4. If your context is adjusted (too short, too long, or contains unknown words), you'll see details in a popup."
        )
        ttk.Label(bottom_frame, text=instructions, wraplength=600).pack(anchor=tk.W, pady=(5, 0))
    
    def setup_generation_tab(self):
        """
        Set up the generation tab.
        """
        # Top frame - Input
        top_frame = ttk.Frame(self.generation_tab)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(top_frame, text="Starting Context:", style="Bold.TLabel").pack(anchor=tk.W)
        
        # Context entry
        self.gen_context_entry = ttk.Entry(top_frame, width=50)
        self.gen_context_entry.pack(side=tk.LEFT, pady=(5, 0))
        
        # Number of words
        ttk.Label(top_frame, text="Number of Words:").pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        self.num_words_var = tk.IntVar(value=20)
        num_words_entry = ttk.Spinbox(top_frame, from_=1, to=100, textvariable=self.num_words_var, width=3)
        num_words_entry.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Temperature
        ttk.Label(top_frame, text="Temperature:").pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        self.temperature_var = tk.DoubleVar(value=1.0)
        temperature_entry = ttk.Spinbox(top_frame, from_=0.1, to=2.0, increment=0.1, textvariable=self.temperature_var, width=3)
        temperature_entry.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Generate button
        self.generate_button = ttk.Button(top_frame, text="Generate Text", command=self.generate_text, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        
        # Middle frame - Results
        middle_frame = ttk.LabelFrame(self.generation_tab, text="Generated Text")
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 10))
        
        # Generated text
        self.generated_text = scrolledtext.ScrolledText(middle_frame, wrap=tk.WORD, height=15)
        self.generated_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame - Instructions
        bottom_frame = ttk.Frame(self.generation_tab)
        bottom_frame.pack(fill=tk.X)
        
        ttk.Label(bottom_frame, text="Instructions:", style="Bold.TLabel").pack(anchor=tk.W)
        instructions = (
            "1. Enter a starting context - the model will generate text continuing from this context.\n"
            "2. Set the number of words to generate.\n"
            "3. Adjust the temperature to control randomness (higher = more random, lower = more predictable).\n"
            "4. Click 'Generate Text' to create a text sequence."
        )
        ttk.Label(bottom_frame, text=instructions, wraplength=600).pack(anchor=tk.W, pady=(5, 0))
    
    def load_text_from_file(self):
        """
        Load training text from a file.
        """
        filepath = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.training_text.delete(1.0, tk.END)
            self.training_text.insert(tk.END, text)
            self.status_var.set(f"Loaded text from {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load text: {str(e)}")
    
    def clear_training_text(self):
        """
        Clear the training text.
        """
        self.training_text.delete(1.0, tk.END)
    
    def train_model(self):
        """
        Train the model on the provided text.
        """
        # Get training text
        training_text = self.training_text.get(1.0, tk.END).strip()
        
        if not training_text:
            messagebox.showerror("Error", "Please enter some training text.")
            return
        
        # Get parameters
        context_size = self.context_size_var.get()
        
        try:
            hidden_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',')]
        except ValueError:
            messagebox.showerror("Error", "Invalid hidden layers format. Use comma-separated integers (e.g., 64,32).")
            return
        
        learning_rate = self.learning_rate_var.get()
        n_iterations = self.iterations_var.get()
        
        # Reset progress
        self.progress_var.set(0)
        self.progress_label.config(text="Starting training...")
        
        # Clear the plot
        self.ax.clear()
        self.ax.set_title('Training and Validation Loss')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True)
        self.canvas.draw()
        
        # Disable train button, enable stop button
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Reset stop event
        self.stop_training_event.clear()
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self._train_model_thread,
            args=(training_text, context_size, hidden_layers, learning_rate, n_iterations)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _train_model_thread(self, text, context_size, hidden_layers, learning_rate, n_iterations):
        """
        Thread function for training the model.
        """
        try:
            # Import here to avoid circular imports
            from multi_layer_perceptron import MultiLayerPerceptron
            
            # Log start of training
            self.progress_queue.put("Starting model training...")
            self.progress_queue.put((0, n_iterations, 0, 0, "Creating model..."))
            
            # Create model
            self.model = MultiLayerPerceptron(
                context_size=context_size,
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                n_iterations=n_iterations,
                tokenizer_type='wordpiece',
                vocab_size=10000,
                use_pretrained=True,
                random_state=42
            )
            
            # Create a detailed logging wrapper for the progress callback
            def detailed_progress_callback(iteration, total_iterations, train_loss, val_loss, message=None):
                # Log every interaction for more detailed feedback
                log_message = message if message else f"Iteration {iteration}/{total_iterations} - Processing data batch"
                self.progress_queue.put((iteration, total_iterations, train_loss, val_loss, log_message))
                
                # Update the status bar with more detailed information
                status_msg = f"Training: iteration {iteration}/{total_iterations}"
                if train_loss is not None:
                    status_msg += f", loss: {train_loss:.6f}"
                if val_loss is not None:
                    status_msg += f", val_loss: {val_loss:.6f}"
                if message:
                    status_msg += f" - {message}"
                self.root.after(0, lambda: self.status_var.set(status_msg))
                
                # Update the plot if we have enough data
                if hasattr(self.model, 'training_loss') and len(self.model.training_loss) > 1:
                    self.root.after(0, self.update_training_plot)
            
            # Train the model with enhanced logging
            self.model.fit(
                text,
                progress_callback=detailed_progress_callback,
                stop_event=self.stop_training_event
            )
            
            # Signal completion
            self.progress_queue.put("Training complete")
            self.progress_queue.put((n_iterations, n_iterations, 
                                   self.model.training_loss[-1] if hasattr(self.model, 'training_loss') and self.model.training_loss else 0, 
                                   self.model.validation_loss[-1] if hasattr(self.model, 'validation_loss') and self.model.validation_loss else 0, 
                                   "Training complete"))
            
            # Update model info
            if hasattr(self.model, 'vocabulary'):
                self.root.after(0, lambda: self.vocab_size_var.set(str(len(self.model.vocabulary))))
            
            if hasattr(self.model, 'hidden_layers') and hasattr(self.model, 'input_size') and hasattr(self.model, 'output_size'):
                arch_str = f"Input({self.model.input_size}) → "
                for layer_size in self.model.hidden_layers:
                    arch_str += f"Hidden({layer_size}) → "
                arch_str += f"Output({self.model.output_size})"
                self.root.after(0, lambda: self.arch_var.set(arch_str))
            
            # Update model status
            self.root.after(0, lambda: self.model_status_var.set("Trained"))
            
            # Enable prediction and generation buttons
            self.root.after(0, lambda: self.predict_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.generate_button.config(state=tk.NORMAL))
            
        except Exception as e:
            # Handle exceptions with more detailed error information
            error_msg = f"Error during training: {str(e)}"
            self.progress_queue.put(error_msg)
            self.progress_queue.put((0, n_iterations, 0, 0, error_msg))
            
            # Log the full exception details
            import traceback
            trace_details = traceback.format_exc()
            print(f"Exception in training thread:\n{trace_details}")
            
            # Re-enable train button
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
    
    def update_training_plot(self):
        """
        Update the training plot with the latest data.
        """
        if not hasattr(self.model, 'training_loss') or not self.model.training_loss:
            return
        
        # Clear the plot
        self.ax.clear()
        
        # Plot training loss
        if hasattr(self.model, 'iteration_count') and len(self.model.iteration_count) == len(self.model.training_loss):
            self.ax.plot(self.model.iteration_count, self.model.training_loss, 'b-', label='Training Loss')
        else:
            self.ax.plot(self.model.training_loss, 'b-', label='Training Loss')
        
        # Plot validation loss if available
        if hasattr(self.model, 'validation_loss') and self.model.validation_loss:
            if hasattr(self.model, 'iteration_count') and len(self.model.iteration_count) == len(self.model.validation_loss):
                self.ax.plot(self.model.iteration_count, self.model.validation_loss, 'r-', label='Validation Loss')
            else:
                self.ax.plot(self.model.validation_loss, 'r-', label='Validation Loss')
        
        # Set labels and legend
        self.ax.set_title('Training and Validation Loss')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.ax.grid(True)
        
        # Redraw the canvas
        self.canvas.draw()
    
    def check_progress_queue(self):
        """
        Check the progress queue for updates from the training thread.
        """
        try:
            while True:
                # Get all available messages from the queue
                message = self.progress_queue.get_nowait()
                
                if isinstance(message, tuple) and len(message) >= 3:
                    iteration, total_iterations, train_loss, val_loss = message[:4]
                    message_text = message[4] if len(message) > 4 else None
                    
                    # Update progress bar
                    progress = (iteration / total_iterations) * 100
                    self.progress_var.set(progress)
                    
                    # Update progress label
                    if message_text:
                        self.progress_label.config(text=f"Iteration {iteration}/{total_iterations} - {message_text}")
                    else:
                        self.progress_label.config(text=f"Iteration {iteration}/{total_iterations}")
                    
                    # If training is complete
                    if message_text and "complete" in message_text.lower():
                        self.on_training_complete()
                
                elif isinstance(message, str):
                    # Update status bar
                    self.status_var.set(message)
                    
                    # If training is complete
                    if "complete" in message.lower():
                        self.on_training_complete()
                
        except queue.Empty:
            # No more messages, schedule the next check
            self.root.after(100, self.check_progress_queue)
    
    def on_training_complete(self):
        """
        Called when training is complete.
        """
        # Re-enable train button, disable stop button
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Enable save button
        self.save_model_button.config(state=tk.NORMAL)
        
        # Enable prediction and generation buttons
        self.predict_button.config(state=tk.NORMAL)
        self.generate_button.config(state=tk.NORMAL)
    
    def stop_training(self):
        """
        Stop the training process.
        """
        self.stop_training_event.set()
        self.status_var.set("Stopping training...")
    
    def save_model(self):
        """
        Save the trained model to a file.
        """
        if not self.model:
            messagebox.showerror("Error", "No model to save.")
            return
        
        # Get file path
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Save the model
            self.model.save_model(filepath)
            
            # Update status
            self.status_var.set(f"Model saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """
        Load a trained model from a file.
        """
        # Get file path
        filepath = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Import here to avoid circular imports
            from multi_layer_perceptron import MultiLayerPerceptron
            
            # Load the model
            self.model = MultiLayerPerceptron.load_model(filepath)
            
            # Update status
            self.status_var.set(f"Model loaded from {filepath}")
            
            # Update UI
            self.context_size_var.set(self.model.context_size)
            self.hidden_layers_var.set(','.join(str(x) for x in self.model.hidden_layers))
            self.learning_rate_var.set(self.model.learning_rate)
            self.iterations_var.set(self.model.n_iterations)
            
            # Update model info
            if hasattr(self.model, 'vocabulary'):
                self.vocab_size_var.set(str(len(self.model.vocabulary)))
            
            if hasattr(self.model, 'hidden_layers') and hasattr(self.model, 'input_size') and hasattr(self.model, 'output_size'):
                arch_str = f"Input({self.model.input_size}) → "
                for layer_size in self.model.hidden_layers:
                    arch_str += f"Hidden({layer_size}) → "
                arch_str += f"Output({self.model.output_size})"
                self.arch_var.set(arch_str)
            
            # Update model status
            self.model_status_var.set("Loaded")
            
            # Enable buttons
            self.save_model_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.NORMAL)
            self.generate_button.config(state=tk.NORMAL)
            
            # Update plot if we have data
            if hasattr(self.model, 'training_loss') and len(self.model.training_loss) > 0:
                self.update_training_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def predict_next_word(self):
        """
        Predict the next word based on the context.
        """
        if not self.model:
            messagebox.showerror("Error", "No model available. Please train or load a model first.")
            return
        
        # Get context
        context = self.context_entry.get().strip()
        
        if not context:
            messagebox.showerror("Error", "Please enter a context.")
            return
        
        # Get number of predictions
        top_n = self.top_n_var.get()
        
        try:
            # Predict next word and get prediction info
            predicted_word, prediction_info = self.model.predict_next_word(context)
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Get the probabilities from the prediction info
            if 'probabilities' in prediction_info and prediction_info['probabilities'] is not None:
                # Sort probabilities in descending order
                word_probs = sorted(
                    [(self.model.idx_to_word[idx], prob) 
                     for idx, prob in enumerate(prediction_info['probabilities'][0])
                     if prob > 0.001],  # Filter out very low probabilities
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Take only the top N
                word_probs = word_probs[:top_n]
                
                # Add to results
                for word, prob in word_probs:
                    self.results_tree.insert("", "end", values=(word, f"{prob:.6f}"))
            else:
                # Fallback if probabilities not available
                self.results_tree.insert("", "end", values=(predicted_word, "N/A"))
            
            # Update status
            self.status_var.set(f"Predicted next word for context: '{context}'")
            
            # Show any adjustments made to the context
            if prediction_info.get('adjustment_made', False):
                adjustment_msg = "Note: The context was adjusted:\n"
                if 'adjusted_context' in prediction_info:
                    adjustment_msg += f"- Adjusted context: {' '.join(prediction_info['adjusted_context'])}\n"
                if 'adjustment_type' in prediction_info:
                    adjustment_msg += f"- Adjustment types: {', '.join(prediction_info['adjustment_type'])}\n"
                if 'unknown_words' in prediction_info:
                    adjustment_msg += f"- Unknown words: {', '.join(prediction_info['unknown_words'])}\n"
                messagebox.showinfo("Context Adjustment", adjustment_msg)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict next word: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_text(self):
        """
        Generate text from the model.
        """
        if not self.model:
            messagebox.showerror("Error", "No model available. Please train or load a model first.")
            return
        
        # Get context
        context = self.gen_context_entry.get().strip()
        
        if not context:
            messagebox.showerror("Error", "Please enter a starting context.")
            return
        
        # Get parameters
        num_words = self.num_words_var.get()
        temperature = self.temperature_var.get()
        
        try:
            # Generate text using predict_next_n_words
            predicted_words, prediction_info = self.model.predict_next_n_words(
                context, 
                n=num_words, 
                temperature=temperature
            )
            
            # Combine the context with the predicted words
            if isinstance(context, str):
                context_words = context.split()
            else:
                context_words = context
                
            # Create the full generated text
            full_text = " ".join(context_words) + " " + " ".join(predicted_words)
            
            # Display generated text
            self.generated_text.delete(1.0, tk.END)
            self.generated_text.insert(tk.END, full_text)
            
            # Update status
            self.status_var.set(f"Generated {num_words} words from context: '{context}'")
            
            # Show generation statistics if available
            if 'generation_stats' in prediction_info:
                stats = prediction_info['generation_stats']
                stats_msg = "Generation Statistics:\n"
                if 'successful_predictions' in stats:
                    stats_msg += f"- Successful predictions: {stats['successful_predictions']}/{num_words}\n"
                if 'avg_confidence' in stats:
                    stats_msg += f"- Average confidence: {stats['avg_confidence']:.4f}\n"
                if 'errors' in stats and stats['errors'] > 0:
                    stats_msg += f"- Errors encountered: {stats['errors']}\n"
                
                messagebox.showinfo("Generation Statistics", stats_msg)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """
    Main function to run the application.
    """
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 10))
    style.configure("TLabel", font=("Arial", 10))
    style.configure("Header.TLabel", font=("Arial", 12, "bold"))
    style.configure("Bold.TLabel", font=("Arial", 10, "bold"))
    
    app = CompleteMlpUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()