#!/usr/bin/env python3
"""
Basic UI for the Multi-Layer Perceptron model.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import pickle
import os
import sys

class BasicMlpUI:
    """
    A basic UI for the Multi-Layer Perceptron model.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        """
        self.root = root
        self.root.title("Basic MLP UI")
        self.root.geometry("800x600")
        
        # Create model
        self.model = None
        self.training_thread = None
        self.stop_training_event = threading.Event()
        self.progress_queue = queue.Queue()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Training data frame
        training_frame = ttk.LabelFrame(self.main_frame, text="Training Data")
        training_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Training text
        self.training_text = scrolledtext.ScrolledText(training_frame, wrap=tk.WORD, height=10)
        self.training_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Text buttons
        text_buttons_frame = ttk.Frame(training_frame)
        text_buttons_frame.pack(fill=tk.X)
        
        load_file_button = ttk.Button(text_buttons_frame, text="Load from File", command=self.load_text_from_file)
        load_file_button.pack(side=tk.LEFT, padx=5)
        
        clear_text_button = ttk.Button(text_buttons_frame, text="Clear Text", command=self.clear_training_text)
        clear_text_button.pack(side=tk.LEFT, padx=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(self.main_frame, text="Model Parameters")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Context Size
        context_frame = ttk.Frame(params_frame)
        context_frame.pack(fill=tk.X, pady=5)
        ttk.Label(context_frame, text="Context Size:").pack(side=tk.LEFT)
        self.context_size_var = tk.IntVar(value=3)
        context_size_entry = ttk.Spinbox(context_frame, from_=1, to=5, textvariable=self.context_size_var, width=5)
        context_size_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Hidden Layers
        hidden_frame = ttk.Frame(params_frame)
        hidden_frame.pack(fill=tk.X, pady=5)
        ttk.Label(hidden_frame, text="Hidden Layers:").pack(side=tk.LEFT)
        self.hidden_layers_var = tk.StringVar(value="64,32")
        hidden_layers_entry = ttk.Entry(hidden_frame, textvariable=self.hidden_layers_var, width=15)
        hidden_layers_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Learning rate
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        learning_rate_entry = ttk.Spinbox(lr_frame, from_=0.001, to=1.0, increment=0.01, textvariable=self.learning_rate_var, width=5)
        learning_rate_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Iterations
        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iterations_var = tk.IntVar(value=1000)
        iterations_entry = ttk.Spinbox(iter_frame, from_=100, to=10000, increment=100, textvariable=self.iterations_var, width=7)
        iterations_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Training buttons
        self.train_button = ttk.Button(buttons_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Model operations
        self.save_model_button = ttk.Button(buttons_frame, text="Save Model", command=self.save_model, state=tk.DISABLED)
        self.save_model_button.pack(side=tk.LEFT, padx=5)
        
        self.load_model_button = ttk.Button(buttons_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.main_frame, text="Training Progress")
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Progress label
        self.progress_label = ttk.Label(progress_frame, text="Not started")
        self.progress_label.pack(anchor=tk.W, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start checking the progress queue
        self.check_progress_queue()
    
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
            
            # Enable save button
            self.save_model_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """
    Main function to run the application.
    """
    root = tk.Tk()
    app = BasicMlpUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()