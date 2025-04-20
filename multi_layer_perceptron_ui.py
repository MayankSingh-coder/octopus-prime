import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import os
import re
import pickle
from multi_layer_perceptron import MultiLayerPerceptron

class MultiLayerPerceptronUI:
    """
    A UI for training and using a multi-layer perceptron language model.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Parameters:
        -----------
        root : tk.Tk
            Root window
        """
        self.root = root
        self.root.title("Multi-Layer Perceptron Language Model")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        self.style.configure("Bold.TLabel", font=("Arial", 10, "bold"))
        
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
        self.info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Model Status
        status_frame = ttk.Frame(self.info_frame)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="Model Status:").pack(side=tk.LEFT)
        self.model_status_var = tk.StringVar(value="Not trained")
        ttk.Label(status_frame, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=(5, 0))
    
        # Left frame - Training data and parameters
        ttk.Label(left_frame, text="Training Data", style="Header.TLabel").pack(pady=(0, 5), anchor=tk.W)
        
        # Text input frame
        text_input_frame = ttk.Frame(left_frame)
        text_input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Training text
        self.training_text = scrolledtext.ScrolledText(text_input_frame, wrap=tk.WORD, height=15)
        self.training_text.pack(fill=tk.BOTH, expand=True)
        
        # Sample text button
        sample_text_button = ttk.Button(text_input_frame, text="Load Sample Text", command=self.load_sample_text)
        sample_text_button.pack(side=tk.LEFT, pady=(5, 0))
        
        # Load from file button
        load_file_button = ttk.Button(text_input_frame, text="Load from File", command=self.load_text_from_file)
        load_file_button.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Clear text button
        clear_text_button = ttk.Button(text_input_frame, text="Clear Text", command=self.clear_training_text)
        clear_text_button.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Parameters frame
        left_frame = ttk.Frame(self.training_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.params_frame = ttk.LabelFrame(left_frame, text="Model Parameters")
        self.params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Context Size
        context_frame = ttk.Frame(self.params_frame)
        context_frame.pack(fill=tk.X, pady=5)
        ttk.Label(context_frame, text="Context Size:").pack(side=tk.LEFT)
        self.context_size_var = tk.IntVar(value=2)
        context_size_entry = ttk.Spinbox(context_frame, from_=1, to=5, textvariable=self.context_size_var, width=5)
        context_size_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(context_frame, text="(Number of previous words to use as context)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Hidden layers
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
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Model info
        info_frame = ttk.LabelFrame(right_frame, text="Model Information")
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Vocabulary size
        vocab_frame = ttk.Frame(info_frame)
        vocab_frame.pack(fill=tk.X, pady=5)
        ttk.Label(vocab_frame, text="Vocabulary Size:").pack(side=tk.LEFT)
        self.vocab_size_var = tk.StringVar(value="N/A")
        ttk.Label(vocab_frame, textvariable=self.vocab_size_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Model architecture
        arch_frame = ttk.Frame(info_frame)
        arch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(arch_frame, text="Model Architecture:").pack(side=tk.LEFT)
        self.arch_var = tk.StringVar(value="N/A")
        ttk.Label(arch_frame, textvariable=self.arch_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Model status
        status_frame = ttk.Frame(info_frame)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="Model Status:").pack(side=tk.LEFT)
        self.model_status_var = tk.StringVar(value="Not trained")
        ttk.Label(status_frame, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=(5, 0))
    
    def setup_prediction_tab(self):
        """
        Set up the prediction tab.
        """
        # Top frame - Input
        top_frame = ttk.Frame(self.prediction_tab)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(top_frame, text="Enter Context Words:", style="Bold.TLabel").pack(anchor=tk.W)
        
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
        
        ttk.Label(top_frame, text="Enter Initial Context:", style="Bold.TLabel").pack(anchor=tk.W)
        
        # Context entry
        self.gen_context_entry = ttk.Entry(top_frame, width=50)
        self.gen_context_entry.pack(side=tk.LEFT, pady=(5, 0))
        
        # Number of words to generate
        ttk.Label(top_frame, text="Words to Generate:").pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        self.gen_words_var = tk.IntVar(value=20)
        gen_words_entry = ttk.Spinbox(top_frame, from_=1, to=100, textvariable=self.gen_words_var, width=3)
        gen_words_entry.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Generate button
        self.generate_button = ttk.Button(top_frame, text="Generate Text", command=self.generate_text, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        
        # Middle frame - Generated text
        middle_frame = ttk.LabelFrame(self.generation_tab, text="Generated Text")
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Generated text
        self.generated_text = scrolledtext.ScrolledText(middle_frame, wrap=tk.WORD)
        self.generated_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame - Instructions
        bottom_frame = ttk.Frame(self.generation_tab)
        bottom_frame.pack(fill=tk.X)
        
        ttk.Label(bottom_frame, text="Instructions:", style="Bold.TLabel").pack(anchor=tk.W)
        instructions = (
            "1. Enter any text as initial context - the model will automatically adjust if needed.\n"
            "2. Specify how many words you want to generate.\n"
            "3. Click 'Generate Text' to create a text sequence starting with your context.\n"
            "4. If your context is adjusted (too short, too long, or contains unknown words), you'll see details in a popup."
        )
        ttk.Label(bottom_frame, text=instructions, wraplength=600).pack(anchor=tk.W, pady=(5, 0))
    
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
                        self.progress_label.config(text=f"Iteration {iteration}/{total_iterations} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - {message_text}")
                    else:
                        self.progress_label.config(text=f"Iteration {iteration}/{total_iterations} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    # Update plot if we have data
                    if self.model and hasattr(self.model, 'training_loss') and len(self.model.training_loss) > 0:
                        self.update_training_plot()
                    
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
    
    def update_training_plot(self):
        """
        Update the training plot with the latest loss values.
        """
        # Clear the plot
        self.ax.clear()
        
        # Plot training and validation loss
        if len(self.model.iteration_count) > 0:
            self.ax.plot(self.model.iteration_count, self.model.training_loss, label='Training Loss')
            self.ax.plot(self.model.iteration_count, self.model.validation_loss, label='Validation Loss')
            self.ax.set_xlabel('Iteration')
            self.ax.set_ylabel('Loss')
            self.ax.set_title('Training Progress')
            self.ax.legend()
            self.ax.grid(True)
            
            # Redraw the canvas
            self.canvas.draw()
    
    def on_training_complete(self):
        """
        Handle the completion of training.
        """
        # Enable/disable buttons
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_model_button.config(state=tk.NORMAL)
        self.predict_button.config(state=tk.NORMAL)
        self.generate_button.config(state=tk.NORMAL)
        
        # Update model status
        self.model_status_var.set("Trained")
        
        # Update model info
        if self.model:
            self.vocab_size_var.set(str(len(self.model.vocabulary)) if self.model.vocabulary else "N/A")
            
            # Update architecture info
            if hasattr(self.model, 'hidden_layers') and self.model.hidden_layers:
                arch_str = f"Input({self.model.input_size}) → "
                for layer_size in self.model.hidden_layers:
                    arch_str += f"Hidden({layer_size}) → "
                arch_str += f"Output({self.model.output_size})"
                self.arch_var.set(arch_str)
            else:
                self.arch_var.set("N/A")
    
    def train_model(self):
        """
        Train the model on the provided text.
        """
        # Get training text
        training_text = self.training_text.get("1.0", tk.END).strip()
        if not training_text:
            messagebox.showerror("Error", "Please provide training text.")
            return
        
        # Get parameters
        context_size = self.context_size_var.get()
        learning_rate = self.learning_rate_var.get()
        n_iterations = self.iterations_var.get()
        
        # Parse hidden layers
        try:
            hidden_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',')]
        except ValueError:
            messagebox.showerror("Error", "Invalid hidden layers format. Use comma-separated integers (e.g., 64,32).")
            return
        
        # Create model
        self.model = MultiLayerPerceptron(
            context_size=context_size,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            tokenizer_type='wordpiece',  # Default to WordPiece tokenizer
            vocab_size=10000,            # Default vocabulary size
            use_pretrained=True          # Use pretrained embeddings
        )
        
        # Reset progress
        self.progress_var.set(0)
        self.progress_label.config(text="Starting training...")
        
        # Clear the plot
        self.ax.clear()
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        self.canvas.draw()
        
        # Reset stop event
        self.stop_training_event.clear()
        
        # Enable/disable buttons
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_model_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.DISABLED)
        
        # Update status
        self.model_status_var.set("Training...")
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self._train_model_thread,
            args=(training_text,)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _train_model_thread(self, text):
        """
        Thread function for training the model.
        """
        try:
            # Log start of training
            self.progress_queue.put("Starting model training...")
            self.progress_queue.put((0, self.model.n_iterations, 0, 0, "Preprocessing text..."))
            
            # Create a detailed logging wrapper for the progress callback
            def detailed_progress_callback(iteration, total_iterations, train_loss, val_loss, message=None):
                # Ensure we have valid values for losses (handle None or invalid values)
                train_loss = 0.0 if train_loss is None or np.isnan(train_loss) else float(train_loss)
                val_loss = 0.0 if val_loss is None or np.isnan(val_loss) else float(val_loss)
                
                # Log every interaction for more detailed feedback
                log_message = message if message else f"Iteration {iteration}/{total_iterations} - Processing data batch"
                self.progress_queue.put((iteration, total_iterations, train_loss, val_loss, log_message))
                
                # Update the status bar with more detailed information
                status_msg = f"Training: iteration {iteration}/{total_iterations}, loss: {train_loss:.6f}, val_loss: {val_loss:.6f}"
                if message:
                    status_msg += f" - {message}"
                self.root.after(0, lambda: self.status_var.set(status_msg))
                
                # Force UI update to show progress in real-time
                self.root.update_idletasks()
            
            # Train the model with enhanced logging
            self.model.fit(
                text,
                progress_callback=detailed_progress_callback,
                stop_event=self.stop_training_event
            )
            
            # Signal completion
            self.progress_queue.put("Training complete")
            self.progress_queue.put((self.model.n_iterations, self.model.n_iterations, 
                                   self.model.training_loss[-1] if self.model.training_loss else 0, 
                                   self.model.validation_loss[-1] if self.model.validation_loss else 0, 
                                   "Training complete"))
            
        except Exception as e:
            # Handle exceptions with more detailed error information
            error_msg = f"Error during training: {str(e)}"
            self.progress_queue.put(error_msg)
            self.progress_queue.put((0, self.model.n_iterations, 0, 0, error_msg))
            
            # Log the full exception details
            import traceback
            trace_details = traceback.format_exc()
            print(f"Exception in training thread:\n{trace_details}")
            
            # Re-enable train button
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
    
    def _progress_callback(self, iteration, total_iterations, train_loss, val_loss, message=None):
        """
        Callback function for reporting training progress.
        """
        if message:
            self.progress_queue.put((iteration, total_iterations, train_loss, val_loss, message))
        else:
            self.progress_queue.put((iteration, total_iterations, train_loss, val_loss))
    
    def stop_training(self):
        """
        Stop the training process.
        """
        if self.training_thread and self.training_thread.is_alive():
            # Set the stop event
            self.stop_training_event.set()
            
            # Update status
            self.status_var.set("Stopping training...")
            
            # Disable stop button
            self.stop_button.config(state=tk.DISABLED)
    
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
            self.vocab_size_var.set(str(len(self.model.vocabulary)) if self.model.vocabulary else "N/A")
            
            # Update architecture info
            if hasattr(self.model, 'hidden_layers') and self.model.hidden_layers:
                arch_str = f"Input({self.model.input_size}) → "
                for layer_size in self.model.hidden_layers:
                    arch_str += f"Hidden({layer_size}) → "
                arch_str += f"Output({self.model.output_size})"
                self.arch_var.set(arch_str)
            else:
                self.arch_var.set("N/A")
            
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
    
    def predict_next_word(self):
        """
        Predict the next word based on the context.
        """
        if not self.model:
            messagebox.showerror("Error", "No model loaded.")
            return
        
        # Get context
        context_text = self.context_entry.get().strip()
        
        try:
            # Get top predictions
            top_n = self.top_n_var.get()
            predictions, info = self.model.get_top_predictions(context_text, top_n=top_n)
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Add new results
            for word, prob in predictions:
                self.results_tree.insert("", "end", values=(word, f"{prob:.6f}"))
            
            # Show context adjustment info if any
            if info["adjustment_made"]:
                adjustment_msg = ""
                if "adjustment_type" in info:
                    if "padded_beginning" in info["adjustment_type"]:
                        adjustment_msg += f"Context was too short. Added padding at the beginning.\n"
                    if "truncated_beginning" in info["adjustment_type"]:
                        adjustment_msg += f"Context was too long. Used only the most recent {self.model.context_size} words.\n"
                    if "replaced_unknown" in info["adjustment_type"]:
                        adjustment_msg += f"Unknown words were replaced with known vocabulary.\n"
                
                if "unknown_words" in info:
                    adjustment_msg += f"Unknown words: {', '.join(info['unknown_words'])}\n"
                
                adjustment_msg += f"Adjusted context: '{' '.join(info['adjusted_context'])}'"
                
                messagebox.showinfo("Context Adjustment", adjustment_msg)
            
            # Update status
            self.status_var.set(f"Predicted next word after '{context_text}'")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def generate_text(self):
        """
        Generate text starting from the given context.
        """
        if not self.model:
            messagebox.showerror("Error", "No model loaded.")
            return
        
        # Get context
        context_text = self.gen_context_entry.get().strip()
        
        try:
            # Generate text
            n_words = self.gen_words_var.get()
            generated_words, info = self.model.predict_next_n_words(context_text, n=n_words)
            
            # Format the result
            result_text = info["full_text"]
            
            # Display the result
            self.generated_text.delete("1.0", tk.END)
            self.generated_text.insert("1.0", result_text)
            
            # Show context adjustment info if any
            if info["adjustment_made"]:
                adjustment_msg = ""
                if info["adjusted_context"]:
                    adjustment_msg += f"Original context: '{' '.join(info['original_context'])}'\n"
                    adjustment_msg += f"Adjusted context: '{' '.join(info['adjusted_context'])}'\n\n"
                
                # Add information about the first prediction step
                first_step = info["prediction_steps"][0]
                if "adjustment_type" in first_step:
                    if "padded_beginning" in first_step["adjustment_type"]:
                        adjustment_msg += f"Context was too short. Added padding at the beginning.\n"
                    if "truncated_beginning" in first_step["adjustment_type"]:
                        adjustment_msg += f"Context was too long. Used only the most recent {self.model.context_size} words.\n"
                    if "replaced_unknown" in first_step["adjustment_type"]:
                        adjustment_msg += f"Unknown words were replaced with known vocabulary.\n"
                
                if "unknown_words" in first_step:
                    adjustment_msg += f"Unknown words: {', '.join(first_step['unknown_words'])}\n"
                
                messagebox.showinfo("Context Adjustment", adjustment_msg)
            
            # Update status
            self.status_var.set(f"Generated {n_words} words starting from '{context_text}'")
            
        except Exception as e:
            messagebox.showerror("Error", f"Text generation failed: {str(e)}")
    
    def load_sample_text(self):
        """
        Load sample text for training.
        """
        sample_text = """
        The quick brown fox jumps over the lazy dog. A wonderful serenity has taken possession of my entire soul, like these sweet mornings of spring which I enjoy with my whole heart. I am alone, and feel the charm of existence in this spot, which was created for the bliss of souls like mine. I am so happy, my dear friend, so absorbed in the exquisite sense of mere tranquil existence, that I neglect my talents.
        
        Far far away, behind the word mountains, far from the countries Vokalia and Consonantia, there live the blind texts. Separated they live in Bookmarksgrove right at the coast of the Semantics, a large language ocean. A small river named Duden flows by their place and supplies it with the necessary regelialia. It is a paradisematic country, in which roasted parts of sentences fly into your mouth.
        
        One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment.
        
        It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.
        """
        
        # Clear and insert the sample text
        self.training_text.delete("1.0", tk.END)
        self.training_text.insert("1.0", sample_text.strip())
        
        # Update status
        self.status_var.set("Sample text loaded")
    
    def load_text_from_file(self):
        """
        Load training text from a file.
        """
        # Get file path
        filepath = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clear and insert the text
            self.training_text.delete("1.0", tk.END)
            self.training_text.insert("1.0", text)
            
            # Update status
            self.status_var.set(f"Text loaded from {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load text: {str(e)}")
    
    def clear_training_text(self):
        """
        Clear the training text.
        """
        self.training_text.delete("1.0", tk.END)
        
        # Update status
        self.status_var.set("Training text cleared")


def main():
    """
    Main function to run the application.
    """
    root = tk.Tk()
    app = MultiLayerPerceptronUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()