#!/usr/bin/env python3
"""
UI for the Single-Layer Perceptron language model.
This UI extends the MultiLayerPerceptronUI with simplified options for a single-layer model.
"""

# Check if tkinter is available
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import pickle
    import threading
    from multi_layer_perceptron_ui import MultiLayerPerceptronUI
    from multi_layer_perceptron import MultiLayerPerceptron
    from single_layer_perceptron import SingleLayerPerceptron
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    # Define a placeholder class for documentation purposes
    class SingleLayerPerceptronUI:
        """
        This class requires tkinter which is not installed.
        Please install tkinter to use the UI.
        """
        def __init__(self, *args, **kwargs):
            raise ImportError("Tkinter is not available. Please install it to use the UI.")

# Only define the real class if tkinter is available
if TKINTER_AVAILABLE:
    class SingleLayerPerceptronUI(MultiLayerPerceptronUI):
        """
        UI for the Single-Layer Perceptron language model.
        This UI simplifies the MultiLayerPerceptronUI to focus on a single hidden layer.
        """
        
        def __init__(self, root):
            """
            Initialize the GUI with single-layer perceptron support.
            
            Parameters:
            -----------
            root : tk.Tk
                Root window
            """
            # Call the parent class initializer
            super().__init__(root)
            
            # Update the window title
            self.root.title("Single-Layer Perceptron Language Model")
            
            # Modify the parameters frame to simplify for single-layer perceptron
            self._modify_parameters_frame()
            
            # Update the model info section
            self._update_model_info_section()
        
        def _modify_parameters_frame(self):
            """
            Modify the parameters frame to simplify for single-layer perceptron.
            """
            # Find the parameters frame in the training tab
            params_frame = self.params_frame
            
            if params_frame:
                # Find and remove the hidden layers frame
                for child in params_frame.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for widget in child.winfo_children():
                            if isinstance(widget, ttk.Label) and widget.cget("text") == "Hidden Layers:":
                                child.destroy()
                                break
                
                # Add hidden size frame
                hidden_frame = ttk.Frame(params_frame)
                hidden_frame.pack(fill=tk.X, pady=5, after=params_frame.winfo_children()[0])
                ttk.Label(hidden_frame, text="Hidden Layer Size:").pack(side=tk.LEFT)
                self.hidden_size_var = tk.IntVar(value=64)
                hidden_size_entry = ttk.Spinbox(hidden_frame, from_=8, to=256, increment=8, 
                                              textvariable=self.hidden_size_var, width=5)
                hidden_size_entry.pack(side=tk.LEFT, padx=(5, 0))
                ttk.Label(hidden_frame, text="(Number of neurons in the hidden layer)").pack(side=tk.LEFT, padx=(5, 0))
        
        def _update_model_info_section(self):
            """
            Update the model info section to show single-layer perceptron info.
            """
            # Find the info frame
            info_frame = self.info_frame
            
            if info_frame:
                # Add model type info
                model_type_frame = ttk.Frame(info_frame)
                if info_frame.winfo_children():
                    model_type_frame.pack(fill=tk.X, pady=5, before=info_frame.winfo_children()[0])
                else:
                    model_type_frame.pack(fill=tk.X, pady=5)
                ttk.Label(model_type_frame, text="Model Type:").pack(side=tk.LEFT)
                self.model_type_info_var = tk.StringVar(value="Single-Layer Perceptron")
                ttk.Label(model_type_frame, textvariable=self.model_type_info_var).pack(side=tk.LEFT, padx=(5, 0))
        
        def train_model(self):
            """
            Train the model on the provided text using a single-layer perceptron.
            """
            # Get training text
            training_text = self.training_text.get("1.0", tk.END).strip()
            if not training_text:
                messagebox.showerror("Error", "Please provide training text.")
                return
            
            # Get parameters
            context_size = self.context_size_var.get()
            hidden_size = self.hidden_size_var.get()
            learning_rate = self.learning_rate_var.get()
            n_iterations = self.iterations_var.get()
            
            # Create model
            self.model = SingleLayerPerceptron(
                context_size=context_size,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                n_iterations=n_iterations,
                tokenizer_type='wordpiece',  # Default to WordPiece tokenizer
                vocab_size=10000,            # Default vocabulary size
                use_pretrained=True          # Use pretrained embeddings
            )
            
            # Update model info
            self.model_type_info_var.set("Single-Layer Perceptron")
            
            # The input_size and output_size might not be initialized yet
            # They will be set after the model processes some text
            # So we'll just show the hidden size for now
            self.arch_var.set(f"Hidden Layer Size: {hidden_size}")
            
            # Reset progress
            self.progress_var.set(0)
            self.progress_label.config(text="Starting training...")
            
            # Clear the plot if matplotlib is available
            if hasattr(self, 'ax') and hasattr(self, 'canvas'):
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
            with open(filepath, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Check if it's a SingleLayerPerceptron model
            if not isinstance(loaded_model, SingleLayerPerceptron):
                # Try to convert the model if it's a MultiLayerPerceptron
                if isinstance(loaded_model, MultiLayerPerceptron):
                    # Create a new SingleLayerPerceptron with the same parameters
                    self.model = SingleLayerPerceptron(
                        context_size=loaded_model.context_size,
                        hidden_size=loaded_model.hidden_layers[0] if loaded_model.hidden_layers else 64,
                        learning_rate=loaded_model.learning_rate,
                        n_iterations=loaded_model.n_iterations,
                        tokenizer_type=loaded_model.tokenizer_type if hasattr(loaded_model, 'tokenizer_type') else 'wordpiece',
                        vocab_size=len(loaded_model.vocabulary) if loaded_model.vocabulary else 10000,
                        use_pretrained=loaded_model.use_pretrained if hasattr(loaded_model, 'use_pretrained') else True
                    )
                    
                    # Copy over the trained weights and vocabulary
                    self.model.weights = loaded_model.weights[:2]  # Only take the first two weight matrices
                    self.model.biases = loaded_model.biases[:2]    # Only take the first two bias vectors
                    self.model.vocabulary = loaded_model.vocabulary
                    self.model.word_to_index = loaded_model.word_to_index
                    self.model.index_to_word = loaded_model.index_to_word
                    self.model.training_loss = loaded_model.training_loss
                    self.model.validation_loss = loaded_model.validation_loss
                    
                    messagebox.showinfo("Model Converted", 
                                       "The loaded model was converted from a Multi-Layer Perceptron to a Single-Layer Perceptron.")
                else:
                    messagebox.showerror("Error", "The loaded model is not compatible with SingleLayerPerceptron.")
                    return
            else:
                self.model = loaded_model
            
            # Update status
            self.status_var.set(f"Model loaded from {filepath}")
            
            # Update UI
            self.context_size_var.set(self.model.context_size)
            self.hidden_size_var.set(self.model.hidden_size)
            self.learning_rate_var.set(self.model.learning_rate)
            self.iterations_var.set(self.model.n_iterations)
            
            # Update model info
            self.vocab_size_var.set(str(len(self.model.vocabulary)) if self.model.vocabulary else "N/A")
            
            # Update architecture info
            arch_str = f"Input({self.model.input_size}) → Hidden({self.model.hidden_size}) → Output({self.model.output_size})"
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
    
    def on_training_complete(self):
        """
        Handle the completion of training for the single-layer perceptron.
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
            
            # Update architecture info for single-layer perceptron
            if hasattr(self.model, 'input_size') and hasattr(self.model, 'output_size'):
                arch_str = f"Input({self.model.input_size}) → Hidden({self.model.hidden_size}) → Output({self.model.output_size})"
                self.arch_var.set(arch_str)
            else:
                self.arch_var.set(f"Hidden Layer Size: {self.model.hidden_size}")
    
    def update_training_plot(self):
        """
        Update the training plot with the latest loss values.
        """
        # Check if matplotlib is available
        if not hasattr(self, 'ax') or not hasattr(self, 'canvas'):
            return
            
        # Clear the plot
        self.ax.clear()
        
        # Plot training and validation loss
        if hasattr(self.model, 'iteration_count') and len(self.model.iteration_count) > 0:
            self.ax.plot(self.model.iteration_count, self.model.training_loss, label='Training Loss')
            self.ax.plot(self.model.iteration_count, self.model.validation_loss, label='Validation Loss')
            self.ax.set_xlabel('Iteration')
            self.ax.set_ylabel('Loss')
            self.ax.set_title('Training Progress')
            self.ax.legend()
            self.ax.grid(True)
            
            # Redraw the canvas
            self.canvas.draw()