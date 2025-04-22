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

from ..models.multi_layer_perceptron import MultiLayerPerceptron
from ..models.attention_perceptron import AttentionPerceptron

class CompleteMlpUI:
    """
    A complete UI for the Multi-Layer Perceptron model with all features.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        """
        self.root = root
        self.root.title("Neural Network Language Model")
        self.root.geometry("1280x800")  # Slightly wider to accommodate all elements
        
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
        
        # Model Type
        model_type_frame = ttk.Frame(self.info_frame)
        model_type_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_type_frame, text="Model Type:").pack(side=tk.LEFT)
        self.model_type_info_var = tk.StringVar(value="N/A")
        ttk.Label(model_type_frame, textvariable=self.model_type_info_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Vocabulary Size
        vocab_frame = ttk.Frame(self.info_frame)
        vocab_frame.pack(fill=tk.X, pady=5)
        ttk.Label(vocab_frame, text="Vocabulary Size:").pack(side=tk.LEFT)
        self.vocab_size_var = tk.StringVar(value="N/A")
        ttk.Label(vocab_frame, textvariable=self.vocab_size_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Training Progress
        progress_frame = ttk.Frame(self.info_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(progress_frame, text="Training Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.StringVar(value="N/A")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Training Loss
        loss_frame = ttk.Frame(self.info_frame)
        loss_frame.pack(fill=tk.X, pady=5)
        ttk.Label(loss_frame, text="Training Loss:").pack(side=tk.LEFT)
        self.loss_var = tk.StringVar(value="N/A")
        ttk.Label(loss_frame, textvariable=self.loss_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Validation Loss
        val_loss_frame = ttk.Frame(self.info_frame)
        val_loss_frame.pack(fill=tk.X, pady=5)
        ttk.Label(val_loss_frame, text="Validation Loss:").pack(side=tk.LEFT)
        self.val_loss_var = tk.StringVar(value="N/A")
        ttk.Label(val_loss_frame, textvariable=self.val_loss_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Model Parameters frame
        params_frame = ttk.LabelFrame(right_frame, text="Model Parameters")
        params_frame.pack(fill=tk.X, pady=10)
        
        # Model Type
        model_type_frame = ttk.Frame(params_frame)
        model_type_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_type_frame, text="Model Type:").pack(side=tk.LEFT)
        self.model_type_var = tk.StringVar(value="mlp")
        ttk.Radiobutton(model_type_frame, text="MLP", variable=self.model_type_var, value="mlp").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Radiobutton(model_type_frame, text="Attention", variable=self.model_type_var, value="attention").pack(side=tk.LEFT, padx=(5, 0))
        
        # Context Size
        context_frame = ttk.Frame(params_frame)
        context_frame.pack(fill=tk.X, pady=5)
        ttk.Label(context_frame, text="Context Size:").pack(side=tk.LEFT)
        self.context_size_var = tk.StringVar(value="3")
        ttk.Spinbox(context_frame, from_=1, to=10, textvariable=self.context_size_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Embedding Dimension
        embedding_frame = ttk.Frame(params_frame)
        embedding_frame.pack(fill=tk.X, pady=5)
        ttk.Label(embedding_frame, text="Embedding Dimension:").pack(side=tk.LEFT)
        self.embedding_dim_var = tk.StringVar(value="50")
        ttk.Spinbox(embedding_frame, from_=10, to=300, textvariable=self.embedding_dim_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Hidden Layers
        self.hidden_frame = ttk.Frame(params_frame)
        self.hidden_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.hidden_frame, text="Hidden Layers:").pack(side=tk.LEFT)
        self.hidden_layers_var = tk.StringVar(value="64,32")
        ttk.Entry(self.hidden_frame, textvariable=self.hidden_layers_var, width=15).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(self.hidden_frame, text="(comma-separated)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Attention Parameters (only visible when attention model is selected)
        self.attention_frame = ttk.LabelFrame(params_frame, text="Attention Parameters")
        
        # Attention Dimension
        attention_dim_frame = ttk.Frame(self.attention_frame)
        attention_dim_frame.pack(fill=tk.X, pady=5)
        ttk.Label(attention_dim_frame, text="Attention Dimension:").pack(side=tk.LEFT)
        self.attention_dim_var = tk.StringVar(value="40")
        ttk.Spinbox(attention_dim_frame, from_=10, to=300, textvariable=self.attention_dim_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Attention Heads
        attention_heads_frame = ttk.Frame(self.attention_frame)
        attention_heads_frame.pack(fill=tk.X, pady=5)
        ttk.Label(attention_heads_frame, text="Attention Heads:").pack(side=tk.LEFT)
        self.attention_heads_var = tk.StringVar(value="2")
        ttk.Spinbox(attention_heads_frame, from_=1, to=8, textvariable=self.attention_heads_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Attention Dropout
        attention_dropout_frame = ttk.Frame(self.attention_frame)
        attention_dropout_frame.pack(fill=tk.X, pady=5)
        ttk.Label(attention_dropout_frame, text="Attention Dropout:").pack(side=tk.LEFT)
        self.attention_dropout_var = tk.StringVar(value="0.1")
        ttk.Spinbox(attention_dropout_frame, from_=0.0, to=0.5, increment=0.1, textvariable=self.attention_dropout_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Show/hide attention parameters based on model type
        self.model_type_var.trace_add("write", self.toggle_attention_params)
        
        # Learning Parameters frame
        learning_frame = ttk.LabelFrame(right_frame, text="Learning Parameters")
        learning_frame.pack(fill=tk.X, pady=10)
        
        # Learning Rate
        lr_frame = ttk.Frame(learning_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.learning_rate_var = tk.StringVar(value="0.01")
        ttk.Spinbox(lr_frame, from_=0.001, to=0.1, increment=0.001, textvariable=self.learning_rate_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Iterations
        iter_frame = ttk.Frame(learning_frame)
        iter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iterations_var = tk.StringVar(value="500")
        ttk.Spinbox(iter_frame, from_=100, to=5000, increment=100, textvariable=self.iterations_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Tokenizer Type
        tokenizer_frame = ttk.Frame(learning_frame)
        tokenizer_frame.pack(fill=tk.X, pady=5)
        ttk.Label(tokenizer_frame, text="Tokenizer:").pack(side=tk.LEFT)
        self.tokenizer_var = tk.StringVar(value="wordpiece")
        ttk.Radiobutton(tokenizer_frame, text="WordPiece", variable=self.tokenizer_var, value="wordpiece").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Radiobutton(tokenizer_frame, text="BPE", variable=self.tokenizer_var, value="bpe").pack(side=tk.LEFT, padx=(5, 0))
        
        # Use Pretrained Embeddings
        pretrained_frame = ttk.Frame(learning_frame)
        pretrained_frame.pack(fill=tk.X, pady=5)
        self.pretrained_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pretrained_frame, text="Use Pretrained Embeddings", variable=self.pretrained_var).pack(side=tk.LEFT)
        
        # Action Buttons frame
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        # Create two rows for buttons to ensure they fit
        button_row1 = ttk.Frame(action_frame)
        button_row1.pack(fill=tk.X, pady=(0, 5))
        
        button_row2 = ttk.Frame(action_frame)
        button_row2.pack(fill=tk.X)
        
        # Train Button
        self.train_button = ttk.Button(button_row1, text="Train Model", command=self.train_model)
        self.train_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Stop Button
        self.stop_button = ttk.Button(button_row1, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Save Button
        self.save_button = ttk.Button(button_row2, text="Save Model", command=self.save_model, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Load Button
        self.load_button = ttk.Button(button_row2, text="Load Model", command=self.load_model)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Text Input frame
        text_frame = ttk.LabelFrame(left_frame, text="Training Text")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Text Input
        self.text_input = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=40, height=10)
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sample Text Button
        self.sample_button = ttk.Button(text_frame, text="Load Sample Text", command=self.load_sample_text)
        self.sample_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Clear Text Button
        self.clear_button = ttk.Button(text_frame, text="Clear Text", command=self.clear_text)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Load Text File Button
        self.load_text_button = ttk.Button(text_frame, text="Load Text File", command=self.load_text_file)
        self.load_text_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Training Progress frame
        progress_frame = ttk.LabelFrame(left_frame, text="Training Progress")
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress Bar
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Loss Plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training and Validation Loss')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=progress_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_prediction_tab(self):
        """
        Set up the prediction tab.
        """
        # Create frames
        left_frame = ttk.Frame(self.prediction_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(self.prediction_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Context Input frame
        context_frame = ttk.LabelFrame(left_frame, text="Context")
        context_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Context Input
        self.context_input = ttk.Entry(context_frame, width=40)
        self.context_input.pack(fill=tk.X, padx=5, pady=5)
        
        # Predict Button
        self.predict_button = ttk.Button(context_frame, text="Predict Next Word", command=self.predict_next_word, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Clear Button
        self.clear_context_button = ttk.Button(context_frame, text="Clear", command=self.clear_context)
        self.clear_context_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Prediction Results frame
        results_frame = ttk.LabelFrame(left_frame, text="Prediction Results")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results Text
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=40, height=10, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(right_frame, text="Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Probability Plot
        self.prob_fig, self.prob_ax = plt.subplots(figsize=(6, 4))
        self.prob_ax.set_title('Word Probabilities')
        self.prob_ax.set_ylabel('Probability')
        self.prob_ax.grid(True)
        
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=viz_frame)
        self.prob_canvas.draw()
        self.prob_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Attention Visualization frame (only visible for attention models)
        self.attention_viz_frame = ttk.LabelFrame(right_frame, text="Attention Weights")
        
        # Attention Plot
        self.attn_fig, self.attn_ax = plt.subplots(figsize=(6, 4))
        self.attn_ax.set_title('Attention Weights')
        
        self.attn_canvas = FigureCanvasTkAgg(self.attn_fig, master=self.attention_viz_frame)
        self.attn_canvas.draw()
        self.attn_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_generation_tab(self):
        """
        Set up the generation tab.
        """
        # Create frames
        left_frame = ttk.Frame(self.generation_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(self.generation_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generation Parameters frame
        params_frame = ttk.LabelFrame(left_frame, text="Generation Parameters")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Starting Context
        context_frame = ttk.Frame(params_frame)
        context_frame.pack(fill=tk.X, pady=5)
        ttk.Label(context_frame, text="Starting Context:").pack(side=tk.LEFT)
        self.gen_context_input = ttk.Entry(context_frame, width=30)
        self.gen_context_input.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Number of Words
        words_frame = ttk.Frame(params_frame)
        words_frame.pack(fill=tk.X, pady=5)
        ttk.Label(words_frame, text="Number of Words:").pack(side=tk.LEFT)
        self.num_words_var = tk.StringVar(value="20")
        ttk.Spinbox(words_frame, from_=1, to=100, textvariable=self.num_words_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Temperature
        temp_frame = ttk.Frame(params_frame)
        temp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temperature_var = tk.StringVar(value="1.0")
        ttk.Spinbox(temp_frame, from_=0.1, to=2.0, increment=0.1, textvariable=self.temperature_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(temp_frame, text="(higher = more random)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Generate Button
        self.generate_button = ttk.Button(params_frame, text="Generate Text", command=self.generate_text, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Clear Button
        self.clear_gen_button = ttk.Button(params_frame, text="Clear", command=self.clear_generation)
        self.clear_gen_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Generated Text frame
        text_frame = ttk.LabelFrame(left_frame, text="Generated Text")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Generated Text
        self.generated_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=40, height=10, state=tk.DISABLED)
        self.generated_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Generation Info frame
        info_frame = ttk.LabelFrame(right_frame, text="Generation Information")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Generation Info Text
        self.gen_info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, width=40, height=20, state=tk.DISABLED)
        self.gen_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def toggle_attention_params(self, *args):
        """
        Show or hide attention parameters based on model type.
        """
        if self.model_type_var.get() == "attention":
            # Simply pack after the hidden layers frame
            self.attention_frame.pack(fill=tk.X, pady=5, after=self.hidden_frame)
            if hasattr(self, 'attention_viz_frame'):
                self.attention_viz_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.attention_frame.pack_forget()
            if hasattr(self, 'attention_viz_frame'):
                self.attention_viz_frame.pack_forget()
    
    def load_sample_text(self):
        """
        Load sample text into the text input.
        """
        sample_text = """
        The self-attention mechanism has revolutionized natural language processing.
        Attention allows models to focus on relevant parts of the input sequence.
        This mechanism is a key component of transformer architectures.
        Language models using attention can generate more coherent and contextually appropriate text.
        Self-attention computes a weighted sum of all words in the sequence.
        The weights are determined by the compatibility of query and key vectors.
        Attention weights can be visualized to interpret model decisions.
        Multi-head attention allows the model to focus on different aspects of the input.
        Transformers use self-attention instead of recurrence or convolution.
        Attention is all you need was the title of the paper that introduced transformers.
        Word embeddings capture semantic relationships between words.
        Contextual word embeddings depend on the surrounding words.
        Pre-trained language models can be fine-tuned for specific tasks.
        Transfer learning leverages knowledge from one task to improve another.
        Neural networks learn hierarchical representations of data.
        Deep learning models require large amounts of training data.
        Gradient descent optimizes model parameters to minimize loss.
        Backpropagation computes gradients through the computational graph.
        Regularization techniques prevent overfitting to training data.
        Dropout randomly deactivates neurons during training.
        Layer normalization stabilizes the training of deep networks.
        Residual connections help gradient flow in deep networks.
        """
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(tk.END, sample_text)
    
    def clear_text(self):
        """
        Clear the text input.
        """
        self.text_input.delete(1.0, tk.END)
    
    def load_text_file(self):
        """
        Load text from a file.
        """
        filepath = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.text_input.delete(1.0, tk.END)
            self.text_input.insert(tk.END, text)
            
            self.status_var.set(f"Loaded text from {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load text file: {str(e)}")
    
    def train_model(self):
        """
        Train the model on the input text.
        """
        # Get text
        text = self.text_input.get(1.0, tk.END)
        
        if not text.strip():
            messagebox.showerror("Error", "Please enter some text for training.")
            return
        
        # Get parameters
        try:
            context_size = int(self.context_size_var.get())
            embedding_dim = int(self.embedding_dim_var.get())
            hidden_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',')]
            learning_rate = float(self.learning_rate_var.get())
            iterations = int(self.iterations_var.get())
            tokenizer_type = self.tokenizer_var.get()
            use_pretrained = self.pretrained_var.get()
            
            # Attention parameters
            if self.model_type_var.get() == "attention":
                attention_dim = int(self.attention_dim_var.get())
                num_attention_heads = int(self.attention_heads_var.get())
                attention_dropout = float(self.attention_dropout_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {str(e)}")
            return
        
        # Import the necessary classes
        from ..models.multi_layer_perceptron import MultiLayerPerceptron
        from ..models.attention_perceptron import AttentionPerceptron
        
        # Create model
        if self.model_type_var.get() == "attention":
            self.model = AttentionPerceptron(
                context_size=context_size,
                embedding_dim=embedding_dim,
                hidden_layers=hidden_layers,
                attention_dim=attention_dim,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                learning_rate=learning_rate,
                n_iterations=iterations,
                tokenizer_type=tokenizer_type,
                use_pretrained=use_pretrained
            )
            self.model_type_info_var.set("Attention Perceptron")
        else:
            self.model = MultiLayerPerceptron(
                context_size=context_size,
                embedding_dim=embedding_dim,
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                n_iterations=iterations,
                tokenizer_type=tokenizer_type,
                use_pretrained=use_pretrained
            )
            self.model_type_info_var.set("Multi-Layer Perceptron")
        
        # Reset progress
        self.progress_bar['value'] = 0
        self.ax.clear()
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training and Validation Loss')
        self.ax.grid(True)
        self.canvas.draw()
        
        # Update UI
        self.model_status_var.set("Training...")
        self.train_button['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL
        self.save_button['state'] = tk.DISABLED
        self.predict_button['state'] = tk.DISABLED
        self.generate_button['state'] = tk.DISABLED
        
        # Reset stop event
        self.stop_training_event.clear()
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(text,)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def train_model_thread(self, text):
        """
        Train the model in a separate thread.
        
        Parameters:
        -----------
        text : str
            Text to train on
        """
        try:
            # Train the model
            self.model.fit(text, progress_callback=self.update_progress, stop_event=self.stop_training_event)
            
            # Update UI
            if not self.stop_training_event.is_set():
                self.progress_queue.put(("training_complete", None))
        except Exception as e:
            self.progress_queue.put(("error", str(e)))
    
    def update_progress(self, iteration, total, train_loss, val_loss, message=None):
        """
        Update training progress.
        
        Parameters:
        -----------
        iteration : int
            Current iteration
        total : int
            Total iterations
        train_loss : float
            Training loss
        val_loss : float
            Validation loss
        message : str, optional
            Additional message
        """
        self.progress_queue.put(("progress", (iteration, total, train_loss, val_loss, message)))
    
    def check_progress_queue(self):
        """
        Check the progress queue for updates.
        """
        try:
            while True:
                message_type, data = self.progress_queue.get_nowait()
                
                if message_type == "progress":
                    iteration, total, train_loss, val_loss, message = data
                    
                    # Update progress bar
                    progress = int(100 * iteration / total)
                    self.progress_bar['value'] = progress
                    
                    # Update progress text
                    self.progress_var.set(f"{iteration}/{total} ({progress}%)")
                    
                    # Update loss text
                    self.loss_var.set(f"{train_loss:.4f}")
                    self.val_loss_var.set(f"{val_loss:.4f}")
                    
                    # Update status
                    if message:
                        self.status_var.set(message)
                    else:
                        self.status_var.set(f"Training: iteration {iteration}/{total}")
                    
                    # Update plot if we have data
                    if hasattr(self.model, 'training_loss') and len(self.model.training_loss) > 0:
                        self.ax.clear()
                        self.ax.plot(self.model.iteration_count, self.model.training_loss, 'b-', label='Training Loss')
                        self.ax.plot(self.model.iteration_count, self.model.validation_loss, 'r-', label='Validation Loss')
                        self.ax.set_xlabel('Iteration')
                        self.ax.set_ylabel('Loss')
                        self.ax.set_title('Training and Validation Loss')
                        self.ax.legend()
                        self.ax.grid(True)
                        self.canvas.draw()
                
                elif message_type == "training_complete":
                    # Update UI
                    self.model_status_var.set("Trained")
                    self.train_button['state'] = tk.NORMAL
                    self.stop_button['state'] = tk.DISABLED
                    self.save_button['state'] = tk.NORMAL
                    self.predict_button['state'] = tk.NORMAL
                    self.generate_button['state'] = tk.NORMAL
                    
                    # Update vocabulary size
                    if hasattr(self.model, 'vocabulary') and self.model.vocabulary:
                        self.vocab_size_var.set(str(len(self.model.vocabulary)))
                    
                    # Update status
                    self.status_var.set("Training complete")
                    
                    # Show message
                    messagebox.showinfo("Training Complete", "Model training has completed successfully.")
                
                elif message_type == "error":
                    # Update UI
                    self.model_status_var.set("Error")
                    self.train_button['state'] = tk.NORMAL
                    self.stop_button['state'] = tk.DISABLED
                    
                    # Update status
                    self.status_var.set(f"Error: {data}")
                    
                    # Show error
                    messagebox.showerror("Training Error", f"An error occurred during training: {data}")
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_progress_queue)
    
    def stop_training(self):
        """
        Stop the training process.
        """
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training_event.set()
            self.status_var.set("Stopping training...")
            self.stop_button['state'] = tk.DISABLED
    
    def save_model(self):
        """
        Save the trained model to a file.
        """
        if not self.model:
            messagebox.showerror("Error", "No model to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.model.save_model(filepath)
            self.status_var.set(f"Model saved to {filepath}")
            messagebox.showinfo("Save Model", f"Model saved successfully to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """
        Load a trained model from a file.
        """
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Try to determine model type from file
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if it's an attention model
            is_attention = 'attention_dim' in model_data
            
            # Import the necessary classes
            from ..models.multi_layer_perceptron import MultiLayerPerceptron
            from ..models.attention_perceptron import AttentionPerceptron
            
            # Load the appropriate model type
            if is_attention:
                self.model = AttentionPerceptron.load_model(filepath)
                self.model_type_info_var.set("Attention Perceptron")
                self.model_type_var.set("attention")
            else:
                self.model = MultiLayerPerceptron.load_model(filepath)
                self.model_type_info_var.set("Multi-Layer Perceptron")
                self.model_type_var.set("mlp")
            
            # Update UI
            self.model_status_var.set("Loaded")
            self.save_button['state'] = tk.NORMAL
            self.predict_button['state'] = tk.NORMAL
            self.generate_button['state'] = tk.NORMAL
            
            # Update parameters
            self.context_size_var.set(str(self.model.context_size))
            self.embedding_dim_var.set(str(self.model.embedding_dim))
            self.hidden_layers_var.set(','.join(map(str, self.model.hidden_layers)))
            self.learning_rate_var.set(str(self.model.learning_rate))
            self.iterations_var.set(str(self.model.n_iterations))
            
            if hasattr(self.model, 'tokenizer_type'):
                self.tokenizer_var.set(self.model.tokenizer_type)
            
            if hasattr(self.model, 'use_pretrained'):
                self.pretrained_var.set(self.model.use_pretrained)
            
            # Update attention parameters if applicable
            if is_attention:
                self.attention_dim_var.set(str(self.model.attention_dim))
                self.attention_heads_var.set(str(self.model.num_attention_heads))
                self.attention_dropout_var.set(str(self.model.attention_dropout))
                # Show attention parameters
                self.toggle_attention_params()
            
            # Update vocabulary size
            if hasattr(self.model, 'vocabulary') and self.model.vocabulary:
                self.vocab_size_var.set(str(len(self.model.vocabulary)))
            
            # Update plot if we have data
            if hasattr(self.model, 'training_loss') and len(self.model.training_loss) > 0:
                self.ax.clear()
                self.ax.plot(self.model.iteration_count, self.model.training_loss, 'b-', label='Training Loss')
                self.ax.plot(self.model.iteration_count, self.model.validation_loss, 'r-', label='Validation Loss')
                self.ax.set_xlabel('Iteration')
                self.ax.set_ylabel('Loss')
                self.ax.set_title('Training and Validation Loss')
                self.ax.legend()
                self.ax.grid(True)
                self.canvas.draw()
            
            self.status_var.set(f"Model loaded from {filepath}")
            messagebox.showinfo("Load Model", f"Model loaded successfully from {filepath}")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error loading model: {error_details}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def clear_context(self):
        """
        Clear the context input.
        """
        self.context_input.delete(0, tk.END)
    
    def predict_next_word(self):
        """
        Predict the next word given the context.
        """
        if not self.model:
            messagebox.showerror("Error", "No model loaded.")
            return
        
        context = self.context_input.get()
        
        if not context.strip():
            messagebox.showerror("Error", "Please enter a context.")
            return
        
        try:
            # Predict next word
            word, info = self.model.predict_next_word(context)
            
            # Update results text
            self.results_text['state'] = tk.NORMAL
            self.results_text.delete(1.0, tk.END)
            
            self.results_text.insert(tk.END, f"Context: {context}\n\n")
            self.results_text.insert(tk.END, f"Predicted next word: {word}\n\n")
            
            if "adjustment_made" in info and info["adjustment_made"]:
                self.results_text.insert(tk.END, "Adjustments made:\n")
                if "adjustment_type" in info:
                    self.results_text.insert(tk.END, f"- {info['adjustment_type']}\n")
                if "unknown_words" in info:
                    self.results_text.insert(tk.END, f"- Unknown words: {', '.join(info['unknown_words'])}\n")
                self.results_text.insert(tk.END, "\n")
            
            if "probabilities" in info:
                self.results_text.insert(tk.END, "Top predictions:\n")
                for word, prob in sorted(info["probabilities"].items(), key=lambda x: x[1], reverse=True):
                    self.results_text.insert(tk.END, f"- {word}: {prob:.4f}\n")
            
            self.results_text['state'] = tk.DISABLED
            
            # Update probability plot
            self.prob_ax.clear()
            
            if "probabilities" in info:
                words = []
                probs = []
                
                for word, prob in sorted(info["probabilities"].items(), key=lambda x: x[1], reverse=True):
                    words.append(word)
                    probs.append(prob)
                
                self.prob_ax.bar(words, probs)
                self.prob_ax.set_title('Word Probabilities')
                self.prob_ax.set_ylabel('Probability')
                self.prob_ax.tick_params(axis='x', rotation=45)
                self.prob_ax.grid(True)
                self.prob_fig.tight_layout()
                self.prob_canvas.draw()
            
            # Import the necessary classes
            from ..models.attention_perceptron import AttentionPerceptron
            
            # Update attention visualization if applicable
            if isinstance(self.model, AttentionPerceptron) and "attention_weights" in info:
                self.attention_viz_frame.pack(fill=tk.BOTH, expand=True)
                
                self.attn_ax.clear()
                
                # Get context words
                context_words = context.split()
                if len(context_words) > self.model.context_size:
                    context_words = context_words[-self.model.context_size:]
                elif len(context_words) < self.model.context_size:
                    context_words = ["<PAD>"] * (self.model.context_size - len(context_words)) + context_words
                
                # Plot attention weights
                attention_weights = np.array(info["attention_weights"])
                im = self.attn_ax.imshow(attention_weights, cmap="YlOrRd")
                self.attn_fig.colorbar(im, ax=self.attn_ax)
                
                self.attn_ax.set_xticks(np.arange(len(context_words)))
                self.attn_ax.set_yticks(np.arange(len(context_words)))
                self.attn_ax.set_xticklabels(context_words)
                self.attn_ax.set_yticklabels(context_words)
                
                plt.setp(self.attn_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                self.attn_ax.set_title(f"Attention Weights (Predicted: '{word}')")
                
                for i in range(len(context_words)):
                    for j in range(len(context_words)):
                        self.attn_ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                                        ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white")
                
                self.attn_ax.set_xlabel("Key Words")
                self.attn_ax.set_ylabel("Query Words")
                
                self.attn_fig.tight_layout()
                self.attn_canvas.draw()
            
            self.status_var.set(f"Predicted next word: {word}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict next word: {str(e)}")
    
    def clear_generation(self):
        """
        Clear the generation inputs and outputs.
        """
        self.gen_context_input.delete(0, tk.END)
        self.generated_text['state'] = tk.NORMAL
        self.generated_text.delete(1.0, tk.END)
        self.generated_text['state'] = tk.DISABLED
        self.gen_info_text['state'] = tk.NORMAL
        self.gen_info_text.delete(1.0, tk.END)
        self.gen_info_text['state'] = tk.DISABLED
    
    def generate_text(self):
        """
        Generate text given a starting context.
        """
        if not self.model:
            messagebox.showerror("Error", "No model loaded.")
            return
        
        context = self.gen_context_input.get()
        
        if not context.strip():
            messagebox.showerror("Error", "Please enter a starting context.")
            return
        
        try:
            # Get parameters
            num_words = int(self.num_words_var.get())
            temperature = float(self.temperature_var.get())
            
            # Generate text
            words, info = self.model.predict_next_n_words(context, n=num_words, temperature=temperature)
            
            # Update generated text
            self.generated_text['state'] = tk.NORMAL
            self.generated_text.delete(1.0, tk.END)
            
            full_text = f"{context} {' '.join(words)}"
            self.generated_text.insert(tk.END, full_text)
            
            self.generated_text['state'] = tk.DISABLED
            
            # Update generation info
            self.gen_info_text['state'] = tk.NORMAL
            self.gen_info_text.delete(1.0, tk.END)
            
            self.gen_info_text.insert(tk.END, f"Starting context: {context}\n")
            self.gen_info_text.insert(tk.END, f"Number of words: {num_words}\n")
            self.gen_info_text.insert(tk.END, f"Temperature: {temperature}\n\n")
            
            if "adjusted_context" in info:
                self.gen_info_text.insert(tk.END, f"Adjusted context: {' '.join(info['adjusted_context'])}\n\n")
            
            if "predictions" in info:
                self.gen_info_text.insert(tk.END, "Generation steps:\n")
                for i, pred_info in enumerate(info["predictions"]):
                    self.gen_info_text.insert(tk.END, f"Step {i+1}:\n")
                    self.gen_info_text.insert(tk.END, f"  Context: {' '.join(pred_info['context'])}\n")
                    self.gen_info_text.insert(tk.END, f"  Predicted: {pred_info['predicted_word']}\n")
                    
                    if "probabilities" in pred_info:
                        top_words = sorted(pred_info["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
                        self.gen_info_text.insert(tk.END, f"  Top alternatives: {', '.join([f'{w} ({p:.4f})' for w, p in top_words])}\n")
                    
                    self.gen_info_text.insert(tk.END, "\n")
            
            if "error" in info:
                self.gen_info_text.insert(tk.END, f"Error at step {info.get('error_step', '?')}:\n{info['error']}\n")
            
            self.gen_info_text['state'] = tk.DISABLED
            
            self.status_var.set(f"Generated {len(words)} words")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text: {str(e)}")

def main():
    """
    Main function to run the UI.
    """
    root = tk.Tk()
    app = CompleteMlpUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()