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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class SimpleLanguageModel:
    """
    A simple language model using a single-layer perceptron for next word prediction.
    """
    
    def __init__(self, context_size=2, learning_rate=0.01, n_iterations=1000, random_state=42):
        """
        Initialize the simple language model.
        """
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.vocabulary = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.encoder = None
        self.weights = None
        self.bias = None
        self.input_size = None
        self.output_size = None
        self.training_loss = []
        self.validation_loss = []
        self.iteration_count = []
        
    def _preprocess_text(self, text):
        """
        Preprocess the text by converting to lowercase, handling special characters,
        and splitting into words.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace contractions with full forms to handle apostrophes better
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "we'll": "we will",
            "they'll": "they will",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "ain't": "am not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters and replace with space
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        
        return words
    
    def _build_vocabulary(self, words):
        """
        Build the vocabulary from the list of words.
        """
        # Get unique words
        unique_words = sorted(set(words))
        
        # Create word to index and index to word mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
        
        # Store vocabulary
        self.vocabulary = unique_words
        self.output_size = len(unique_words)
        
        # Create one-hot encoder for words
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.array(range(len(unique_words))).reshape(-1, 1))
    
    def _create_training_data(self, words):
        """
        Create training data from the list of words.
        Each input is context_size words, and the target is the next word.
        """
        X = []
        y = []
        
        for i in range(len(words) - self.context_size):
            # Get context words
            context = words[i:i+self.context_size]
            
            # Get target word
            target = words[i+self.context_size]
            
            # Convert words to indices
            context_indices = [self.word_to_idx[word] for word in context]
            target_index = self.word_to_idx[target]
            
            # Convert context indices to one-hot vectors and flatten
            context_vectors = [self.encoder.transform([[idx]])[0] for idx in context_indices]
            context_vector = np.concatenate(context_vectors)
            
            X.append(context_vector)
            y.append(target_index)
        
        return np.array(X), np.array(y)
    
    def _softmax(self, x):
        """
        Compute softmax values for each set of scores in x.
        """
        # Subtract max for numerical stability
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def _forward(self, X):
        """
        Forward pass through the network.
        """
        # Calculate the net input
        z = np.dot(X, self.weights) + self.bias
        
        # Apply softmax activation
        a = self._softmax(z)
        
        return a
    
    def _cross_entropy_loss(self, y_pred, y_true):
        """
        Calculate the cross-entropy loss.
        """
        # Convert y_true to one-hot encoding
        y_true_one_hot = self.encoder.transform(y_true.reshape(-1, 1))
        
        # Calculate cross-entropy loss
        loss = -np.sum(y_true_one_hot * np.log(y_pred + 1e-10)) / len(y_true)
        
        return loss
    
    def fit(self, text, progress_callback=None, stop_event=None):
        """
        Train the language model on the given text.
        
        Parameters:
        -----------
        text : str
            The text to train on
        progress_callback : function
            Callback function to report progress
        stop_event : threading.Event
            Event to signal stopping the training
        """
        # Preprocess the text
        words = self._preprocess_text(text)
        
        # Build vocabulary
        self._build_vocabulary(words)
        
        # Create training data
        X, y = self._create_training_data(words)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Initialize weights and bias
        np.random.seed(self.random_state)
        self.input_size = X.shape[1]
        # Initialize with slightly larger weights for better gradient flow
        self.weights = np.random.normal(0, 0.2, (self.input_size, self.output_size))
        self.bias = np.zeros(self.output_size)
        
        # Adaptive learning rate
        initial_learning_rate = self.learning_rate
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 50  # Number of iterations to wait for improvement
        patience_counter = 0
        
        # Clear previous training data
        self.training_loss = []
        self.validation_loss = []
        self.iteration_count = []
        
        for iteration in range(self.n_iterations):
            # Check if training should be stopped
            if stop_event and stop_event.is_set():
                break
                
            # Adaptive learning rate - decrease over time
            current_lr = initial_learning_rate / (1 + iteration / 200)
            
            # Forward pass
            y_pred = self._forward(X_train)
            
            # Calculate loss
            loss = self._cross_entropy_loss(y_pred, y_train)
            
            # Calculate validation loss
            y_val_pred = self._forward(X_val)
            val_loss = self._cross_entropy_loss(y_val_pred, y_val)
            
            # Store losses
            self.training_loss.append(loss)
            self.validation_loss.append(val_loss)
            self.iteration_count.append(iteration)
            
            # Report progress
            if iteration % 10 == 0 and progress_callback:
                progress_callback(iteration, self.n_iterations, loss, val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback(iteration, self.n_iterations, loss, val_loss, 
                                     f"Early stopping at iteration {iteration}")
                # Restore best weights
                self.weights = best_weights
                self.bias = best_bias
                break
            
            # Backward pass (gradient descent)
            # Convert y_train to one-hot encoding
            y_train_one_hot = self.encoder.transform(y_train.reshape(-1, 1))
            
            # Calculate gradients
            dZ = y_pred - y_train_one_hot
            dW = np.dot(X_train.T, dZ) / len(X_train)
            db = np.sum(dZ, axis=0) / len(X_train)
            
            # Add L2 regularization to prevent overfitting
            reg_lambda = 0.001
            dW += reg_lambda * self.weights
            
            # Update weights and bias with current learning rate
            self.weights -= current_lr * dW
            self.bias -= current_lr * db
            
            # Shuffle the training data every 50 iterations
            if iteration % 50 == 0:
                shuffle_idx = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
        
        # Final progress report
        if progress_callback:
            progress_callback(iteration + 1, self.n_iterations, loss, val_loss, "Training complete")
    
    def predict_next_word(self, context):
        """
        Predict the next word given a context.
        
        Parameters:
        -----------
        context : list of str
            List of context words
        
        Returns:
        --------
        str
            Predicted next word
        """
        # Check if context has the right length
        if len(context) != self.context_size:
            raise ValueError(f"Context must have {self.context_size} words")
        
        # Preprocess context
        context = [word.lower() for word in context]
        
        # Check if all words are in vocabulary
        for word in context:
            if word not in self.word_to_idx:
                raise ValueError(f"Word '{word}' not in vocabulary")
        
        # Convert context to indices
        context_indices = [self.word_to_idx[word] for word in context]
        
        # Convert context indices to one-hot vectors and flatten
        context_vectors = [self.encoder.transform([[idx]])[0] for idx in context_indices]
        context_vector = np.concatenate(context_vectors)
        
        # Forward pass
        y_pred = self._forward(context_vector.reshape(1, -1))[0]
        
        # Get the word with the highest probability
        predicted_idx = np.argmax(y_pred)
        predicted_word = self.idx_to_word[predicted_idx]
        
        return predicted_word
    
    def predict_next_n_words(self, initial_context, n=5):
        """
        Predict the next n words given an initial context.
        
        Parameters:
        -----------
        initial_context : list of str
            Initial context words
        n : int
            Number of words to predict
        
        Returns:
        --------
        list of str
            Predicted words
        """
        # Check if initial context has the right length
        if len(initial_context) != self.context_size:
            raise ValueError(f"Initial context must have {self.context_size} words")
        
        # Make a copy of the initial context
        context = initial_context.copy()
        
        # Predict n words
        predicted_words = []
        for _ in range(n):
            # Predict next word
            next_word = self.predict_next_word(context)
            predicted_words.append(next_word)
            
            # Update context
            context = context[1:] + [next_word]
        
        return predicted_words
    
    def get_top_predictions(self, context, top_n=5):
        """
        Get the top n predicted words with their probabilities.
        
        Parameters:
        -----------
        context : list of str
            List of context words
        top_n : int
            Number of top predictions to return
        
        Returns:
        --------
        list of tuples
            List of (word, probability) tuples
        """
        # Check if context has the right length
        if len(context) != self.context_size:
            raise ValueError(f"Context must have {self.context_size} words")
        
        # Preprocess context
        context = [word.lower() for word in context]
        
        # Check if all words are in vocabulary
        for word in context:
            if word not in self.word_to_idx:
                raise ValueError(f"Word '{word}' not in vocabulary")
        
        # Convert context to indices
        context_indices = [self.word_to_idx[word] for word in context]
        
        # Convert context indices to one-hot vectors and flatten
        context_vectors = [self.encoder.transform([[idx]])[0] for idx in context_indices]
        context_vector = np.concatenate(context_vectors)
        
        # Forward pass
        y_pred = self._forward(context_vector.reshape(1, -1))[0]
        
        # Get the top n predictions
        top_indices = np.argsort(y_pred)[-top_n:][::-1]
        top_probs = y_pred[top_indices]
        
        # Convert to words
        top_words = [self.idx_to_word[idx] for idx in top_indices]
        
        return list(zip(top_words, top_probs))
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        # Calculate the encoder dimension for future reference
        encoder_dim = 0
        if len(self.vocabulary) > 0:
            sample_idx = 0
            sample_encoded = self.encoder.transform([[sample_idx]])[0]
            encoder_dim = len(sample_encoded)
        
        model_data = {
            'context_size': self.context_size,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'random_state': self.random_state,
            'vocabulary': self.vocabulary,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'weights': self.weights,
            'bias': self.bias,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'encoder_dim': encoder_dim,  # Save encoder dimension for reference
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'iteration_count': self.iteration_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
        
        Returns:
        --------
        SimpleLanguageModel
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(
            context_size=model_data['context_size'],
            learning_rate=model_data['learning_rate'],
            n_iterations=model_data['n_iterations'],
            random_state=model_data['random_state']
        )
        
        # Restore model attributes
        model.vocabulary = model_data['vocabulary']
        model.word_to_idx = model_data['word_to_idx']
        model.idx_to_word = model_data['idx_to_word']
        model.weights = model_data['weights']
        model.bias = model_data['bias']
        model.input_size = model_data['input_size']
        model.output_size = model_data['output_size']
        model.training_loss = model_data.get('training_loss', [])
        model.validation_loss = model_data.get('validation_loss', [])
        model.iteration_count = model_data.get('iteration_count', [])
        
        # Recreate the encoder with the correct shape
        model.encoder = OneHotEncoder(sparse_output=False)
        model.encoder.fit(np.array(range(len(model.vocabulary))).reshape(-1, 1))
        
        # Verify that the encoder output dimension matches the expected input size
        # This is to ensure compatibility with the model weights
        sample_idx = 0
        if len(model.vocabulary) > 0:
            sample_encoded = model.encoder.transform([[sample_idx]])[0]
            context_size = model_data['context_size']
            expected_input_size = len(sample_encoded) * context_size
            
            # Check if we have saved encoder dimension
            if 'encoder_dim' in model_data and model_data['encoder_dim'] > 0:
                saved_input_size = model_data['encoder_dim'] * context_size
                
                # If there's a mismatch between current encoder and saved encoder dimensions
                if saved_input_size != expected_input_size:
                    print(f"Warning: Current encoder dimension ({len(sample_encoded)}) doesn't match saved encoder dimension ({model_data['encoder_dim']}).")
                    print("This might cause issues when making predictions.")
            
            # Check if encoder output matches the expected input size for the model weights
            if expected_input_size != model.input_size:
                print(f"Warning: Encoder output dimension mismatch. Expected {model.input_size}, got {expected_input_size}.")
                print("This might cause issues when making predictions.")
        
        return model


class LanguageModelUI:
    """
    GUI for the Simple Language Model.
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
        self.root.title("Simple Language Model UI")
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
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
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
        params_frame = ttk.LabelFrame(left_frame, text="Model Parameters")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Context size
        context_frame = ttk.Frame(params_frame)
        context_frame.pack(fill=tk.X, pady=5)
        ttk.Label(context_frame, text="Context Size:").pack(side=tk.LEFT)
        self.context_size_var = tk.IntVar(value=2)
        context_size_entry = ttk.Spinbox(context_frame, from_=1, to=5, textvariable=self.context_size_var, width=5)
        context_size_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(context_frame, text="(Number of previous words to use as context)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Learning rate
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        learning_rate_entry = ttk.Spinbox(lr_frame, from_=0.001, to=1.0, increment=0.01, textvariable=self.learning_rate_var, width=5)
        learning_rate_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(lr_frame, text="(Step size for weight updates)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Iterations
        iter_frame = ttk.Frame(params_frame)
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
        ttk.Label(top_frame, text="Number of predictions:").pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        self.num_predictions_var = tk.IntVar(value=5)
        num_predictions_entry = ttk.Spinbox(top_frame, from_=1, to=20, textvariable=self.num_predictions_var, width=3)
        num_predictions_entry.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Middle frame - Results
        middle_frame = ttk.LabelFrame(self.prediction_tab, text="Prediction Results")
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Results text
        self.results_text = scrolledtext.ScrolledText(middle_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Bottom frame - Visualization
        bottom_frame = ttk.LabelFrame(self.prediction_tab, text="Probability Distribution")
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Prediction plot
        self.pred_fig, self.pred_ax = plt.subplots(figsize=(5, 4))
        self.pred_ax.set_title("Word Probabilities")
        self.pred_ax.set_xlabel("Word")
        self.pred_ax.set_ylabel("Probability")
        self.pred_ax.grid(True)
        
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=bottom_frame)
        self.pred_canvas.draw()
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
    
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
        
        # Generate button
        self.generate_button = ttk.Button(top_frame, text="Generate Text", command=self.generate_text, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Number of words
        ttk.Label(top_frame, text="Number of words:").pack(side=tk.LEFT, padx=(10, 0), pady=(5, 0))
        self.num_words_var = tk.IntVar(value=20)
        num_words_entry = ttk.Spinbox(top_frame, from_=1, to=100, textvariable=self.num_words_var, width=3)
        num_words_entry.pack(side=tk.LEFT, padx=(5, 0), pady=(5, 0))
        
        # Middle frame - Results
        middle_frame = ttk.LabelFrame(self.generation_tab, text="Generated Text")
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Generated text
        self.generated_text = scrolledtext.ScrolledText(middle_frame, wrap=tk.WORD, height=15)
        self.generated_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Bottom frame - Word cloud
        bottom_frame = ttk.LabelFrame(self.generation_tab, text="Generation Information")
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info text
        self.gen_info_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=8)
        self.gen_info_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def load_sample_text(self):
        """
        Load sample text into the training text area.
        """
        sample_text = """
        The cat chases the mouse.
        The dog chases the cat.
        The mouse runs from the cat.
        The cat runs from the dog.
        Dogs like to play with balls.
        Cats like to play with yarn.
        Mice like to eat cheese.
        Dogs eat meat and treats.
        Cats eat fish and mice.
        The big dog barks loudly.
        The small cat meows softly.
        The tiny mouse squeaks quietly.
        People love dogs and cats.
        Children play with dogs.
        Adults take care of pets.
        Pets make people happy.
        Dogs are loyal animals.
        Cats are independent animals.
        Mice are small animals.
        Animals need food and water.
        Water is essential for life.
        Food provides energy for animals.
        Energy helps animals move.
        Movement is important for health.
        Health is important for all living things.
        
        Artificial intelligence is transforming how we live and work.
        Machine learning models can recognize patterns in data.
        Neural networks are inspired by the human brain.
        Deep learning has revolutionized computer vision.
        Natural language processing helps computers understand text.
        Robots can perform complex tasks autonomously.
        Voice assistants use speech recognition technology.
        Computer vision systems can identify objects in images.
        Reinforcement learning teaches agents through rewards.
        Supervised learning uses labeled training data.
        Unsupervised learning finds patterns without labels.
        Transfer learning applies knowledge from one task to another.
        Generative models can create new content.
        Recommendation systems suggest items users might like.
        Autonomous vehicles use AI to navigate roads.
        Healthcare AI can help diagnose diseases.
        Financial models predict market trends.
        Smart homes use AI to optimize energy use.
        AI ethics considers the impact of artificial intelligence.
        The future of AI depends on responsible development.
        """
        
        self.training_text.delete(1.0, tk.END)
        self.training_text.insert(tk.END, sample_text)
        self.status_var.set("Sample text loaded")
    
    def load_text_from_file(self):
        """
        Load text from a file into the training text area.
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
            
            self.training_text.delete(1.0, tk.END)
            self.training_text.insert(tk.END, text)
            self.status_var.set(f"Text loaded from {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def clear_training_text(self):
        """
        Clear the training text area.
        """
        self.training_text.delete(1.0, tk.END)
        self.status_var.set("Training text cleared")
    
    def train_model(self):
        """
        Train the model on the provided text.
        """
        # Get training text
        text = self.training_text.get(1.0, tk.END)
        
        if not text.strip():
            messagebox.showerror("Error", "Please provide training text")
            return
        
        # Get parameters
        context_size = self.context_size_var.get()
        learning_rate = self.learning_rate_var.get()
        n_iterations = self.iterations_var.get()
        
        # Create model
        self.model = SimpleLanguageModel(
            context_size=context_size,
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )
        
        # Reset stop event
        self.stop_training_event.clear()
        
        # Update UI
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_model_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_label.config(text="Training started...")
        self.model_status_var.set("Training...")
        
        # Clear plot
        self.ax.clear()
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        self.canvas.draw()
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(text,)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def train_model_thread(self, text):
        """
        Thread function for training the model.
        
        Parameters:
        -----------
        text : str
            Training text
        """
        try:
            self.model.fit(text, self.report_progress, self.stop_training_event)
            
            # Update UI after training
            if not self.stop_training_event.is_set():
                self.progress_queue.put(("complete", None))
        except Exception as e:
            self.progress_queue.put(("error", str(e)))
    
    def report_progress(self, iteration, total_iterations, train_loss, val_loss, message=None):
        """
        Report training progress.
        
        Parameters:
        -----------
        iteration : int
            Current iteration
        total_iterations : int
            Total number of iterations
        train_loss : float
            Training loss
        val_loss : float
            Validation loss
        message : str
            Optional message
        """
        progress = (iteration / total_iterations) * 100
        self.progress_queue.put(("progress", (progress, iteration, total_iterations, train_loss, val_loss, message)))
    
    def check_progress_queue(self):
        """
        Check the progress queue for updates from the training thread.
        """
        try:
            while True:
                message_type, data = self.progress_queue.get_nowait()
                
                if message_type == "progress":
                    progress, iteration, total_iterations, train_loss, val_loss, message = data
                    self.progress_var.set(progress)
                    
                    if message:
                        status_text = f"Iteration {iteration}/{total_iterations}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f} - {message}"
                    else:
                        status_text = f"Iteration {iteration}/{total_iterations}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                    
                    self.progress_label.config(text=status_text)
                    self.status_var.set(status_text)
                    
                    # Update plot
                    self.ax.clear()
                    self.ax.plot(self.model.iteration_count, self.model.training_loss, label="Training Loss")
                    self.ax.plot(self.model.iteration_count, self.model.validation_loss, label="Validation Loss")
                    self.ax.set_title("Training Progress")
                    self.ax.set_xlabel("Iteration")
                    self.ax.set_ylabel("Loss")
                    self.ax.legend()
                    self.ax.grid(True)
                    self.canvas.draw()
                
                elif message_type == "complete":
                    self.training_complete()
                
                elif message_type == "error":
                    error_message = data
                    messagebox.showerror("Training Error", f"An error occurred during training: {error_message}")
                    self.stop_training()
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_progress_queue)
    
    def training_complete(self):
        """
        Handle training completion.
        """
        # Update UI
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_model_button.config(state=tk.NORMAL)
        self.predict_button.config(state=tk.NORMAL)
        self.generate_button.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.progress_label.config(text="Training complete")
        self.model_status_var.set("Trained")
        self.status_var.set("Model training complete")
        
        # Update vocabulary size
        if self.model and self.model.vocabulary:
            self.vocab_size_var.set(str(len(self.model.vocabulary)))
    
    def stop_training(self):
        """
        Stop the training process.
        """
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training_event.set()
            self.status_var.set("Stopping training...")
            
            # Update UI
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            if self.model and self.model.vocabulary:
                self.save_model_button.config(state=tk.NORMAL)
                self.predict_button.config(state=tk.NORMAL)
                self.generate_button.config(state=tk.NORMAL)
                self.vocab_size_var.set(str(len(self.model.vocabulary)))
                self.model_status_var.set("Partially trained")
    
    def save_model(self):
        """
        Save the model to a file.
        """
        if not self.model:
            messagebox.showerror("Error", "No model to save")
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
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {e}")
    
    def load_model(self):
        """
        Load a model from a file.
        """
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.model = SimpleLanguageModel.load_model(filepath)
            
            # Update UI
            self.context_size_var.set(self.model.context_size)
            self.learning_rate_var.set(self.model.learning_rate)
            self.iterations_var.set(self.model.n_iterations)
            
            self.save_model_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.NORMAL)
            self.generate_button.config(state=tk.NORMAL)
            
            self.vocab_size_var.set(str(len(self.model.vocabulary)))
            self.model_status_var.set("Loaded")
            self.status_var.set(f"Model loaded from {filepath}")
            
            # Update plot if training history is available
            if hasattr(self.model, 'training_loss') and hasattr(self.model, 'validation_loss') and hasattr(self.model, 'iteration_count'):
                if self.model.training_loss and self.model.validation_loss and self.model.iteration_count:
                    self.ax.clear()
                    self.ax.plot(self.model.iteration_count, self.model.training_loss, label="Training Loss")
                    self.ax.plot(self.model.iteration_count, self.model.validation_loss, label="Validation Loss")
                    self.ax.set_title("Training Progress")
                    self.ax.set_xlabel("Iteration")
                    self.ax.set_ylabel("Loss")
                    self.ax.legend()
                    self.ax.grid(True)
                    self.canvas.draw()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def predict_next_word(self):
        """
        Predict the next word based on the context.
        """
        if not self.model:
            messagebox.showerror("Error", "No model available")
            return
        
        # Get context
        context_text = self.context_entry.get().strip()
        
        if not context_text:
            messagebox.showerror("Error", "Please enter context words")
            return
        
        # Split context into words
        context = context_text.lower().split()
        
        # Check context length
        if len(context) != self.model.context_size:
            messagebox.showerror("Error", f"Context must have exactly {self.model.context_size} words")
            return
        
        try:
            # Get top predictions
            num_predictions = self.num_predictions_var.get()
            top_predictions = self.model.get_top_predictions(context, top_n=num_predictions)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Context: '{' '.join(context)}'\n\n")
            self.results_text.insert(tk.END, f"Top {num_predictions} predictions:\n")
            
            for i, (word, prob) in enumerate(top_predictions, 1):
                self.results_text.insert(tk.END, f"{i}. '{word}' (probability: {prob:.4f})\n")
            
            # Update plot
            self.pred_ax.clear()
            words = [word for word, _ in top_predictions]
            probs = [prob for _, prob in top_predictions]
            
            # Create bar chart
            bars = self.pred_ax.bar(range(len(words)), probs, color='skyblue')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                self.pred_ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{height:.3f}', ha='center', va='bottom', rotation=0, fontsize=8)
            
            self.pred_ax.set_title(f"Top {num_predictions} Word Probabilities")
            self.pred_ax.set_xlabel("Word")
            self.pred_ax.set_ylabel("Probability")
            self.pred_ax.set_xticks(range(len(words)))
            self.pred_ax.set_xticklabels(words, rotation=45, ha='right')
            self.pred_ax.grid(True, axis='y')
            self.pred_fig.tight_layout()
            self.pred_canvas.draw()
            
            self.status_var.set(f"Predicted next word for context: '{' '.join(context)}'")
        
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def generate_text(self):
        """
        Generate text based on the initial context.
        """
        if not self.model:
            messagebox.showerror("Error", "No model available")
            return
        
        # Get context
        context_text = self.gen_context_entry.get().strip()
        
        if not context_text:
            messagebox.showerror("Error", "Please enter initial context words")
            return
        
        # Split context into words
        context = context_text.lower().split()
        
        # Check context length
        if len(context) != self.model.context_size:
            messagebox.showerror("Error", f"Context must have exactly {self.model.context_size} words")
            return
        
        try:
            # Generate text
            num_words = self.num_words_var.get()
            generated_words = self.model.predict_next_n_words(context, n=num_words)
            
            # Display results
            self.generated_text.delete(1.0, tk.END)
            self.generated_text.insert(tk.END, f"Initial context: '{' '.join(context)}'\n\n")
            self.generated_text.insert(tk.END, f"Generated text ({num_words} words):\n")
            self.generated_text.insert(tk.END, f"'{' '.join(context)} {' '.join(generated_words)}'\n")
            
            # Display generation info
            self.gen_info_text.delete(1.0, tk.END)
            self.gen_info_text.insert(tk.END, "Generation Process:\n\n")
            
            current_context = context.copy()
            for i, word in enumerate(generated_words, 1):
                self.gen_info_text.insert(tk.END, f"Step {i}:\n")
                self.gen_info_text.insert(tk.END, f"  Context: '{' '.join(current_context)}'\n")
                
                # Get top 3 predictions for this context
                top_predictions = self.model.get_top_predictions(current_context, top_n=3)
                self.gen_info_text.insert(tk.END, f"  Top 3 predictions:\n")
                for j, (pred_word, prob) in enumerate(top_predictions, 1):
                    if pred_word == word:
                        self.gen_info_text.insert(tk.END, f"    {j}. '{pred_word}' (probability: {prob:.4f}) ← Selected\n")
                    else:
                        self.gen_info_text.insert(tk.END, f"    {j}. '{pred_word}' (probability: {prob:.4f})\n")
                
                # Update context for next step
                current_context = current_context[1:] + [word]
                self.gen_info_text.insert(tk.END, "\n")
            
            self.status_var.set(f"Generated {num_words} words from context: '{' '.join(context)}'")
        
        except ValueError as e:
            messagebox.showerror("Error", str(e))


def main():
    """
    Main function to run the application.
    """
    root = tk.Tk()
    app = LanguageModelUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()