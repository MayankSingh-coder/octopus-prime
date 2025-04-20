#!/usr/bin/env python3
"""
Simplified versions of the perceptron models that don't require matplotlib.
"""

import numpy as np
import re
import pickle
import os
import time
from collections import Counter, defaultdict

class SimpleWordEmbeddings:
    """
    A simplified version of word embeddings that doesn't require external libraries.
    """
    
    def __init__(self, embedding_dim=50, use_pretrained=False):
        """
        Initialize the word embeddings.
        
        Parameters:
        -----------
        embedding_dim : int
            Dimension of the word embeddings
        use_pretrained : bool
            Whether to use pretrained embeddings (not supported in this simplified version)
        """
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.random_state = np.random.RandomState(42)
    
    def get_embedding(self, word):
        """
        Get the embedding for a word.
        
        Parameters:
        -----------
        word : str
            The word to get the embedding for
            
        Returns:
        --------
        numpy.ndarray
            The embedding vector
        """
        if word not in self.embeddings:
            self.embeddings[word] = self.random_state.randn(self.embedding_dim)
        return self.embeddings[word]

class SimpleMultiLayerPerceptron:
    """
    A simplified multi-layer perceptron implementation for language modeling.
    """
    
    def __init__(self, context_size=2, embedding_dim=50, hidden_layers=[64, 32], learning_rate=0.01, n_iterations=1000, batch_size=32, validation_split=0.1, random_state=None, tokenizer_type='simple', vocab_size=5000, use_pretrained=False):
        """
        Initialize the multi-layer perceptron language model.
        
        Parameters:
        -----------
        context_size : int
            Number of previous words to use as context
        embedding_dim : int
            Dimension of word embeddings
        hidden_layers : list
            List of hidden layer sizes
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of training iterations
        batch_size : int
            Batch size for mini-batch gradient descent
        validation_split : float
            Fraction of data to use for validation
        random_state : int or None
            Random seed for reproducibility
        tokenizer_type : str
            Type of tokenizer to use ('simple' or 'wordpiece')
        vocab_size : int
            Maximum vocabulary size
        use_pretrained : bool
            Whether to use pretrained word embeddings
        """
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.use_pretrained = use_pretrained
        
        # Set random state
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState()
        
        # Initialize embeddings
        self.word_embeddings = SimpleWordEmbeddings(embedding_dim, use_pretrained)
        
        # Initialize vocabulary
        self.vocabulary = None
        self.word_to_index = None
        self.index_to_word = None
        
        # Initialize weights
        self.weights = None
        self.biases = None
        
        # Initialize input and output sizes
        self.input_size = None
        self.output_size = None
        
        # Initialize training history
        self.training_loss = []
        self.validation_loss = []
        self.iteration_count = []
        
        # Model name for saving/loading
        self.model_name = "SimpleMultiLayerPerceptron"
    
    def _tokenize_text(self, text):
        """
        Tokenize the text into words.
        
        Parameters:
        -----------
        text : str
            The text to tokenize
            
        Returns:
        --------
        list
            List of tokens
        """
        # Simple tokenization (split on whitespace and punctuation)
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_vocabulary(self, tokens, max_size=None):
        """
        Build a vocabulary from the tokens.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
        max_size : int or None
            Maximum vocabulary size
            
        Returns:
        --------
        tuple
            (vocabulary, word_to_index, index_to_word)
        """
        # Count word frequencies
        word_counts = Counter(tokens)
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size if specified
        if max_size is not None:
            sorted_words = sorted_words[:max_size]
        
        # Create vocabulary
        vocabulary = [word for word, _ in sorted_words]
        
        # Create mappings
        word_to_index = {word: i for i, word in enumerate(vocabulary)}
        index_to_word = {i: word for i, word in enumerate(vocabulary)}
        
        return vocabulary, word_to_index, index_to_word
    
    def _create_training_data(self, tokens):
        """
        Create training data from tokens.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
            
        Returns:
        --------
        tuple
            (X, y)
        """
        X = []
        y = []
        
        for i in range(len(tokens) - self.context_size):
            # Get context
            context = tokens[i:i+self.context_size]
            
            # Get target word
            target = tokens[i+self.context_size]
            
            # Skip if any word is not in vocabulary
            if any(word not in self.word_to_index for word in context) or target not in self.word_to_index:
                continue
            
            # Convert context to indices
            context_indices = [self.word_to_index[word] for word in context]
            
            # Convert target to one-hot encoding
            target_index = self.word_to_index[target]
            target_one_hot = np.zeros(len(self.vocabulary))
            target_one_hot[target_index] = 1
            
            X.append(context_indices)
            y.append(target_one_hot)
        
        return np.array(X), np.array(y)
    
    def _initialize_weights(self):
        """
        Initialize the weights for the neural network.
        """
        # Input size is context_size * embedding_dim
        self.input_size = self.context_size * self.embedding_dim
        
        # Output size is vocabulary size
        self.output_size = len(self.vocabulary)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(self.random_state.randn(self.input_size, self.hidden_layers[0]) * 0.01)
        self.biases.append(np.zeros(self.hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.weights.append(self.random_state.randn(self.hidden_layers[i-1], self.hidden_layers[i]) * 0.01)
            self.biases.append(np.zeros(self.hidden_layers[i]))
        
        # Last hidden layer to output
        self.weights.append(self.random_state.randn(self.hidden_layers[-1], self.output_size) * 0.01)
        self.biases.append(np.zeros(self.output_size))
    
    def _relu(self, x):
        """
        ReLU activation function.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input
            
        Returns:
        --------
        numpy.ndarray
            Output
        """
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """
        Derivative of ReLU activation function.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input
            
        Returns:
        --------
        numpy.ndarray
            Output
        """
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        """
        Softmax activation function.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input
            
        Returns:
        --------
        numpy.ndarray
            Output
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _forward_pass(self, X):
        """
        Perform a forward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        tuple
            (output, hidden_activations)
        """
        hidden_activations = []
        
        # Input to first hidden layer
        hidden_input = np.dot(X, self.weights[0]) + self.biases[0]
        hidden_activation = self._relu(hidden_input)
        hidden_activations.append(hidden_activation)
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            hidden_input = np.dot(hidden_activations[-1], self.weights[i]) + self.biases[i]
            hidden_activation = self._relu(hidden_input)
            hidden_activations.append(hidden_activation)
        
        # Last hidden layer to output
        output_input = np.dot(hidden_activations[-1], self.weights[-1]) + self.biases[-1]
        output = self._softmax(output_input)
        
        return output, hidden_activations
    
    def _backward_pass(self, X, y, output, hidden_activations):
        """
        Perform a backward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray
            Target data
        output : numpy.ndarray
            Output from forward pass
        hidden_activations : list
            List of hidden layer activations
            
        Returns:
        --------
        tuple
            (weight_gradients, bias_gradients)
        """
        # Output layer error
        output_error = output - y
        
        # Initialize gradients
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        
        # Last hidden layer to output
        weight_gradients[-1] = np.dot(hidden_activations[-1].T, output_error)
        bias_gradients[-1] = np.sum(output_error, axis=0)
        
        # Propagate error backward through hidden layers
        error = output_error
        for i in range(len(self.hidden_layers) - 1, 0, -1):
            # Compute error for hidden layer
            hidden_error = np.dot(error, self.weights[i+1].T) * self._relu_derivative(hidden_activations[i])
            
            # Compute gradients
            weight_gradients[i] = np.dot(hidden_activations[i-1].T, hidden_error)
            bias_gradients[i] = np.sum(hidden_error, axis=0)
            
            # Update error for next layer
            error = hidden_error
        
        # Input to first hidden layer
        hidden_error = np.dot(error, self.weights[1].T) * self._relu_derivative(hidden_activations[0])
        weight_gradients[0] = np.dot(X.T, hidden_error)
        bias_gradients[0] = np.sum(hidden_error, axis=0)
        
        return weight_gradients, bias_gradients
    
    def _update_weights(self, weight_gradients, bias_gradients):
        """
        Update the weights using gradient descent.
        
        Parameters:
        -----------
        weight_gradients : list
            List of weight gradients
        bias_gradients : list
            List of bias gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Compute the cross-entropy loss.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted probabilities
            
        Returns:
        --------
        float
            Loss
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute loss
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        
        return loss
    
    def fit(self, text, progress_callback=None, stop_event=None):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        text : str
            Training text
        progress_callback : function or None
            Callback function for reporting progress
        stop_event : threading.Event or None
            Event for stopping training
            
        Returns:
        --------
        self
        """
        # Tokenize text
        tokens = self._tokenize_text(text)
        
        # Build vocabulary
        self.vocabulary, self.word_to_index, self.index_to_word = self._build_vocabulary(tokens, self.vocab_size)
        
        # Create training data
        X_indices, y = self._create_training_data(tokens)
        
        # Split into training and validation sets
        # Simple split without using sklearn
        n_samples = len(X_indices)
        n_val = int(n_samples * self.validation_split)
        indices = np.arange(n_samples)
        self.random_state.shuffle(indices)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train_indices = X_indices[train_indices]
        X_val_indices = X_indices[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        # Convert indices to embeddings
        X_train = np.zeros((len(X_train_indices), self.context_size * self.embedding_dim))
        X_val = np.zeros((len(X_val_indices), self.context_size * self.embedding_dim))
        
        for i, indices in enumerate(X_train_indices):
            embeddings = [self.word_embeddings.get_embedding(self.index_to_word[idx]) for idx in indices]
            X_train[i] = np.concatenate(embeddings)
        
        for i, indices in enumerate(X_val_indices):
            embeddings = [self.word_embeddings.get_embedding(self.index_to_word[idx]) for idx in indices]
            X_val[i] = np.concatenate(embeddings)
        
        # Initialize weights
        self._initialize_weights()
        
        # Reset training history
        self.training_loss = []
        self.validation_loss = []
        self.iteration_count = []
        
        # Train the model
        n_samples = len(X_train)
        n_batches = max(1, n_samples // self.batch_size)
        
        for iteration in range(self.n_iterations):
            # Check if training should be stopped
            if stop_event is not None and stop_event.is_set():
                break
            
            # Shuffle data
            indices = np.arange(n_samples)
            self.random_state.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch gradient descent
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                output, hidden_activations = self._forward_pass(X_batch)
                
                # Backward pass
                weight_gradients, bias_gradients = self._backward_pass(X_batch, y_batch, output, hidden_activations)
                
                # Update weights
                self._update_weights(weight_gradients, bias_gradients)
            
            # Compute training loss
            train_output, _ = self._forward_pass(X_train)
            train_loss = self._cross_entropy_loss(y_train, train_output)
            
            # Compute validation loss
            val_output, _ = self._forward_pass(X_val)
            val_loss = self._cross_entropy_loss(y_val, val_output)
            
            # Store losses
            self.training_loss.append(train_loss)
            self.validation_loss.append(val_loss)
            self.iteration_count.append(iteration)
            
            # Report progress
            if progress_callback is not None and iteration % 10 == 0:
                progress_callback(iteration, self.n_iterations, train_loss, val_loss)
        
        return self
    
    def predict_next_word(self, context_text, top_n=5):
        """
        Predict the next word given a context.
        
        Parameters:
        -----------
        context_text : str
            Context text
        top_n : int
            Number of top predictions to return
            
        Returns:
        --------
        tuple
            (predictions, info)
        """
        if self.vocabulary is None:
            raise ValueError("Model not trained yet.")
        
        # Tokenize context
        context_tokens = self._tokenize_text(context_text)
        
        # Prepare info dictionary
        info = {
            "original_context": context_tokens,
            "adjusted_context": None,
            "adjustment_made": False,
            "adjustment_type": [],
            "unknown_words": []
        }
        
        # Adjust context length
        if len(context_tokens) < self.context_size:
            # Pad with empty strings
            padding = [""] * (self.context_size - len(context_tokens))
            context_tokens = padding + context_tokens
            info["adjustment_made"] = True
            info["adjustment_type"].append("padded_beginning")
        elif len(context_tokens) > self.context_size:
            # Use only the last context_size tokens
            context_tokens = context_tokens[-self.context_size:]
            info["adjustment_made"] = True
            info["adjustment_type"].append("truncated_beginning")
        
        # Replace unknown words with known words
        for i, token in enumerate(context_tokens):
            if token not in self.word_to_index:
                info["unknown_words"].append(token)
                # Replace with a random known word
                context_tokens[i] = self.vocabulary[self.random_state.randint(0, len(self.vocabulary))]
                info["adjustment_made"] = True
                if "replaced_unknown" not in info["adjustment_type"]:
                    info["adjustment_type"].append("replaced_unknown")
        
        # Store adjusted context
        info["adjusted_context"] = context_tokens
        
        # Convert tokens to embeddings
        X = np.zeros((1, self.context_size * self.embedding_dim))
        embeddings = [self.word_embeddings.get_embedding(token) for token in context_tokens]
        X[0] = np.concatenate(embeddings)
        
        # Forward pass
        output, _ = self._forward_pass(X)
        
        # Get top N predictions
        top_indices = np.argsort(output[0])[-top_n:][::-1]
        predictions = [(self.index_to_word[idx], float(output[0][idx])) for idx in top_indices]
        
        return predictions, info
    
    def predict_next_n_words(self, context_text, n=10):
        """
        Predict the next n words given a context.
        
        Parameters:
        -----------
        context_text : str
            Context text
        n : int
            Number of words to predict
            
        Returns:
        --------
        tuple
            (generated_words, info)
        """
        if self.vocabulary is None:
            raise ValueError("Model not trained yet.")
        
        # Tokenize context
        context_tokens = self._tokenize_text(context_text)
        
        # Prepare info dictionary
        info = {
            "original_context": context_tokens,
            "adjusted_context": None,
            "adjustment_made": False,
            "prediction_steps": [],
            "full_text": ""
        }
        
        # Generate words
        generated_words = []
        current_context = context_tokens.copy()
        
        for i in range(n):
            # Predict next word
            predictions, step_info = self.predict_next_word(" ".join(current_context), top_n=1)
            
            # Store step info
            info["prediction_steps"].append(step_info)
            
            # Update adjustment info
            if step_info["adjustment_made"] and not info["adjustment_made"]:
                info["adjustment_made"] = True
                info["adjusted_context"] = step_info["adjusted_context"]
            
            # Get predicted word
            predicted_word, _ = predictions[0]
            generated_words.append(predicted_word)
            
            # Update context
            current_context.append(predicted_word)
            if len(current_context) > self.context_size:
                current_context = current_context[-self.context_size:]
        
        # Create full text
        if info["adjusted_context"]:
            full_text = " ".join(info["adjusted_context"] + generated_words)
        else:
            full_text = " ".join(context_tokens + generated_words)
        
        info["full_text"] = full_text
        
        return generated_words, info
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
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
        MultiLayerPerceptron
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
        --------
        dict
            Dictionary containing model information
        """
        return {
            'model_type': 'multi_layer',
            'context_size': self.context_size,
            'embedding_dim': self.embedding_dim,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'vocabulary_size': len(self.vocabulary) if self.vocabulary else 0,
            'input_size': self.input_size,
            'output_size': self.output_size
        }

class SimpleSingleLayerPerceptron(SimpleMultiLayerPerceptron):
    """
    A simplified single-layer perceptron implementation for language modeling.
    """
    
    def __init__(self, context_size=2, hidden_size=64, learning_rate=0.1, n_iterations=1000,
                 batch_size=32, validation_split=0.1, random_state=None, tokenizer_type='simple',
                 vocab_size=5000, use_pretrained=False, embedding_dim=50):
        """
        Initialize the Single-Layer Perceptron model.
        
        Parameters:
        -----------
        context_size : int
            Number of previous words to use as context
        hidden_size : int
            Size of the single hidden layer
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of training iterations
        batch_size : int
            Batch size for mini-batch gradient descent
        validation_split : float
            Fraction of data to use for validation
        random_state : int or None
            Random seed for reproducibility
        tokenizer_type : str
            Type of tokenizer to use ('simple' or 'wordpiece')
        vocab_size : int
            Maximum vocabulary size
        use_pretrained : bool
            Whether to use pretrained word embeddings
        embedding_dim : int
            Dimension of word embeddings
        """
        # Call the parent class initializer with a single hidden layer
        super().__init__(
            context_size=context_size,
            hidden_layers=[hidden_size],  # Single hidden layer
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            use_pretrained=use_pretrained,
            embedding_dim=embedding_dim
        )
        
        # Store the hidden size for easier access
        self.hidden_size = hidden_size
        
        # Override the model name
        self.model_name = "SimpleSingleLayerPerceptron"
    
    def _initialize_weights(self):
        """
        Initialize the weights for the single-layer perceptron.
        """
        # Call the parent method to initialize weights
        super()._initialize_weights()
        
        # Ensure we only have a single hidden layer
        if len(self.weights) > 2:
            # Keep only the input-to-hidden and hidden-to-output weights
            self.weights = self.weights[:2]
            self.biases = self.biases[:2]
    
    def _forward_pass(self, X):
        """
        Perform a forward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        tuple
            (output, hidden_activations)
        """
        # Input to hidden layer
        hidden_input = np.dot(X, self.weights[0]) + self.biases[0]
        hidden_activation = self._relu(hidden_input)
        
        # Hidden to output layer
        output_input = np.dot(hidden_activation, self.weights[1]) + self.biases[1]
        output = self._softmax(output_input)
        
        return output, [hidden_activation]
    
    def _backward_pass(self, X, y, output, hidden_activations):
        """
        Perform a backward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray
            Target data
        output : numpy.ndarray
            Output from forward pass
        hidden_activations : list
            List of hidden layer activations
            
        Returns:
        --------
        tuple
            (weight_gradients, bias_gradients)
        """
        # Get the single hidden layer activation
        hidden_activation = hidden_activations[0]
        
        # Output layer error
        output_error = output - y
        
        # Hidden layer error
        hidden_error = np.dot(output_error, self.weights[1].T) * self._relu_derivative(hidden_activation)
        
        # Weight gradients
        weight_gradients = [
            np.dot(X.T, hidden_error),
            np.dot(hidden_activation.T, output_error)
        ]
        
        # Bias gradients
        bias_gradients = [
            np.sum(hidden_error, axis=0),
            np.sum(output_error, axis=0)
        ]
        
        return weight_gradients, bias_gradients
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
        --------
        dict
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            'model_type': 'single_layer',
            'hidden_size': self.hidden_size
        })
        return info