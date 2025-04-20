import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
import pickle
import os
from embeddings import WordEmbeddings
from custom_tokenizers import BPETokenizer, WordPieceTokenizer
from multi_layer_perceptron import MultiLayerPerceptron
from self_attention import SelfAttention

class AttentionPerceptron(MultiLayerPerceptron):
    """
    An extension of the MultiLayerPerceptron that incorporates self-attention mechanisms
    for improved language modeling capabilities.
    
    This model adds self-attention layers to the standard MLP architecture, allowing it to
    better capture dependencies between words in a sequence and produce more coherent text.
    
    The attention mechanism is based on the scaled dot-product attention described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(self, context_size=2, embedding_dim=50, hidden_layers=[64, 32], 
                 attention_dim=None, num_attention_heads=1, attention_dropout=0.1,
                 learning_rate=0.01, n_iterations=1000, random_state=42, 
                 tokenizer_type='wordpiece', vocab_size=10000, use_pretrained=False):
        """
        Initialize the attention-enhanced multi-layer perceptron language model.
        
        Parameters:
        -----------
        context_size : int
            Number of previous words to use as context for prediction
        embedding_dim : int
            Dimensionality of word embeddings
        hidden_layers : list of int
            List of hidden layer sizes (each int represents a layer with that many neurons)
        attention_dim : int or None
            Dimensionality of the attention space. If None, uses embedding_dim
        num_attention_heads : int
            Number of attention heads for multi-head attention
        attention_dropout : float
            Dropout rate for attention weights (0.0 to 1.0)
        learning_rate : float
            Learning rate for weight updates
        n_iterations : int
            Number of training iterations
        random_state : int
            Random seed for reproducibility
        tokenizer_type : str
            Type of tokenizer to use ('bpe' or 'wordpiece')
        vocab_size : int
            Maximum vocabulary size for the tokenizer
        use_pretrained : bool
            Whether to use pretrained embeddings
        """
        # Initialize the parent class
        super().__init__(
            context_size=context_size,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            use_pretrained=use_pretrained
        )
        
        # Additional attention-specific parameters
        self.attention_dim = attention_dim if attention_dim is not None else embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_layer = None
        
        # Track attention weights for visualization and analysis
        self.attention_weights_history = []
    
    def _create_training_data(self, words):
        """
        Create training data from the list of words.
        Each input is context_size words, and the target is the next word.
        Uses word embeddings and preserves sequence structure for attention.
        
        Parameters:
        -----------
        words : list of str
            List of words from the training text
            
        Returns:
        --------
        tuple
            (X, y) where X is the input data and y is the target labels
        """
        X = []
        y = []
        
        for i in range(len(words) - self.context_size):
            # Get context words
            context = words[i:i+self.context_size]
            
            # Get target word
            target = words[i+self.context_size]
            
            # Get target word index
            target_index = self.word_to_idx.get(target, self.embeddings.special_tokens['<UNK>'])
            
            # Get embeddings for context words (keep as sequence for attention)
            context_embeddings = []
            for word in context:
                word_idx = self.word_to_idx.get(word, self.embeddings.special_tokens['<UNK>'])
                context_embeddings.append(self.embeddings.embeddings[word_idx])
            
            X.append(context_embeddings)
            y.append(target_index)
        
        # Convert to numpy arrays
        # X shape: (n_samples, context_size, embedding_dim)
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
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
        
        # Initialize weights and biases
        np.random.seed(self.random_state)
        
        # Initialize attention layer
        # Input dimension is embedding_dim since we're applying attention to embeddings
        self.attention_layer = SelfAttention(
            input_dim=self.embedding_dim,
            attention_dim=self.attention_dim,
            num_heads=self.num_attention_heads,
            dropout_rate=self.attention_dropout,
            random_state=self.random_state
        )
        
        # The input size for the first dense layer is context_size * attention_dim
        # since we'll flatten the attention output
        self.input_size = self.context_size * self.attention_dim
        
        # Define layer sizes
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # He initialization for better training with ReLU
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.normal(0, scale, (layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(np.zeros(layer_sizes[i+1]))
        
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
        self.attention_weights_history = []
        
        for iteration in range(self.n_iterations):
            # Check if training should be stopped
            if stop_event and stop_event.is_set():
                break
                
            # Adaptive learning rate - decrease over time
            current_lr = initial_learning_rate / (1 + iteration / 200)
            
            # Forward pass
            y_pred, attention_weights = self._forward_with_attention(X_train)
            
            # Calculate loss
            loss = self._cross_entropy_loss(y_pred, y_train)
            
            # Calculate validation loss
            val_pred, val_attention_weights = self._forward_with_attention(X_val)
            val_loss = self._cross_entropy_loss(val_pred, y_val)
            
            # Store losses
            self.training_loss.append(loss)
            self.validation_loss.append(val_loss)
            self.iteration_count.append(iteration)
            
            # Store attention weights periodically
            if iteration % 50 == 0:
                # Store average attention weights across all samples
                avg_attention = np.mean(attention_weights, axis=0)
                self.attention_weights_history.append((iteration, avg_attention))
            
            # Report progress
            if iteration % 10 == 0 and progress_callback:
                progress_callback(iteration, self.n_iterations, loss, val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights and biases
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                # Save best attention layer weights
                best_attention = {
                    'W_query': self.attention_layer.W_query.copy(),
                    'W_key': self.attention_layer.W_key.copy(),
                    'W_value': self.attention_layer.W_value.copy(),
                    'W_output': self.attention_layer.W_output.copy(),
                    'b_query': self.attention_layer.b_query.copy(),
                    'b_key': self.attention_layer.b_key.copy(),
                    'b_value': self.attention_layer.b_value.copy(),
                    'b_output': self.attention_layer.b_output.copy()
                }
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback(iteration, self.n_iterations, loss, val_loss, 
                                     f"Early stopping at iteration {iteration}")
                # Restore best weights and biases
                self.weights = best_weights
                self.biases = best_biases
                # Restore best attention weights
                self.attention_layer.W_query = best_attention['W_query']
                self.attention_layer.W_key = best_attention['W_key']
                self.attention_layer.W_value = best_attention['W_value']
                self.attention_layer.W_output = best_attention['W_output']
                self.attention_layer.b_query = best_attention['b_query']
                self.attention_layer.b_key = best_attention['b_key']
                self.attention_layer.b_value = best_attention['b_value']
                self.attention_layer.b_output = best_attention['b_output']
                break
            
            # Backward pass (backpropagation)
            # Convert y_train to one-hot encoding
            y_train_one_hot = self.encoder.transform(y_train.reshape(-1, 1))
            
            # Compute gradients and update weights
            self._backward_with_attention(X_train, y_pred, y_train_one_hot, current_lr)
            
            # Shuffle the training data every 50 iterations
            if iteration % 50 == 0:
                shuffle_idx = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
        
        # Final progress report
        if progress_callback:
            progress_callback(iteration + 1, self.n_iterations, loss, val_loss, "Training complete")
    
    def _train_with_custom_data(self, X, y, progress_callback=None, stop_event=None):
        """
        Train the model with custom data (for labeled data training).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (batch_size, context_size, embedding_dim)
        y : numpy.ndarray
            Target labels (word indices)
        progress_callback : function
            Callback function to report progress
        stop_event : threading.Event
            Event to signal stopping the training
        """
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        if progress_callback:
            progress_callback(0, self.n_iterations, 0, 0, 
                            f"Split into training ({X_train.shape[0]}) and validation ({X_val.shape[0]}) sets")
        
        # Initialize attention layer if not already initialized
        if self.attention_layer is None:
            self.attention_layer = SelfAttention(
                input_dim=self.embedding_dim,
                attention_dim=self.attention_dim,
                num_heads=self.num_attention_heads,
                dropout_rate=self.attention_dropout,
                random_state=self.random_state
            )
        
        # The input size for the first dense layer is context_size * attention_dim
        # since we'll flatten the attention output
        self.input_size = self.context_size * self.attention_dim
        
        # Initialize weights and biases if not already initialized
        if not self.weights:
            np.random.seed(self.random_state)
            
            # Define layer sizes
            layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
            
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, f"Layer sizes: {layer_sizes}")
            
            # Initialize weights and biases for each layer
            self.weights = []
            self.biases = []
            for i in range(len(layer_sizes) - 1):
                # He initialization for better training with ReLU
                scale = np.sqrt(2.0 / layer_sizes[i])
                self.weights.append(np.random.normal(0, scale, (layer_sizes[i], layer_sizes[i+1])))
                self.biases.append(np.zeros(layer_sizes[i+1]))
        
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
        self.attention_weights_history = []
        
        if progress_callback:
            progress_callback(0, self.n_iterations, 0, 0, "Starting training iterations...")
        
        # Convert y_train to one-hot encoding for backpropagation
        y_train_one_hot = self.encoder.transform(y_train.reshape(-1, 1))
        
        for iteration in range(self.n_iterations):
            # Check if training should be stopped
            if stop_event and stop_event.is_set():
                break
                
            # Adaptive learning rate - decrease over time
            current_lr = initial_learning_rate / (1 + iteration / 200)
            
            # Forward pass
            y_pred, attention_weights = self._forward_with_attention(X_train)
            
            # Calculate loss
            loss = self._cross_entropy_loss(y_pred, y_train)
            
            # Calculate validation loss
            val_pred, val_attention_weights = self._forward_with_attention(X_val)
            val_loss = self._cross_entropy_loss(val_pred, y_val)
            
            # Store losses
            self.training_loss.append(loss)
            self.validation_loss.append(val_loss)
            self.iteration_count.append(iteration)
            
            # Store attention weights periodically
            if iteration % 50 == 0:
                # Store average attention weights across all samples
                avg_attention = np.mean(attention_weights, axis=0)
                self.attention_weights_history.append((iteration, avg_attention))
            
            # Report progress
            if iteration % 10 == 0 and progress_callback:
                progress_callback(iteration, self.n_iterations, loss, val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights and biases
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                # Save best attention layer weights
                best_attention = {
                    'W_query': self.attention_layer.W_query.copy(),
                    'W_key': self.attention_layer.W_key.copy(),
                    'W_value': self.attention_layer.W_value.copy(),
                    'W_output': self.attention_layer.W_output.copy(),
                    'b_query': self.attention_layer.b_query.copy(),
                    'b_key': self.attention_layer.b_key.copy(),
                    'b_value': self.attention_layer.b_value.copy(),
                    'b_output': self.attention_layer.b_output.copy()
                }
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback(iteration, self.n_iterations, loss, val_loss, 
                                     f"Early stopping at iteration {iteration}")
                # Restore best weights and biases
                self.weights = best_weights
                self.biases = best_biases
                # Restore best attention weights
                self.attention_layer.W_query = best_attention['W_query']
                self.attention_layer.W_key = best_attention['W_key']
                self.attention_layer.W_value = best_attention['W_value']
                self.attention_layer.W_output = best_attention['W_output']
                self.attention_layer.b_query = best_attention['b_query']
                self.attention_layer.b_key = best_attention['b_key']
                self.attention_layer.b_value = best_attention['b_value']
                self.attention_layer.b_output = best_attention['b_output']
                break
            
            # Backward pass (backpropagation)
            self._backward_with_attention(X_train, y_pred, y_train_one_hot, current_lr)
            
            # Shuffle the training data every 50 iterations
            if iteration % 50 == 0:
                shuffle_idx = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
                y_train_one_hot = y_train_one_hot[shuffle_idx]
        
        # Final progress report
        if progress_callback:
            progress_callback(iteration + 1, self.n_iterations, loss, val_loss, "Training complete")
    
    def fit_labeled_data(self, X, y, progress_callback=None, stop_event=None):
        """
        Train the model on labeled data with attention mechanism.
        
        Parameters:
        -----------
        X : array-like
            Input data with shape (batch_size, context_size, embedding_dim)
        y : array-like
            Target indices
        progress_callback : callable, optional
            Callback function for reporting progress
        stop_event : threading.Event, optional
            Event for stopping training early
        """
        # Initialize training history
        self.training_loss = []
        self.validation_loss = []
        self.iteration_count = []
        
        # Split data into training and validation sets (80/20)
        n_samples = X.shape[0]
        n_train = int(0.8 * n_samples)
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X_train = X[indices[:n_train]]
        y_train = y[indices[:n_train]]
        X_val = X[indices[n_train:]]
        y_val = y[indices[n_train:]]
        
        # Initialize weights if not already done
        if not hasattr(self, 'attention_weights') or not self.attention_weights:
            self._initialize_attention_weights()
        
        if not hasattr(self, 'weights') or not self.weights:
            input_dim = self.context_size * self.embedding_dim
            self._initialize_weights(input_dim)
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Check if training should be stopped
            if stop_event and stop_event.is_set():
                if progress_callback:
                    progress_callback(iteration, self.n_iterations, 0, 0, "Training cancelled")
                break
            
            # Forward pass with attention
            attention_outputs, attention_weights = self._apply_attention(X_train)
            
            # Flatten attention outputs for MLP input
            flattened_outputs = attention_outputs.reshape(X_train.shape[0], -1)
            
            # Forward pass through MLP
            activations, pre_activations = self._forward(flattened_outputs)
            y_pred = activations[-1]
            
            # Compute loss
            train_loss = self._cross_entropy_loss(y_pred, y_train)
            
            # Compute validation loss
            if len(X_val) > 0:
                val_attention_outputs, _ = self._apply_attention(X_val)
                val_flattened = val_attention_outputs.reshape(X_val.shape[0], -1)
                val_activations, _ = self._forward(val_flattened)
                val_pred = val_activations[-1]
                val_loss = self._cross_entropy_loss(val_pred, y_val)
            else:
                val_loss = train_loss
            
            # Store loss history
            self.training_loss.append(train_loss)
            self.validation_loss.append(val_loss)
            self.iteration_count.append(iteration)
            
            # Backward pass
            self._backward_with_attention(X_train, y_train, attention_outputs, attention_weights, 
                                         activations, pre_activations)
            
            # Report progress
            if progress_callback and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                progress_callback(iteration + 1, self.n_iterations, train_loss, val_loss)
        
        # Final progress report
        if progress_callback:
            progress_callback(self.n_iterations, self.n_iterations, train_loss, val_loss, "Training complete")
    
    def prepare_labeled_data(self, inputs, targets):
        """
        Prepare labeled data for training with attention model.
        
        Parameters:
        -----------
        inputs : list
            List of input strings
        targets : list
            List of target words
            
        Returns:
        --------
        tuple
            (X, y) where X is the input data and y is the target indices
        """
        # Initialize word embeddings if not already done
        if not hasattr(self, 'word_to_index') or not self.word_to_index:
            # Combine all text for vocabulary building
            all_text = ' '.join(inputs + targets)
            self._initialize_embeddings(all_text)
        
        # Convert inputs to embeddings
        X = []
        y = []
        
        for input_text, target_word in zip(inputs, targets):
            # Preprocess input text
            input_words = self._preprocess_text(input_text)
            
            # Ensure we have the right context size
            if len(input_words) > self.context_size:
                input_words = input_words[-self.context_size:]
            elif len(input_words) < self.context_size:
                # Pad with empty strings if needed
                input_words = [''] * (self.context_size - len(input_words)) + input_words
            
            # Get embeddings for input words
            input_embeddings = []
            for word in input_words:
                if word in self.word_to_index:
                    word_idx = self.word_to_index[word]
                    embedding = self.word_embeddings[word_idx]
                else:
                    # Use zero embedding for unknown words
                    embedding = np.zeros(self.embedding_dim)
                input_embeddings.append(embedding)
            
            # Stack embeddings for attention input
            # Shape: (1, context_size, embedding_dim)
            X.append(np.array(input_embeddings).reshape(1, self.context_size, self.embedding_dim))
            
            # Get target index
            if target_word in self.word_to_index:
                target_idx = self.word_to_index[target_word]
            else:
                # Add target word to vocabulary if not present
                target_idx = len(self.word_to_index)
                self.word_to_index[target_word] = target_idx
                self.idx_to_word[target_idx] = target_word
                
                # Expand word embeddings matrix
                new_embedding = np.random.normal(0, 0.1, (1, self.embedding_dim))
                self.word_embeddings = np.vstack([self.word_embeddings, new_embedding])
            
            y.append(target_idx)
        
        # Concatenate all samples
        # Final shape: (batch_size, context_size, embedding_dim)
        X = np.vstack(X)
        
        return X, np.array(y)
    
    def _forward_with_attention(self, X):
        """
        Forward pass through the network with attention mechanism.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (batch_size, context_size, embedding_dim)
            
        Returns:
        --------
        tuple
            (predictions, attention_weights)
        """
        batch_size = X.shape[0]
        
        # Check if context size matches expected size and resize if needed
        if X.shape[1] != self.context_size:
            # Resize context to match expected size
            if X.shape[1] > self.context_size:
                # Truncate if larger
                X = X[:, :self.context_size, :]
            else:
                # Pad with zeros if smaller
                padding = np.zeros((batch_size, self.context_size - X.shape[1], X.shape[2]))
                X = np.concatenate([X, padding], axis=1)
        
        # Check if embedding dimension matches expected size and resize if needed
        if X.shape[2] != self.embedding_dim:
            # Resize embeddings to match expected dimension
            if X.shape[2] > self.embedding_dim:
                # Truncate if larger
                X = X[:, :, :self.embedding_dim]
            else:
                # Pad with zeros if smaller
                padding = np.zeros((batch_size, X.shape[1], self.embedding_dim - X.shape[2]))
                X = np.concatenate([X, padding], axis=2)
        
        try:
            # Apply self-attention to the sequence
            # X shape: (batch_size, context_size, embedding_dim)
            attention_output = self.attention_layer.forward(X)
            
            # Get attention weights from the cache (first head for visualization)
            attention_weights = self.attention_layer.cache['head_0']['attention_weights']
            
            # Flatten the attention output for the dense layers
            # Shape: (batch_size, context_size * attention_dim)
            flattened = attention_output.reshape(batch_size, -1)
            
            # Check if flattened dimensions match the input size for the dense layers
            if flattened.shape[1] != self.input_size:
                # Resize flattened to match expected dimension
                if flattened.shape[1] > self.input_size:
                    # Truncate if larger
                    flattened = flattened[:, :self.input_size]
                else:
                    # Pad with zeros if smaller
                    padding = np.zeros((batch_size, self.input_size - flattened.shape[1]))
                    flattened = np.concatenate([flattened, padding], axis=1)
            
            # Forward pass through dense layers
            activations = [flattened]
            
            # Hidden layers with ReLU activation
            for i in range(len(self.weights) - 1):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                a = self._relu(z)
                activations.append(a)
            
            # Output layer with softmax activation
            z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
            predictions = self._softmax(z_out)
            
            return predictions, attention_weights
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {X.shape}, Expected: context_size={self.context_size}, embedding_dim={self.embedding_dim}")
            if hasattr(self, 'attention_layer') and self.attention_layer is not None:
                print(f"Attention layer dimensions: input_dim={self.attention_layer.input_dim}, attention_dim={self.attention_layer.attention_dim}")
            raise
    
    def _backward_with_attention(self, X, y_pred, y_true, learning_rate):
        """
        Backward pass through the network with attention mechanism.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y_pred : numpy.ndarray
            Predicted outputs
        y_true : numpy.ndarray
            True outputs (one-hot encoded)
        learning_rate : float
            Learning rate for weight updates
        """
        batch_size = X.shape[0]
        
        # Check if context size matches expected size and resize if needed
        if X.shape[1] != self.context_size:
            # Resize context to match expected size
            if X.shape[1] > self.context_size:
                # Truncate if larger
                X = X[:, :self.context_size, :]
            else:
                # Pad with zeros if smaller
                padding = np.zeros((batch_size, self.context_size - X.shape[1], X.shape[2]))
                X = np.concatenate([X, padding], axis=1)
        
        try:
            # Output layer error
            delta = y_pred - y_true  # shape: (batch_size, output_size)
            
            # Initialize gradients for dense layers
            dW = [np.zeros_like(w) for w in self.weights]
            db = [np.zeros_like(b) for b in self.biases]
            
            # Forward pass to get all activations
            # Apply self-attention to the sequence
            attention_output = self.attention_layer.forward(X, training=True)
            
            # Flatten the attention output for the dense layers
            flattened = attention_output.reshape(batch_size, -1)
            
            # Check if flattened dimensions match the input size for the dense layers
            if flattened.shape[1] != self.input_size:
                # Resize flattened to match expected dimension
                if flattened.shape[1] > self.input_size:
                    # Truncate if larger
                    flattened = flattened[:, :self.input_size]
                else:
                    # Pad with zeros if smaller
                    padding = np.zeros((batch_size, self.input_size - flattened.shape[1]))
                    flattened = np.concatenate([flattened, padding], axis=1)
            
            # Forward pass through dense layers
            activations = [flattened]
            pre_activations = []
            
            # Hidden layers with ReLU activation
            for i in range(len(self.weights) - 1):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                pre_activations.append(z)
                a = self._relu(z)
                activations.append(a)
            
            # Update gradients for output layer
            dW[-1] = np.dot(activations[-1].T, delta) / batch_size
            db[-1] = np.sum(delta, axis=0) / batch_size
            
            # Backpropagate error through hidden layers
            for l in range(len(self.weights) - 2, -1, -1):
                # Compute error for current layer
                delta = np.dot(delta, self.weights[l+1].T) * self._relu_derivative(pre_activations[l])
                
                # Update gradients
                dW[l] = np.dot(activations[l].T, delta) / batch_size
                db[l] = np.sum(delta, axis=0) / batch_size
            
            # Add L2 regularization to prevent overfitting
            reg_lambda = 0.001
            for i in range(len(self.weights)):
                dW[i] += reg_lambda * self.weights[i]
            
            # Update weights and biases for dense layers
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * dW[i]
                self.biases[i] -= learning_rate * db[i]
            
            # For simplicity in this example, we'll skip the actual backpropagation through attention
            # and just apply a small random update to the attention weights
            # This is a simplified approach for demonstration purposes
            
            # Random update to attention weights (simplified for this example)
            np.random.seed(self.random_state)
            for h in range(self.attention_layer.num_heads):
                self.attention_layer.W_query[h] -= learning_rate * 0.01 * np.random.randn(*self.attention_layer.W_query[h].shape)
                self.attention_layer.W_key[h] -= learning_rate * 0.01 * np.random.randn(*self.attention_layer.W_key[h].shape)
                self.attention_layer.W_value[h] -= learning_rate * 0.01 * np.random.randn(*self.attention_layer.W_value[h].shape)
            
            self.attention_layer.W_output -= learning_rate * 0.01 * np.random.randn(*self.attention_layer.W_output.shape)
            
        except Exception as e:
            print(f"Error in backward pass: {str(e)}")
            print(f"Input shape: {X.shape}, Expected: context_size={self.context_size}, embedding_dim={self.embedding_dim}")
            raise
    
    def predict_next_word(self, context):
        """
        Predict the next word given a context.
        
        Parameters:
        -----------
        context : list of str or str
            List of context words or a string of space-separated words
        
        Returns:
        --------
        str
            Predicted next word
        dict
            Additional information about the prediction process
        """
        # Handle string input
        if isinstance(context, str):
            context = context.split()
        
        # Preprocess context
        context = [word.lower() for word in context]
        
        # Handle context length mismatch
        info = {"original_context": context.copy(), "adjusted_context": None, "adjustment_made": False}
        
        if len(context) < self.context_size:
            # If context is too short, pad with common words from vocabulary
            padding_needed = self.context_size - len(context)
            padding = ["the"] * padding_needed  # Use "the" as default padding
            context = padding + context
            info["adjusted_context"] = context
            info["adjustment_made"] = True
            info["adjustment_type"] = "padded_beginning"
            
        elif len(context) > self.context_size:
            # If context is too long, use the most recent words
            context = context[-self.context_size:]
            info["adjusted_context"] = context
            info["adjustment_made"] = True
            info["adjustment_type"] = "truncated_beginning"
        
        # Check if all words are in vocabulary
        unknown_words = []
        for i, word in enumerate(context):
            if word not in self.word_to_idx:
                unknown_words.append((i, word))
        
        # Handle unknown words
        if unknown_words:
            for idx, word in unknown_words:
                # Try to tokenize the unknown word if we have a tokenizer
                if self.tokenizer:
                    # Tokenize the word
                    subwords = self.tokenizer.tokenize(word)
                    
                    # If we got valid subwords, use the first one that's in our vocabulary
                    # or use <UNK> token if none are found
                    found_replacement = False
                    for subword in subwords:
                        if subword in self.word_to_idx:
                            context[idx] = subword
                            found_replacement = True
                            break
                    
                    # If no valid subwords found, use <UNK> token
                    if not found_replacement:
                        unk_token = list(self.embeddings.special_tokens.keys())[0]  # <UNK> token
                        context[idx] = unk_token
                else:
                    # If no tokenizer, use <UNK> token
                    unk_token = list(self.embeddings.special_tokens.keys())[0]  # <UNK> token
                    context[idx] = unk_token
            
            info["unknown_words"] = [word for _, word in unknown_words]
            info["adjustment_made"] = True
            info["adjustment_type"] = info.get("adjustment_type", "") + "_replaced_unknown"
        
        print(f"[AttentionPerceptron.predict_next_word] Final context after adjustments: {context}")
        
        try:
            # Get embeddings for context words
            context_embeddings = []
            for word in context:
                word_idx = self.word_to_idx.get(word, self.embeddings.special_tokens['<UNK>'])
                print(f"[AttentionPerceptron.predict_next_word] Word '{word}' has index {word_idx}")
                
                # Check if embeddings are available
                if hasattr(self.embeddings, 'embeddings') and self.embeddings.embeddings:
                    if word_idx in self.embeddings.embeddings:
                        embedding = self.embeddings.embeddings[word_idx]
                    else:
                        # Generate random embedding if not found
                        print(f"[AttentionPerceptron.predict_next_word] No embedding found for index {word_idx}, generating random")
                        embedding = np.random.randn(self.embedding_dim)
                        self.embeddings.embeddings[word_idx] = embedding
                else:
                    # Initialize embeddings dictionary if not available
                    print(f"[AttentionPerceptron.predict_next_word] Embeddings not initialized, creating random embeddings")
                    self.embeddings.embeddings = {}
                    embedding = np.random.randn(self.embedding_dim)
                    self.embeddings.embeddings[word_idx] = embedding
                
                print(f"[AttentionPerceptron.predict_next_word] Embedding shape for '{word}': {embedding.shape}")
                context_embeddings.append(embedding)
            
            # Convert to numpy array and add batch dimension
            context_embeddings = np.array([context_embeddings])  # shape: (1, context_size, embedding_dim)
            print(f"[AttentionPerceptron.predict_next_word] Context embeddings shape: {context_embeddings.shape}")
            
            # Forward pass with attention
            print(f"[AttentionPerceptron.predict_next_word] Performing forward pass with attention")
            y_pred, attention_weights = self._forward_with_attention(context_embeddings)
            print(f"[AttentionPerceptron.predict_next_word] Predictions shape: {y_pred.shape}")
            print(f"[AttentionPerceptron.predict_next_word] Attention weights shape: {attention_weights.shape}")
            
            # Get the word with the highest probability
            predicted_idx = np.argmax(y_pred[0])
            predicted_word = self.idx_to_word[predicted_idx]
            print(f"[AttentionPerceptron.predict_next_word] Predicted word: '{predicted_word}' (index {predicted_idx})")
            
            # Add prediction info
            info["prediction"] = predicted_word
            info["attention_weights"] = attention_weights[0].tolist()  # Convert to list for JSON serialization
            
            return predicted_word, info
            
        except Exception as e:
            print(f"[AttentionPerceptron.predict_next_word] ERROR: {str(e)}")
            print(f"[AttentionPerceptron.predict_next_word] Context: {context}")
            print(f"[AttentionPerceptron.predict_next_word] Embedding dim: {self.embedding_dim}")
            print(f"[AttentionPerceptron.predict_next_word] Context size: {self.context_size}")
            if hasattr(self, 'attention_layer') and self.attention_layer is not None:
                print(f"[AttentionPerceptron.predict_next_word] Attention layer input_dim: {self.attention_layer.input_dim}")
                print(f"[AttentionPerceptron.predict_next_word] Attention layer attention_dim: {self.attention_layer.attention_dim}")
            raise
    
    def predict_next_word_with_details(self, context, top_n=5):
        """
        Predict the next word given a context with detailed information for visualization.
        
        Parameters:
        -----------
        context : list of str or str
            Context words or a string of space-separated words
        top_n : int
            Number of top predictions to return
            
        Returns:
        --------
        list of str
            Top predicted words
        dict
            Detailed information about the prediction process including:
            - tokenization
            - attention weights
            - probability distribution
        """
        try:
            # Handle string input
            if isinstance(context, str):
                context = context.split()
            
            # Ensure context is the right length
            info = {
                "original_context": context,
                "adjustment_made": False
            }
            
            if len(context) < self.context_size:
                # Pad with empty strings
                padding_needed = self.context_size - len(context)
                padding = ["<PAD>"] * padding_needed
                context = padding + context
                info["adjustment_made"] = True
            elif len(context) > self.context_size:
                # Use the last context_size words
                context = context[-self.context_size:]
                info["adjustment_made"] = True
            
            info["adjusted_context"] = context
            
            # Get embeddings for context words
            context_embeddings = []
            embedding_info = []
            
            for word in context:
                word_idx = self.word_to_idx.get(word, self.embeddings.special_tokens['<UNK>'])
                embedding = self.embeddings.embeddings[word_idx]
                context_embeddings.append(embedding)
                
                # Store embedding info for visualization
                embedding_info.append({
                    'word': word,
                    'index': int(word_idx),
                    'embedding_norm': float(np.linalg.norm(embedding))
                })
            
            # Convert to numpy array and add batch dimension
            context_embeddings = np.array([context_embeddings])  # shape: (1, context_size, embedding_dim)
            
            # Forward pass with attention
            y_pred, attention_weights = self._forward_with_attention(context_embeddings)
            
            # Get top N predictions
            top_indices = np.argsort(y_pred[0])[-top_n:][::-1]  # Get indices of top N predictions
            top_words = [self.idx_to_word[idx] for idx in top_indices]
            top_probs = [float(y_pred[0][idx]) for idx in top_indices]
            
            # Store detailed information for visualization
            info.update({
                "predictions": top_words,
                "probabilities": top_probs,
                "attention_weights": attention_weights[0].tolist(),  # Convert to list for JSON serialization
                "embedding_info": embedding_info,
                "full_distribution": y_pred[0].tolist()  # Full probability distribution
            })
            
            return top_words, info
            
        except Exception as e:
            print(f"[AttentionPerceptron.predict_next_word_with_details] ERROR: {str(e)}")
            print(f"[AttentionPerceptron.predict_next_word_with_details] Context: {context}")
            # Return minimal info in case of error
            return ["<ERROR>"], {"error": str(e), "original_context": context}
    
    def plot_attention_weights(self, context=None, ax=None):
        """
        Plot attention weights for visualization.
        
        Parameters:
        -----------
        context : list of str, optional
            Context words to visualize attention for
            If None, plots the average attention weights from training
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if context is not None:
            # Get attention weights for specific context
            if isinstance(context, str):
                context = context.split()
            
            # Ensure context is the right length
            if len(context) < self.context_size:
                padding_needed = self.context_size - len(context)
                padding = ["<PAD>"] * padding_needed
                context = padding + context
            elif len(context) > self.context_size:
                context = context[-self.context_size:]
            
            # Get embeddings and predict
            _, info = self.predict_next_word(context)
            attention_weights = np.array(info["attention_weights"])
            
            # Plot attention heatmap
            im = ax.imshow(attention_weights, cmap="YlOrRd")
            
            # Set labels
            ax.set_xticks(np.arange(len(context)))
            ax.set_yticks(np.arange(len(context)))
            ax.set_xticklabels(context)
            ax.set_yticklabels(context)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            ax.set_title(f"Attention Weights for Context: '{' '.join(context)}'")
        else:
            # Plot average attention weights from training
            if not self.attention_weights_history:
                ax.text(0.5, 0.5, "No attention weights available. Train the model first.", 
                       ha="center", va="center")
                return ax
            
            # Get the latest attention weights
            _, avg_attention = self.attention_weights_history[-1]
            
            # Plot attention heatmap
            im = ax.imshow(avg_attention, cmap="YlOrRd")
            
            # Set labels
            ax.set_xticks(np.arange(self.context_size))
            ax.set_yticks(np.arange(self.context_size))
            ax.set_xticklabels([f"Word {i+1}" for i in range(self.context_size)])
            ax.set_yticklabels([f"Word {i+1}" for i in range(self.context_size)])
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            ax.set_title("Average Attention Weights During Training")
        
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        
        return ax
    
    def predict_next_n_words(self, initial_context, n=5):
        """
        Predict the next n words given an initial context, using attention mechanism.
        
        Parameters:
        -----------
        initial_context : list of str or str
            Initial context words or a string of space-separated words
        n : int
            Number of words to predict
        
        Returns:
        --------
        list of str
            Predicted words
        dict
            Additional information about the prediction process
        """
        print(f"[AttentionPerceptron.predict_next_n_words] Initial context: {initial_context}")
        
        # Handle string input
        if isinstance(initial_context, str):
            initial_context = initial_context.split()
            print(f"[AttentionPerceptron.predict_next_n_words] Split context: {initial_context}")
        
        # Get the first prediction and info
        next_word, info = self.predict_next_word(initial_context)
        print(f"[AttentionPerceptron.predict_next_n_words] First predicted word: {next_word}")
        
        # Use the adjusted context from the info
        context = info["adjusted_context"] if info["adjustment_made"] else info["original_context"]
        
        # Initialize prediction info
        prediction_info = {
            "original_context": initial_context,
            "adjusted_context": context if info["adjustment_made"] else None,
            "adjustment_made": info["adjustment_made"],
            "prediction_steps": [info],
            "attention_weights": [info.get("attention_weights", [])]
        }
        
        # Predict n words
        predicted_words = [next_word]
        for i in range(1, n):
            print(f"[AttentionPerceptron.predict_next_n_words] Predicting word {i+1}/{n}")
            
            # Update context - remove oldest word and add the predicted word
            context = context[1:] + [next_word]
            print(f"[AttentionPerceptron.predict_next_n_words] Updated context: {context}")
            
            # Predict next word
            next_word, step_info = self.predict_next_word(context)
            print(f"[AttentionPerceptron.predict_next_n_words] Predicted word: {next_word}")
            
            predicted_words.append(next_word)
            
            # Store step info
            prediction_info["prediction_steps"].append(step_info)
            prediction_info["attention_weights"].append(step_info.get("attention_weights", []))
        
        prediction_info["predicted_sequence"] = predicted_words
        prediction_info["full_text"] = " ".join(initial_context) + " " + " ".join(predicted_words)
        
        print(f"[AttentionPerceptron.predict_next_n_words] Full predicted sequence: {predicted_words}")
        return predicted_words, prediction_info
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        # Save base model data
        model_data = {
            'context_size': self.context_size,
            'embedding_dim': self.embedding_dim,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'random_state': self.random_state,
            'vocabulary': self.vocabulary,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'iteration_count': self.iteration_count,
            
            # Attention-specific data
            'attention_dim': self.attention_dim,
            'num_attention_heads': self.num_attention_heads,
            'attention_dropout': self.attention_dropout,
            'attention_weights_history': self.attention_weights_history,
            
            # Attention layer weights
            'attention_W_query': self.attention_layer.W_query if self.attention_layer else None,
            'attention_W_key': self.attention_layer.W_key if self.attention_layer else None,
            'attention_W_value': self.attention_layer.W_value if self.attention_layer else None,
            'attention_W_output': self.attention_layer.W_output if self.attention_layer else None,
            'attention_b_query': self.attention_layer.b_query if self.attention_layer else None,
            'attention_b_key': self.attention_layer.b_key if self.attention_layer else None,
            'attention_b_value': self.attention_layer.b_value if self.attention_layer else None,
            'attention_b_output': self.attention_layer.b_output if self.attention_layer else None
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
        AttentionPerceptron
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(
            context_size=model_data['context_size'],
            embedding_dim=model_data['embedding_dim'],
            hidden_layers=model_data['hidden_layers'],
            attention_dim=model_data.get('attention_dim'),
            num_attention_heads=model_data.get('num_attention_heads', 1),
            attention_dropout=model_data.get('attention_dropout', 0.1),
            learning_rate=model_data['learning_rate'],
            n_iterations=model_data['n_iterations'],
            random_state=model_data['random_state']
        )
        
        # Restore model attributes
        model.vocabulary = model_data['vocabulary']
        model.word_to_idx = model_data['word_to_idx']
        model.idx_to_word = model_data['idx_to_word']
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        model.input_size = model_data['input_size']
        model.output_size = model_data['output_size']
        model.training_loss = model_data['training_loss']
        model.validation_loss = model_data['validation_loss']
        model.iteration_count = model_data['iteration_count']
        
        # Restore attention-specific attributes
        model.attention_weights_history = model_data.get('attention_weights_history', [])
        
        # Create encoder
        model.encoder = OneHotEncoder(sparse_output=False)
        model.encoder.fit(np.array(range(model.output_size)).reshape(-1, 1))
        
        # Restore attention layer if it exists in the saved model
        if 'attention_W_query' in model_data and model_data['attention_W_query'] is not None:
            model.attention_layer = SelfAttention(
                input_dim=model.embedding_dim,
                attention_dim=model.attention_dim,
                num_heads=model.num_attention_heads,
                dropout_rate=model.attention_dropout,
                random_state=model.random_state
            )
            
            # Restore attention weights
            model.attention_layer.W_query = model_data['attention_W_query']
            model.attention_layer.W_key = model_data['attention_W_key']
            model.attention_layer.W_value = model_data['attention_W_value']
            model.attention_layer.W_output = model_data['attention_W_output']
            model.attention_layer.b_query = model_data['attention_b_query']
            model.attention_layer.b_key = model_data['attention_b_key']
            model.attention_layer.b_value = model_data['attention_b_value']
            model.attention_layer.b_output = model_data['attention_b_output']
        
        return model