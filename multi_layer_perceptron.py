import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
import pickle
import os
from embeddings import WordEmbeddings
from custom_tokenizers import BPETokenizer, WordPieceTokenizer

class MultiLayerPerceptron:
    """
    A multi-layer perceptron implementation for language modeling.
    """
    
    def __init__(self, context_size=2, embedding_dim=50, hidden_layers=[64, 32], learning_rate=0.01, n_iterations=1000, random_state=42, tokenizer_type='wordpiece', vocab_size=10000, use_pretrained=False):
        """
        Initialize the multi-layer perceptron language model.
        
        Parameters:
        -----------
        context_size : int
            Number of previous words to use as context for prediction
        embedding_dim : int
            Dimensionality of word embeddings
        hidden_layers : list of int
            List of hidden layer sizes (each int represents a layer with that many neurons)
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
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.vocabulary = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.embeddings = None
        self.weights = []  # List to store weights for each layer
        self.biases = []   # List to store biases for each layer
        self.input_size = None
        self.output_size = None
        self.training_loss = []
        self.validation_loss = []
        self.iteration_count = []
        self.encoder = None  # One-hot encoder for target words
        self.tokenizer_type = tokenizer_type.lower()
        self.vocab_size = vocab_size
        self.use_pretrained = use_pretrained
        self.tokenizer = None
        
        # Initialize tokenizer based on type
        if self.tokenizer_type == 'bpe':
            self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)
        else:  # default to wordpiece
            self.tokenizer = WordPieceTokenizer(vocab_size=self.vocab_size)
        
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
        Build the vocabulary from the list of words and initialize embeddings.
        Uses tokenizer to handle out-of-vocabulary words.
        """
        # Train the tokenizer on the words
        if self.tokenizer:
            # Convert words list to text for tokenizer training
            text = " ".join(words)
            self.tokenizer.fit(text)
        
        # Initialize word embeddings with pretrained embeddings if specified
        self.embeddings = WordEmbeddings(
            embedding_dim=self.embedding_dim, 
            random_state=self.random_state,
            use_pretrained=self.use_pretrained,
            pretrained_source='glove'  # Default to GloVe embeddings
        )
        
        # Build vocabulary and initialize embeddings
        self.embeddings.build_vocabulary(words)
        
        # Get vocabulary mappings
        self.word_to_idx = self.embeddings.word_to_idx
        self.idx_to_word = self.embeddings.idx_to_word
        
        # Store vocabulary
        self.vocabulary = self.embeddings.vocabulary
        self.output_size = len(self.vocabulary) + len(self.embeddings.special_tokens)
        
        # Initialize one-hot encoder for target words
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.array(range(self.output_size)).reshape(-1, 1))
    
    def _create_training_data(self, words):
        """
        Create training data from the list of words.
        Each input is context_size words, and the target is the next word.
        Uses word embeddings instead of one-hot encoding.
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
            
            # Get embeddings for context words and concatenate
            context_vector = self.embeddings.get_embeddings_for_context(context)
            
            X.append(context_vector)
            y.append(target_index)
        
        return np.array(X), np.array(y)
    
    def _relu(self, x):
        """
        ReLU activation function.
        """
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """
        Derivative of ReLU activation function.
        """
        return np.where(x > 0, 1, 0)
    
    def fit_labeled_data(self, X, y, progress_callback=None, stop_event=None):
        """
        Train the model on labeled data.
        
        Parameters:
        -----------
        X : array-like
            Input data
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
        if not hasattr(self, 'weights') or not self.weights:
            self._initialize_weights(X_train.shape[1])
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Check if training should be stopped
            if stop_event and stop_event.is_set():
                if progress_callback:
                    progress_callback(iteration, self.n_iterations, 0, 0, "Training cancelled")
                break
            
            # Forward pass
            activations, pre_activations = self._forward(X_train)
            y_pred = activations[-1]
            
            # Compute loss
            train_loss = self._cross_entropy_loss(y_pred, y_train)
            
            # Compute validation loss
            if len(X_val) > 0:
                val_activations, _ = self._forward(X_val)
                val_pred = val_activations[-1]
                val_loss = self._cross_entropy_loss(val_pred, y_val)
            else:
                val_loss = train_loss
            
            # Store loss history
            self.training_loss.append(train_loss)
            self.validation_loss.append(val_loss)
            self.iteration_count.append(iteration)
            
            # Backward pass
            self._backward(X_train, y_train, activations, pre_activations)
            
            # Report progress
            if progress_callback and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                progress_callback(iteration + 1, self.n_iterations, train_loss, val_loss)
        
        # Final progress report
        if progress_callback:
            progress_callback(self.n_iterations, self.n_iterations, train_loss, val_loss, "Training complete")
    
    def prepare_labeled_data(self, inputs, targets):
        """
        Prepare labeled data for training.
        
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
            
            # Flatten embeddings
            X.append(np.concatenate(input_embeddings))
            
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
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        tuple
            (activations, pre_activations) for each layer
        """
        activations = [X]  # List to store activations for each layer
        pre_activations = []  # List to store pre-activation values
        
        # Hidden layers with ReLU activation
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            a = self._relu(z)
            activations.append(a)
        
        # Output layer with softmax activation
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z_out)
        a_out = self._softmax(z_out)
        activations.append(a_out)
        
        return activations, pre_activations
    
    def _cross_entropy_loss(self, y_pred, y_true):
        """
        Calculate the cross-entropy loss.
        """
        # Create one-hot vectors for true labels
        y_true_one_hot = np.zeros((len(y_true), self.output_size))
        for i, label in enumerate(y_true):
            y_true_one_hot[i, label] = 1
        
        # Calculate cross-entropy loss
        loss = -np.sum(y_true_one_hot * np.log(y_pred + 1e-10)) / len(y_true)
        
        return loss
    
    def _train_with_custom_data(self, X, y, progress_callback=None, stop_event=None):
        """
        Train the model with custom data (for labeled data training).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data, shape depends on model type
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
        
        # Initialize weights and biases if not already initialized
        if not self.weights:
            np.random.seed(self.random_state)
            self.input_size = X.shape[1]
            
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
            activations, pre_activations = self._forward(X_train)
            y_pred = activations[-1]
            
            # Calculate loss
            loss = self._cross_entropy_loss(y_pred, y_train)
            
            # Calculate validation loss
            val_activations, val_pre_activations = self._forward(X_val)
            val_pred = val_activations[-1]
            val_loss = self._cross_entropy_loss(val_pred, y_val)
            
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
                # Save best weights and biases
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback(iteration, self.n_iterations, loss, val_loss, 
                                     f"Early stopping at iteration {iteration}")
                # Restore best weights and biases
                self.weights = best_weights
                self.biases = best_biases
                break
            
            # Backward pass (backpropagation)
            # Initialize gradients
            dW = [np.zeros_like(w) for w in self.weights]
            db = [np.zeros_like(b) for b in self.biases]
            
            # Output layer error
            delta = y_pred - y_train_one_hot
            
            # Update gradients for output layer
            dW[-1] = np.dot(activations[-2].T, delta) / len(X_train)
            db[-1] = np.sum(delta, axis=0) / len(X_train)
            
            # Backpropagate error through hidden layers
            for l in range(len(self.weights) - 2, -1, -1):
                delta = np.dot(delta, self.weights[l+1].T) * self._relu_derivative(pre_activations[l])
                dW[l] = np.dot(activations[l].T, delta) / len(X_train)
                db[l] = np.sum(delta, axis=0) / len(X_train)
            
            # Update weights and biases
            for l in range(len(self.weights)):
                self.weights[l] -= current_lr * dW[l]
                self.biases[l] -= current_lr * db[l]
            
            # Shuffle the training data every 50 iterations
            if iteration % 50 == 0:
                shuffle_idx = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
                y_train_one_hot = y_train_one_hot[shuffle_idx]
        
        # Final progress report
        if progress_callback:
            progress_callback(iteration + 1, self.n_iterations, loss, val_loss, "Training complete")
    
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
        try:
            # Log sta

            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, "Starting preprocessing...")
            
            # Preprocess the text
            words = self._preprocess_text(text)
            print(f'Preprocessed text contains {len(words)} words')
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, f"Preprocessed text contains {len(words)} words")
            
            # Build vocabulary
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, "Building vocabulary...")
            self._build_vocabulary(words)

            print(f'Vocabulary size: {len(self.vocabulary)}')
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, f"Vocabulary size: {len(self.vocabulary)}")

            # Create training data
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, "Creating training data...")
            X, y = self._create_training_data(words)

            print(f'Created training data with {X.shape[0]} samples and {X.shape[1]} features')
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, 
                                f"Created training data with {X.shape[0]} samples and {X.shape[1]} features")

            # Split into training and validation sets
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, "Splitting into training and validation sets...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )

            print(f'Split into training ({X_train.shape[0]}) and validation ({X_val.shape[0]}) sets')
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, 
                                f"Split into training ({X_train.shape[0]}) and validation ({X_val.shape[0]}) sets")
            
            # Initialize weights and biases
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, "Initializing model parameters...")
            np.random.seed(self.random_state)
            self.input_size = X.shape[1]
            
            # Define layer sizes
            layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

            print(f'Layer sizes: {layer_sizes}')
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
                if progress_callback:
                    progress_callback(0, self.n_iterations, 0, 0, 
                                    f"Initialized layer {i+1}/{len(layer_sizes)-1}: {layer_sizes[i]} → {layer_sizes[i+1]}")
            
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
            
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, "Starting training iterations...")
            
            # Convert y_train to one-hot encoding for backpropagation
            y_train_one_hot = self.encoder.transform(y_train.reshape(-1, 1))
            
            for iteration in range(self.n_iterations):
                # Check if training should be stopped
                if stop_event and stop_event.is_set():
                    if progress_callback:
                        progress_callback(iteration, self.n_iterations, 
                                        self.training_loss[-1] if self.training_loss else 0, 
                                        self.validation_loss[-1] if self.validation_loss else 0, 
                                        "Training stopped by user")
                    break
                    
                # Adaptive learning rate - decrease over time
                current_lr = initial_learning_rate / (1 + iteration / 200)
                
                # Forward pass
                activations, pre_activations = self._forward(X_train)
                y_pred = activations[-1]
                
                # Calculate loss
                loss = self._cross_entropy_loss(y_pred, y_train)
                
                # Calculate validation loss
                val_activations, _ = self._forward(X_val)
                val_pred = val_activations[-1]
                val_loss = self._cross_entropy_loss(val_pred, y_val)
                
                # Store losses
                self.training_loss.append(loss)
                self.validation_loss.append(val_loss)
                self.iteration_count.append(iteration)
                
                # Report progress - log every iteration for more detailed feedback
                if progress_callback:
                    # Log more frequently for better visibility
                    if iteration % 10 == 0 or iteration < 10 or iteration == self.n_iterations - 1:
                        message = f"Iteration {iteration+1}/{self.n_iterations} - LR: {current_lr:.6f}"
                        progress_callback(iteration, self.n_iterations, loss, val_loss, message)
                
                # Backward pass (backpropagation)
                # Initialize gradients
                dW = [np.zeros_like(w) for w in self.weights]
                db = [np.zeros_like(b) for b in self.biases]
                
                # Output layer error
                delta = y_pred - y_train_one_hot  # shape: (batch_size, output_size)
                
                # Update gradients for output layer
                dW[-1] = np.dot(activations[-2].T, delta) / len(X_train)
                db[-1] = np.sum(delta, axis=0) / len(X_train)
                
                # Backpropagate error through hidden layers
                for l in range(len(self.weights) - 2, -1, -1):
                    # Compute error for current layer
                    delta = np.dot(delta, self.weights[l+1].T) * self._relu_derivative(pre_activations[l])
                    
                    # Update gradients
                    dW[l] = np.dot(activations[l].T, delta) / len(X_train)
                    db[l] = np.sum(delta, axis=0) / len(X_train)
                
                # Add L2 regularization to prevent overfitting
                reg_lambda = 0.001
                for i in range(len(self.weights)):
                    dW[i] += reg_lambda * self.weights[i]
                
                # Update weights and biases with current learning rate
                for i in range(len(self.weights)):
                    self.weights[i] -= current_lr * dW[i]
                    self.biases[i] -= current_lr * db[i]
                
                # Shuffle the training data every 50 iterations
                if iteration % 50 == 0:
                    shuffle_idx = np.random.permutation(len(X_train))
                    X_train = X_train[shuffle_idx]
                    y_train = y_train[shuffle_idx]
                    # Update one-hot encoded targets after shuffling
                    y_train_one_hot = self.encoder.transform(y_train.reshape(-1, 1))
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights and biases
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    
                    if progress_callback:
                        progress_callback(iteration, self.n_iterations, loss, val_loss, 
                                        f"New best validation loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    
                    if progress_callback and patience_counter % 10 == 0:
                        progress_callback(iteration, self.n_iterations, loss, val_loss, 
                                        f"No improvement for {patience_counter} iterations. Best val_loss: {best_val_loss:.6f}")
                    
                if patience_counter >= patience:
                    if progress_callback:
                        progress_callback(iteration, self.n_iterations, loss, val_loss, 
                                         f"Early stopping at iteration {iteration}")
                    # Restore best weights and biases
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            # Final progress report
            if progress_callback:
                progress_callback(iteration + 1, self.n_iterations, 
                                 self.training_loss[-1] if self.training_loss else 0, 
                                 self.validation_loss[-1] if self.validation_loss else 0, 
                                 "Training complete")
                
        except Exception as e:
            # Log any exceptions that occur during training
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if progress_callback:
                progress_callback(0, self.n_iterations, 0, 0, f"Error during training: {str(e)}")
    
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
        try:
            # Handle string input
            if isinstance(context, str):
                context = context.split()
            
            # Preprocess context
            context = [word.lower() for word in context]
            
            # Handle context length mismatch
            info = {
                "original_context": context.copy(), 
                "adjusted_context": None, 
                "adjustment_made": False,
                "adjustment_type": [],
                "confidence_score": 0.0
            }
            
            if len(context) < self.context_size:
                # If context is too short, pad with common words from vocabulary
                # Use the most frequent words in the vocabulary as padding
                # For simplicity, we'll use the first words in the vocabulary
                padding_needed = self.context_size - len(context)
                padding = ["the"] * padding_needed  # Use "the" as default padding
                context = padding + context
                info["adjusted_context"] = context
                info["adjustment_made"] = True
                info["adjustment_type"].append("padded_beginning")
                
            elif len(context) > self.context_size:
                # If context is too long, use the most recent words
                context = context[-self.context_size:]
                info["adjusted_context"] = context
                info["adjustment_made"] = True
                info["adjustment_type"].append("truncated_beginning")
            
            # Check if all words are in vocabulary
            unknown_words = []
            for i, word in enumerate(context):
                if word not in self.word_to_idx:
                    unknown_words.append((i, word))
            
            # Handle unknown words
            if unknown_words:
                info["unknown_words"] = [word for _, word in unknown_words]
                info["adjustment_made"] = True
                info["adjustment_type"].append("replaced_unknown")
                
                for idx, word in unknown_words:
                    # Try to tokenize the unknown word if we have a tokenizer
                    if self.tokenizer:
                        try:
                            # Tokenize the word
                            subwords = self.tokenizer.tokenize(word)
                            
                            # If we got valid subwords, use the first one that's in our vocabulary
                            # or use <UNK> token if none are found
                            found_replacement = False
                            for subword in subwords:
                                if subword in self.word_to_idx:
                                    context[idx] = subword
                                    found_replacement = True
                                    info["replacements"] = info.get("replacements", []) + [(word, subword)]
                                    break
                            
                            # If no valid subwords found, use <UNK> token
                            if not found_replacement:
                                unk_token = '<UNK>'  # Use explicit <UNK> token
                                context[idx] = unk_token
                                info["replacements"] = info.get("replacements", []) + [(word, unk_token)]
                        except Exception as e:
                            unk_token = '<UNK>'  # Use explicit <UNK> token
                            context[idx] = unk_token
                            info["replacements"] = info.get("replacements", []) + [(word, unk_token)]
                            info["tokenization_errors"] = info.get("tokenization_errors", []) + [(word, str(e))]
                    else:
                        # If no tokenizer, use <UNK> token
                        unk_token = '<UNK>'  # Use explicit <UNK> token
                        context[idx] = unk_token
                        info["replacements"] = info.get("replacements", []) + [(word, unk_token)]
            
            # Update adjusted context if any adjustments were made
            if info["adjustment_made"] and info["adjusted_context"] is None:
                info["adjusted_context"] = context
            
            # Convert context to indices, handling any special tokens
            context_indices = []
            for word in context:
                if word in self.word_to_idx:
                    context_indices.append(self.word_to_idx[word])
                elif word in self.embeddings.special_tokens:
                    context_indices.append(self.embeddings.special_tokens[word])
                else:
                    # Fallback to <UNK> token
                    context_indices.append(self.embeddings.special_tokens['<UNK>'])
            
            # Get embeddings for context words and concatenate
            try:
                context_vector = self.embeddings.get_embeddings_for_context(context)
                
                # Forward pass
                activations, _ = self._forward(context_vector.reshape(1, -1))
                y_pred = activations[-1][0]
                
                # Get the top predictions with probabilities
                top_n = 10  # Increase from 5 to 10 for more options
                top_indices = np.argsort(y_pred)[-top_n:][::-1]
                top_probs = y_pred[top_indices]
                top_words = [self.idx_to_word[idx] for idx in top_indices]
                
                # Get the word with the highest probability
                predicted_idx = top_indices[0]
                predicted_word = self.idx_to_word[predicted_idx]
                predicted_prob = top_probs[0]
                
                # Add prediction info
                info["prediction"] = predicted_word
                info["confidence_score"] = float(predicted_prob)  # Convert to native Python float
                info["top_predictions"] = [(word, float(prob)) for word, prob in zip(top_words, top_probs)]
                info["prediction_entropy"] = float(-np.sum(top_probs * np.log2(top_probs + 1e-10)))
                
                return predicted_word, info
            except Exception as e:
                # Return a fallback prediction with detailed error info
                info["error"] = str(e)
                info["error_type"] = "prediction_calculation_error"
                info["prediction"] = "the"  # Fallback to common word
                info["confidence_score"] = 0.0
                info["top_predictions"] = [("the", 1.0)]
                return "the", info
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # Return a fallback prediction with detailed error info
            info = {
                "original_context": context if isinstance(context, list) else context.split() if isinstance(context, str) else [],
                "adjusted_context": None,
                "adjustment_made": False,
                "error": str(e),
                "error_type": "general_prediction_error",
                "error_traceback": error_traceback,
                "prediction": "the",  # Fallback to common word
                "confidence_score": 0.0,
                "top_predictions": [("the", 1.0)]
            }
            return "the", info
    
    def predict_next_n_words(self, initial_context, n=5, temperature=1.0):
        """
        Predict the next n words given an initial context.
        
        Parameters:
        -----------
        initial_context : list of str or str
            Initial context words or a string of space-separated words
        n : int
            Number of words to predict
        temperature : float
            Controls randomness in prediction. Higher values (e.g., 1.5) increase diversity,
            lower values (e.g., 0.5) make predictions more deterministic.
            Default is 1.0 (standard prediction).
        
        Returns:
        --------
        list of str
            Predicted words
        dict
            Additional information about the prediction process
        """
        # Handle string input
        if isinstance(initial_context, str):
            initial_context = initial_context.split()
        
        # Preprocess context words to lowercase
        initial_context = [word.lower() for word in initial_context]
        
        # Get the first prediction and info
        next_word, info = self.predict_next_word(initial_context)
        
        # Use the adjusted context from the info
        context = info["adjusted_context"] if info["adjustment_made"] else info["original_context"]
        
        # Initialize prediction info with more detailed metadata
        prediction_info = {
            "original_context": initial_context,
            "adjusted_context": context if info["adjustment_made"] else None,
            "adjustment_made": info["adjustment_made"],
            "prediction_steps": [info],
            "generation_params": {
                "temperature": temperature,
                "requested_words": n,
                "model_context_size": self.context_size,
                "embedding_dim": self.embedding_dim
            },
            "generation_stats": {
                "successful_predictions": 1,  # Start with 1 for the first word
                "errors": 0,
                "avg_confidence": info.get("confidence_score", 0.0)
            }
        }
        
        # Predict n words
        predicted_words = [next_word]
        confidence_scores = [info.get("confidence_score", 0.0)]
        
        for i in range(1, n):
            try:
                # Update context - remove oldest word and add the predicted word
                context = context[1:] + [next_word]
                
                # Predict next word
                next_word, step_info = self.predict_next_word(context)
                predicted_words.append(next_word)
                confidence_scores.append(step_info.get("confidence_score", 0.0))
                
                # Store step info
                prediction_info["prediction_steps"].append(step_info)
                
                # Update generation stats
                prediction_info["generation_stats"]["successful_predictions"] += 1
                prediction_info["generation_stats"]["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
                
            except Exception as e:
                # Log the error but continue with what we have
                error_info = {
                    "step": i,
                    "error": str(e),
                    "context_at_error": context
                }
                prediction_info["generation_stats"]["errors"] += 1
                prediction_info["error_info"] = prediction_info.get("error_info", []) + [error_info]
                break
        
        # Add final generation results
        prediction_info["predicted_sequence"] = predicted_words
        prediction_info["full_text"] = " ".join(initial_context) + " " + " ".join(predicted_words)
        prediction_info["generation_stats"]["completion_rate"] = prediction_info["generation_stats"]["successful_predictions"] / n
        prediction_info["generation_stats"]["confidence_scores"] = confidence_scores
        
        return predicted_words, prediction_info
    
    def get_top_predictions(self, context, top_n=5, include_details=True):
        """
        Get the top n predicted words with their probabilities.
        
        Parameters:
        -----------
        context : list of str or str
            List of context words or a string of space-separated words
        top_n : int
            Number of top predictions to return
        include_details : bool
            Whether to include detailed prediction information in the result
        
        Returns:
        --------
        list of tuples
            List of (word, probability) tuples
        dict
            Additional information about the prediction process
        """
        try:
            # First get the prediction and info using the predict_next_word method
            _, info = self.predict_next_word(context)
            
            # Use the adjusted context from the info
            context = info["adjusted_context"] if info["adjustment_made"] else info["original_context"]
            
            # Get embeddings for context words and concatenate
            context_vector = self.embeddings.get_embeddings_for_context(context)
            
            # Forward pass
            activations, _ = self._forward(context_vector.reshape(1, -1))
            y_pred = activations[-1][0]
            
            # Get the top n predictions
            top_indices = np.argsort(y_pred)[-top_n:][::-1]
            top_probs = y_pred[top_indices]
            
            # Convert to words
            top_words = [self.idx_to_word[idx] for idx in top_indices]
            
            # Calculate entropy (measure of prediction uncertainty)
            normalized_probs = top_probs / np.sum(top_probs)
            entropy = -np.sum(normalized_probs * np.log2(normalized_probs + 1e-10))
            
            # Add enhanced prediction information
            prediction_details = {
                "top_predictions": [(word, float(prob)) for word, prob in zip(top_words, top_probs)],
                "entropy": float(entropy),
                "normalized_probabilities": [float(prob) for prob in normalized_probs],
                "probability_distribution": {
                    "max": float(np.max(top_probs)),
                    "min": float(np.min(top_probs)),
                    "mean": float(np.mean(top_probs)),
                    "std": float(np.std(top_probs))
                }
            }
            
            # Add detailed prediction info if requested
            if include_details:
                info.update(prediction_details)
            else:
                # Just include the basic top predictions
                info["top_predictions"] = prediction_details["top_predictions"]
            
            return [(word, float(prob)) for word, prob in zip(top_words, top_probs)], info
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # Return fallback predictions with error info
            fallback_predictions = [("the", 0.5), ("a", 0.3), ("an", 0.2)]
            error_info = {
                "error": str(e),
                "error_type": "top_predictions_error",
                "error_traceback": error_traceback,
                "top_predictions": fallback_predictions[:top_n]
            }
            
            return fallback_predictions[:top_n], error_info
    
    def plot_training_loss(self):
        """
        Plot the training loss over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.iteration_count, self.training_loss, label='Training Loss')
        plt.plot(self.iteration_count, self.validation_loss, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        return plt
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'context_size': self.context_size,
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
            'embedding_dim': self.embedding_dim,
            'use_pretrained': self.use_pretrained,
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size
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
        MultiLayerPerceptron
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(
            context_size=model_data['context_size'],
            hidden_layers=model_data['hidden_layers'],
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
        
        # Create encoder
        model.encoder = OneHotEncoder(sparse_output=False)
        model.encoder.fit(np.array(range(model.output_size)).reshape(-1, 1))
        
        # Initialize embeddings
        model.embeddings = WordEmbeddings(
            embedding_dim=model_data.get('embedding_dim', 50),
            random_state=model.random_state,
            use_pretrained=model_data.get('use_pretrained', False)
        )
        model.embeddings.word_to_idx = model.word_to_idx
        model.embeddings.idx_to_word = model.idx_to_word
        model.embeddings.vocabulary = model.vocabulary
        model.embeddings.special_tokens = {'<UNK>': 0, '<PAD>': 1, '<BOS>': 2, '<EOS>': 3}
        
        # Initialize tokenizer
        tokenizer_type = model_data.get('tokenizer_type', 'wordpiece').lower()
        vocab_size = model_data.get('vocab_size', 10000)
        if tokenizer_type == 'bpe':
            model.tokenizer = BPETokenizer(vocab_size=vocab_size)
        else:  # default to wordpiece
            model.tokenizer = WordPieceTokenizer(vocab_size=vocab_size)
        
        return model