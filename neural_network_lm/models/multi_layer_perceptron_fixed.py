"""
Multi-Layer Perceptron Implementation for Language Modeling

This module provides an implementation of a multi-layer perceptron
specifically designed for language modeling tasks.
"""

import numpy as np
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ..utils.embeddings import WordEmbeddings
from ..tokenizers.custom_tokenizers import BPETokenizer, WordPieceTokenizer

class MultiLayerPerceptron:
    """
    A multi-layer perceptron implementation for language modeling.
    
    This model uses word embeddings and multiple hidden layers to predict
    the next word in a sequence based on context words.
    """
    
    def __init__(self, context_size=2, embedding_dim=50, hidden_layers=[64, 32], 
                 learning_rate=0.01, n_iterations=1000, random_state=42, 
                 tokenizer_type='wordpiece', vocab_size=10000, use_pretrained=False):
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
        self.weights = []
        self.biases = []
        self.input_size = None
        self.output_size = None
        self.training_loss = []
        self.validation_loss = []
        self.iteration_count = []
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.use_pretrained = use_pretrained
        self.tokenizer = None
        self.encoder = None
        
        # Initialize embeddings
        self.embeddings = WordEmbeddings(
            embedding_dim=self.embedding_dim,
            random_state=self.random_state,
            use_pretrained=self.use_pretrained
        )
        
        # Initialize tokenizer
        if tokenizer_type.lower() == 'bpe':
            self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)
        else:  # default to wordpiece
            self.tokenizer = WordPieceTokenizer(vocab_size=self.vocab_size)
        
    def _preprocess_text(self, text):
        """
        Preprocess the text by converting to lowercase, removing special characters,
        and splitting into words.
        
        Parameters:
        -----------
        text : str
            The text to preprocess
            
        Returns:
        --------
        list of str
            List of preprocessed words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace newlines with spaces
        text = text.replace('\n', ' ')
        
        # Remove special characters and digits, keeping only letters, spaces, and basic punctuation
        text = re.sub(r'[^a-z\s.,!?]', '', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Split into words
        words = text.split()
        
        return words
    
    def _build_vocabulary(self, words):
        """
        Build vocabulary from the list of words.
        
        Parameters:
        -----------
        words : list of str
            List of words from the training text
        """
        # Initialize word embeddings
        self.embeddings = WordEmbeddings(
            embedding_dim=self.embedding_dim,
            random_state=self.random_state,
            use_pretrained=self.use_pretrained
        )
        
        # Initialize tokenizer
        if self.tokenizer_type.lower() == 'bpe':
            self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)
        else:  # default to wordpiece
            self.tokenizer = WordPieceTokenizer(vocab_size=self.vocab_size)
        
        # Train tokenizer on words
        self.tokenizer.train(words)
        
        # Get vocabulary from tokenizer
        self.vocabulary = self.tokenizer.get_vocabulary()
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Set output size to vocabulary size
        self.output_size = len(self.vocabulary)
        
        # Initialize one-hot encoder for target words
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.array(range(self.output_size)).reshape(-1, 1))
        
        # Initialize embeddings for all words in vocabulary
        for word in self.vocabulary:
            self.embeddings.add_word(word)
    
    def _create_training_data(self, words):
        """
        Create training data from the list of words.
        Each input is context_size words, and the target is the next word.
        
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
            
            # Get embeddings for context words
            context_embeddings = []
            for word in context:
                word_idx = self.word_to_idx.get(word, self.embeddings.special_tokens['<UNK>'])
                context_embeddings.append(self.embeddings.embeddings[word_idx])
            
            # Flatten context embeddings
            flattened_embeddings = np.concatenate(context_embeddings)
            
            X.append(flattened_embeddings)
            y.append(target_index)
        
        # Convert to numpy arrays
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
        
        # Input size is context_size * embedding_dim
        self.input_size = self.context_size * self.embedding_dim
        
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
            val_activations, _ = self._forward(X_val)
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
            # Convert y_train to one-hot encoding
            y_train_one_hot = self.encoder.transform(y_train.reshape(-1, 1))
            
            # Compute gradients and update weights
            self._backward(X_train, activations, pre_activations, y_train_one_hot, current_lr)
            
            # Shuffle the training data every 50 iterations
            if iteration % 50 == 0:
                shuffle_idx = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
        
        # Final progress report
        if progress_callback:
            progress_callback(iteration + 1, self.n_iterations, loss, val_loss, "Training complete")
    
    def _forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        tuple
            (activations, pre_activations)
        """
        activations = [X]  # List to store activations of each layer
        pre_activations = []  # List to store pre-activations (before applying activation function)
        
        # Hidden layers with ReLU activation
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            a = self._relu(z)
            activations.append(a)
        
        # Output layer with softmax activation
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z_out)
        output = self._softmax(z_out)
        activations.append(output)
        
        return activations, pre_activations
    
    def _backward(self, X, activations, pre_activations, y_true, learning_rate):
        """
        Backward pass through the network (backpropagation).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        activations : list of numpy.ndarray
            Activations from forward pass
        pre_activations : list of numpy.ndarray
            Pre-activations from forward pass
        y_true : numpy.ndarray
            True labels (one-hot encoded)
        learning_rate : float
            Learning rate for weight updates
        """
        batch_size = X.shape[0]
        
        # Calculate output layer error (derivative of cross-entropy with softmax)
        delta = activations[-1] - y_true  # Shape: (batch_size, output_size)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, 0, -1):
            # Calculate gradients for current layer
            dW = np.dot(activations[i].T, delta) / batch_size
            db = np.sum(delta, axis=0) / batch_size
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Calculate error for previous layer
            delta = np.dot(delta, self.weights[i].T)
            
            # Apply derivative of ReLU
            delta = delta * self._relu_derivative(pre_activations[i-1])
        
        # Update first layer weights and biases
        dW = np.dot(X.T, delta) / batch_size
        db = np.sum(delta, axis=0) / batch_size
        self.weights[0] -= learning_rate * dW
        self.biases[0] -= learning_rate * db
    
    def _relu(self, x):
        """
        ReLU activation function.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input values
            
        Returns:
        --------
        numpy.ndarray
            Output values
        """
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """
        Derivative of ReLU activation function.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input values
            
        Returns:
        --------
        numpy.ndarray
            Derivative values
        """
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        """
        Softmax activation function with numerical stability.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input values
            
        Returns:
        --------
        numpy.ndarray
            Output probabilities
        """
        # Shift for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _cross_entropy_loss(self, y_pred, y_true):
        """
        Calculate cross-entropy loss.
        
        Parameters:
        -----------
        y_pred : numpy.ndarray
            Predicted probabilities
        y_true : numpy.ndarray
            True labels (indices)
            
        Returns:
        --------
        float
            Cross-entropy loss
        """
        batch_size = y_pred.shape[0]
        
        # Get predicted probabilities for true labels
        log_probs = -np.log(np.clip(y_pred[np.arange(batch_size), y_true], 1e-10, 1.0))
        
        # Calculate mean loss
        loss = np.mean(log_probs)
        
        return loss
    
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
            padding = ["<PAD>"] * padding_needed  # Use "<PAD>" as default padding
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
            if word not in self.word_to_idx and word not in self.embeddings.special_tokens:
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
                        context[idx] = "<UNK>"
                else:
                    # If no tokenizer, use <UNK> token
                    context[idx] = "<UNK>"
            
            info["unknown_words"] = [word for _, word in unknown_words]
            info["adjustment_made"] = True
            info["adjustment_type"] = info.get("adjustment_type", "") + "_replaced_unknown"
        
        try:
            # Check if model has been trained
            if not self.vocabulary or len(self.vocabulary) == 0:
                info["error"] = "Model has not been trained yet"
                return "<ERROR>", info
                
            # Check if embeddings exist
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                info["error"] = "Embeddings not initialized"
                return "<ERROR>", info
                
            # Get embeddings for context words
            context_embeddings = []
            for word in context:
                word_idx = self.word_to_idx.get(word, self.embeddings.special_tokens.get('<UNK>', 1))
                
                # Check if word_idx is valid
                if word_idx >= len(self.embeddings.embeddings):
                    info["error"] = f"Invalid word index: {word_idx}"
                    return "<ERROR>", info
                    
                embedding = self.embeddings.embeddings[word_idx]
                context_embeddings.append(embedding)
            
            # Flatten context embeddings
            flattened_embeddings = np.concatenate(context_embeddings)
            
            # Add batch dimension
            X = flattened_embeddings.reshape(1, -1)
            
            # Forward pass
            activations, _ = self._forward(X)
            y_pred = activations[-1]
            
            # Get the word with the highest probability
            predicted_idx = np.argmax(y_pred[0])
            predicted_word = self.idx_to_word[predicted_idx]
            
            # Get top 5 predictions with probabilities
            top_indices = np.argsort(y_pred[0])[-5:][::-1]
            top_probs = {self.idx_to_word[idx]: float(y_pred[0][idx]) for idx in top_indices}
            
            # Add prediction info
            info["prediction"] = predicted_word
            info["probabilities"] = top_probs
            
            return predicted_word, info
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in prediction: {error_details}")
            info["error"] = str(e)
            return "<ERROR>", info
    
    def _predict_next_word_with_temperature(self, context, temperature=1.0):
        """
        Predict the next word with temperature sampling.
        
        Parameters:
        -----------
        context : list of str
            Context words
        temperature : float
            Temperature parameter for controlling randomness
            
        Returns:
        --------
        tuple
            (predicted_word, prediction_info)
        """
        # Check if model has been trained
        if not self.vocabulary or len(self.vocabulary) == 0:
            return "<ERROR>", {"error": "Model has not been trained yet", "probabilities": {}}
            
        # Get prediction and info
        word, info = self.predict_next_word(context)
        
        # If there was an error, return it
        if word == "<ERROR>":
            return word, info
            
        # If temperature is close to 1.0, just return the most likely word
        if abs(temperature - 1.0) < 0.01:
            return word, info
            
        # Get probabilities
        if "probabilities" in info and info["probabilities"]:
            probs = []
            words = []
            
            # Extract words and probabilities
            for word, prob in info["probabilities"].items():
                words.append(word)
                probs.append(prob)
                
            # Apply temperature to probabilities
            probs = np.array(probs)
            if temperature != 1.0:
                # Apply temperature scaling
                probs = np.power(probs, 1.0 / temperature)
                # Renormalize
                probs = probs / np.sum(probs)
                
            # Sample from the distribution
            try:
                chosen_idx = np.random.choice(len(words), p=probs)
                chosen_word = words[chosen_idx]
                
                # Update info
                info["temperature"] = temperature
                info["sampling_method"] = "temperature"
                info["sampled_word"] = chosen_word
                
                return chosen_word, info
            except Exception as e:
                print(f"Error in temperature sampling: {str(e)}")
                # Fall back to the original prediction
                return word, info
        else:
            # If no probabilities available, return the original prediction
            return word, info
    
    def predict_next_n_words(self, context, n=5, temperature=1.0):
        """
        Predict the next n words given an initial context.
        
        Parameters:
        -----------
        context : list of str or str
            Initial context words
        n : int
            Number of words to predict
        temperature : float
            Temperature parameter for controlling randomness in sampling
            Higher values (e.g., 1.5) make the output more random
            Lower values (e.g., 0.5) make the output more deterministic
            
        Returns:
        --------
        tuple
            (predicted_words, prediction_info)
            predicted_words: list of str - the predicted words
            prediction_info: dict - detailed information about the prediction process
        """
        # Handle string input
        if isinstance(context, str):
            context = context.split()
            
        # Make a copy of the initial context
        current_context = context.copy()
        
        # Ensure context has the right length
        if len(current_context) < self.context_size:
            # Pad with PAD tokens
            padding_needed = self.context_size - len(current_context)
            padding = ["<PAD>"] * padding_needed
            current_context = padding + current_context
        elif len(current_context) > self.context_size:
            # Use the last context_size words
            current_context = current_context[-self.context_size:]
            
        # Store prediction info
        prediction_info = {
            "original_context": context,
            "adjusted_context": current_context,
            "temperature": temperature,
            "predictions": []
        }
        
        # Check if model has been trained
        if not self.vocabulary or len(self.vocabulary) == 0:
            prediction_info["error"] = "Model has not been trained yet. Please train the model first."
            
            # Generate some sample text instead of errors
            sample_words = ["the", "a", "is", "of", "and", "to", "in", "that", "it", "with", 
                           "for", "as", "on", "by", "at", "this", "from", "but", "not", "or"]
            
            # Use the input context to seed the generation
            seed_word = None
            for word in context:
                if word not in ["<PAD>", "<UNK>", "<ERROR>"]:
                    seed_word = word
                    break
            
            # Generate words with some randomness
            predicted_words = []
            for i in range(n):
                if seed_word and np.random.random() < 0.3:  # 30% chance to repeat a seed word
                    next_word = seed_word
                else:
                    next_word = np.random.choice(sample_words)
                
                predicted_words.append(next_word)
                
                # Store prediction info
                step_info = {
                    "step": i+1,
                    "context": current_context.copy(),
                    "predicted_word": next_word,
                    "note": "Random word (model not trained)"
                }
                
                prediction_info["predictions"].append(step_info)
                
                # Update context for next prediction
                current_context = current_context[1:] + [next_word]
            
            return predicted_words, prediction_info
        
        # Predict n words
        predicted_words = []
        error_count = 0  # Track consecutive errors
        
        for i in range(n):
            try:
                # If we've had too many consecutive errors, switch to fallback mode
                if error_count >= 3:
                    # Use common words as fallback
                    common_words = ["the", "a", "is", "of", "and", "to", "in", "that", "it", "with"]
                    next_word = np.random.choice(common_words)
                    word_info = {"probabilities": {next_word: 1.0}, "fallback": True}
                    error_count = 0  # Reset error count
                else:
                    # Predict next word with temperature
                    next_word, word_info = self._predict_next_word_with_temperature(current_context, temperature)
                
                # If we got an error, increment error count
                if next_word == "<ERROR>":
                    error_count += 1
                    # Try to use a word from the context instead
                    for word in current_context:
                        if word not in ["<PAD>", "<UNK>", "<ERROR>"]:
                            next_word = word
                            word_info["fallback"] = True
                            word_info["fallback_source"] = "context"
                            break
                else:
                    error_count = 0  # Reset error count on successful prediction
                
                predicted_words.append(next_word)
                
                # Store prediction info
                step_info = {
                    "step": i+1,
                    "context": current_context.copy(),
                    "predicted_word": next_word,
                }
                
                # Add probabilities if available
                if "probabilities" in word_info:
                    step_info["probabilities"] = word_info["probabilities"]
                
                # Add error if there was one
                if "error" in word_info:
                    step_info["error"] = word_info["error"]
                
                # Add fallback info if applicable
                if "fallback" in word_info and word_info["fallback"]:
                    step_info["fallback"] = True
                    if "fallback_source" in word_info:
                        step_info["fallback_source"] = word_info["fallback_source"]
                
                prediction_info["predictions"].append(step_info)
                
                # Update context for next prediction
                current_context = current_context[1:] + [next_word]
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error in prediction step {i+1}: {error_details}")
                
                # Use a fallback word instead of ERROR
                fallback_words = ["the", "a", "is", "of", "and", "to", "in"]
                next_word = np.random.choice(fallback_words)
                
                predicted_words.append(next_word)
                error_count += 1
                
                # Add error info to the current step
                prediction_info["predictions"].append({
                    "step": i+1,
                    "context": current_context.copy(),
                    "predicted_word": next_word,
                    "error": str(e),
                    "fallback": True,
                    "fallback_source": "exception_handler"
                })
                
                # Update context for next prediction
                current_context = current_context[1:] + [next_word]
                
        return predicted_words, prediction_info
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data for saving
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
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size,
            'use_pretrained': self.use_pretrained
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
            embedding_dim=model_data['embedding_dim'],
            hidden_layers=model_data['hidden_layers'],
            learning_rate=model_data['learning_rate'],
            n_iterations=model_data['n_iterations'],
            random_state=model_data['random_state'],
            tokenizer_type=model_data.get('tokenizer_type', 'wordpiece'),
            vocab_size=model_data.get('vocab_size', 10000),
            use_pretrained=model_data.get('use_pretrained', False)
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
        
        return model