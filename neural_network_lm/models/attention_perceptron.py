"""
Attention-Enhanced Multi-Layer Perceptron Implementation

This module provides an implementation of a multi-layer perceptron
with self-attention mechanisms for improved language modeling capabilities.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

from .multi_layer_perceptron import MultiLayerPerceptron
from ..models.self_attention import SelfAttention

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
            True labels (one-hot encoded)
        learning_rate : float
            Learning rate for weight updates
        """
        batch_size = X.shape[0]
        
        # Calculate output layer error (derivative of cross-entropy with softmax)
        delta = y_pred - y_true  # Shape: (batch_size, output_size)
        
        # Backpropagate through dense layers
        deltas = [delta]
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            # Calculate error for previous layer
            delta = np.dot(deltas[0], self.weights[i].T)
            
            # Apply derivative of ReLU
            if i > 0:  # No need to apply activation derivative for input layer
                # Get the pre-activation values for this layer
                z = np.dot(X if i == 0 else np.dot(X, self.weights[i-1]) + self.biases[i-1], self.weights[i]) + self.biases[i]
                delta = delta * self._relu_derivative(z)
            
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            # Calculate gradients
            if i == 0:
                # For the first layer after attention, use the flattened attention output
                attention_output = self.attention_layer.forward(X)
                flattened = attention_output.reshape(batch_size, -1)
                dW = np.dot(flattened.T, deltas[i]) / batch_size
            else:
                # For other layers, use the previous layer's activations
                prev_activations = self._relu(np.dot(X if i == 1 else np.dot(X, self.weights[i-2]) + self.biases[i-2], self.weights[i-1]) + self.biases[i-1])
                dW = np.dot(prev_activations.T, deltas[i]) / batch_size
            
            db = np.sum(deltas[i], axis=0) / batch_size
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
        
        # Backpropagate through attention layer (simplified)
        # In a full implementation, we would compute gradients for attention parameters
        # and update them, but this is complex and beyond the scope of this example
    
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
                        unk_token = list(self.embeddings.special_tokens.keys())[1]  # <UNK> token
                        context[idx] = unk_token
                else:
                    # If no tokenizer, use <UNK> token
                    unk_token = list(self.embeddings.special_tokens.keys())[1]  # <UNK> token
                    context[idx] = unk_token
            
            info["unknown_words"] = [word for _, word in unknown_words]
            info["adjustment_made"] = True
            info["adjustment_type"] = info.get("adjustment_type", "") + "_replaced_unknown"
        
        try:
            # Get embeddings for context words
            context_embeddings = []
            for word in context:
                word_idx = self.word_to_idx.get(word, self.embeddings.special_tokens['<UNK>'])
                embedding = self.embeddings.embeddings[word_idx]
                context_embeddings.append(embedding)
            
            # Convert to numpy array and add batch dimension
            context_embeddings = np.array([context_embeddings])  # shape: (1, context_size, embedding_dim)
            
            # Forward pass with attention
            y_pred, attention_weights = self._forward_with_attention(context_embeddings)
            
            # Get the word with the highest probability
            predicted_idx = np.argmax(y_pred[0])
            predicted_word = self.idx_to_word[predicted_idx]
            
            # Get top 5 predictions with probabilities
            top_indices = np.argsort(y_pred[0])[-5:][::-1]
            top_probs = {self.idx_to_word[idx]: float(y_pred[0][idx]) for idx in top_indices}
            
            # Add prediction info
            info["prediction"] = predicted_word
            info["probabilities"] = top_probs
            info["attention_weights"] = attention_weights[0].tolist()  # Convert to list for JSON serialization
            
            return predicted_word, info
            
        except Exception as e:
            info["error"] = str(e)
            return "<ERROR>", info
    
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
            # Pad with empty strings
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
        
        # Predict n words
        predicted_words = []
        for i in range(n):
            try:
                # Check if embeddings object exists
                if self.embeddings is None:
                    print(f"[AttentionPerceptron.predict_next_n_words] ERROR: embeddings is None, initializing")
                    # Initialize embeddings
                    from ..utils.embeddings import WordEmbeddings
                    self.embeddings = WordEmbeddings(embedding_dim=self.embedding_dim, random_state=self.random_state)
                    # Add special tokens
                    self.embeddings.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
                    # Initialize word_to_idx if needed
                    if not hasattr(self, 'word_to_idx') or self.word_to_idx is None:
                        self.word_to_idx = {}
                
                # Predict next word with temperature
                next_word, word_info = self._predict_next_word_with_temperature(current_context, temperature)
                predicted_words.append(next_word)
                
                # Store prediction info
                prediction_info["predictions"].append({
                    "step": i+1,
                    "context": current_context.copy(),
                    "predicted_word": next_word,
                    "probabilities": word_info.get("probabilities", {}),
                    "attention_weights": word_info.get("attention_weights", [])
                })
                
                # Update context for next prediction
                current_context = current_context[1:] + [next_word]
                
            except Exception as e:
                print(f"Error in prediction step {i+1}: {str(e)}")
                prediction_info["error"] = str(e)
                prediction_info["error_step"] = i+1
                break
                
        return predicted_words, prediction_info
    
    def plot_attention_weights(self, context, ax=None):
        """
        Plot attention weights for a given context.
        
        Parameters:
        -----------
        context : list of str or str
            Context words
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes with the plot
        """
        import matplotlib.pyplot as plt
        
        # Handle string input
        if isinstance(context, str):
            context = context.split()
        
        # Get prediction and attention weights
        _, info = self.predict_next_word(context)
        
        # Extract attention weights
        if "attention_weights" not in info:
            raise ValueError("No attention weights available for this context")
            
        attention_weights = np.array(info["attention_weights"])
        
        # Create figure if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Plot attention weights as a heatmap
        im = ax.imshow(attention_weights, cmap="YlOrRd")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        ax.set_xticks(np.arange(len(context)))
        ax.set_yticks(np.arange(len(context)))
        ax.set_xticklabels(context)
        ax.set_yticklabels(context)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        predicted_word = info.get("prediction", "unknown")
        ax.set_title(f"Attention Weights (Predicted: '{predicted_word}')")
        
        # Add text annotations
        for i in range(len(context)):
            for j in range(len(context)):
                ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                       ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white")
        
        ax.set_xlabel("Key Words")
        ax.set_ylabel("Query Words")
        
        plt.tight_layout()
        
        return ax
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
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