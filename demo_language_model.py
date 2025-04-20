import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import re
import pickle
import os

class SimpleLanguageModel:
    """
    A simple language model using a single-layer perceptron for next word prediction.
    """
    
    def __init__(self, context_size=2, learning_rate=0.01, n_iterations=1000, random_state=42):
        """
        Initialize the simple language model.
        
        Parameters:
        -----------
        context_size : int
            Number of previous words to use as context for prediction
        learning_rate : float
            Learning rate for weight updates
        n_iterations : int
            Number of training iterations
        random_state : int
            Random seed for reproducibility
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
            "don't": "do not"
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
    
    def fit(self, text):
        """
        Train the language model on the given text.
        
        Parameters:
        -----------
        text : str
            The text to train on
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
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Forward pass
            y_pred = self._forward(X_train)
            
            # Calculate loss
            loss = self._cross_entropy_loss(y_pred, y_train)
            self.training_loss.append(loss)
            
            # Calculate validation loss
            y_val_pred = self._forward(X_val)
            val_loss = self._cross_entropy_loss(y_val_pred, y_val)
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Training Loss = {loss:.4f}, Validation Loss = {val_loss:.4f}")
            
            # Backward pass (gradient descent)
            # Convert y_train to one-hot encoding
            y_train_one_hot = self.encoder.transform(y_train.reshape(-1, 1))
            
            # Calculate gradients
            dZ = y_pred - y_train_one_hot
            dW = np.dot(X_train.T, dZ) / len(X_train)
            db = np.sum(dZ, axis=0) / len(X_train)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            # Shuffle the training data every 50 iterations
            if iteration % 50 == 0:
                shuffle_idx = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
    
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

def main():
    """
    Train and demonstrate the simple language model with a focused example.
    """
    # Sample text for training - simple sentences with clear patterns
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
    """
    
    # Create and train the model
    print("Creating and training the language model...")
    model = SimpleLanguageModel(context_size=2, learning_rate=0.1, n_iterations=1000)
    model.fit(sample_text)
    
    # Demonstrate prediction
    print("\nDemonstrating next word prediction:")
    
    test_contexts = [
        ["the", "cat"],
        ["the", "dog"],
        ["cats", "like"],
        ["dogs", "eat"],
        ["mice", "like"],
        ["animals", "need"],
        ["is", "important"]
    ]
    
    for context in test_contexts:
        next_word = model.predict_next_word(context)
        print(f"Context: '{' '.join(context)}' → Next word: '{next_word}'")
        
        # Show top 3 predictions
        top_predictions = model.get_top_predictions(context, top_n=3)
        print("Top 3 predictions:")
        for word, prob in top_predictions:
            print(f"  '{word}': {prob:.4f}")
        print()
    
    print("\nLanguage model training and demonstration complete.")

if __name__ == "__main__":
    main()