import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import pickle
import os
from embeddings import WordEmbeddings

class SimpleLanguageModel:
    """
    A simple language model using a single-layer perceptron for next word prediction.
    """
    
    def __init__(self, context_size=2, embedding_dim=50, learning_rate=0.01, n_iterations=1000, random_state=42):
        """
        Initialize the simple language model.
        
        Parameters:
        -----------
        context_size : int
            Number of previous words to use as context for prediction
        embedding_dim : int
            Dimensionality of word embeddings
        learning_rate : float
            Learning rate for weight updates
        n_iterations : int
            Number of training iterations
        random_state : int
            Random seed for reproducibility
        """
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.vocabulary = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.embeddings = None
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
        
        # Remove remaining special characters and replace with space
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        
        return words
    
    def _build_vocabulary(self, words):
        """
        Build the vocabulary from the list of words and initialize embeddings.
        """
        # Initialize word embeddings
        self.embeddings = WordEmbeddings(embedding_dim=self.embedding_dim, random_state=self.random_state)
        
        # Build vocabulary and initialize embeddings
        self.embeddings.build_vocabulary(words)
        
        # Get vocabulary mappings
        self.word_to_idx = self.embeddings.word_to_idx
        self.idx_to_word = self.embeddings.idx_to_word
        
        # Store vocabulary
        self.vocabulary = self.embeddings.vocabulary
        self.output_size = len(self.vocabulary) + len(self.embeddings.special_tokens)
    
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
        # Check if input dimensions match the weights
        if X.shape[1] != self.weights.shape[0]:
            print(f"Warning: Input dimension mismatch. Expected {self.weights.shape[0]}, got {X.shape[1]}.")
            
            # Resize input to match expected dimension
            if X.shape[1] > self.weights.shape[0]:
                # Truncate if larger
                X = X[:, :self.weights.shape[0]]
                print(f"Truncated input to shape {X.shape}")
            else:
                # Pad with zeros if smaller
                padding = np.zeros((X.shape[0], self.weights.shape[0] - X.shape[1]))
                X = np.concatenate([X, padding], axis=1)
                print(f"Padded input to shape {X.shape}")
        
        # Calculate the net input
        z = np.dot(X, self.weights) + self.bias
        
        # Apply softmax activation
        a = self._softmax(z)
        
        return a
    
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
        self.input_size = X.shape[1]  # This will be context_size * embedding_dim
        # Initialize with slightly larger weights for better gradient flow
        self.weights = np.random.normal(0, 0.1, (self.input_size, self.output_size))
        self.bias = np.zeros(self.output_size)
        
        # Adaptive learning rate
        initial_learning_rate = self.learning_rate
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 50  # Number of iterations to wait for improvement
        patience_counter = 0
        
        for iteration in range(self.n_iterations):
            # Adaptive learning rate - decrease over time
            current_lr = initial_learning_rate / (1 + iteration / 200)
            
            # Forward pass
            y_pred = self._forward(X_train)
            
            # Calculate loss
            loss = self._cross_entropy_loss(y_pred, y_train)
            self.training_loss.append(loss)
            
            # Calculate validation loss
            y_val_pred = self._forward(X_val)
            val_loss = self._cross_entropy_loss(y_val_pred, y_val)
            
            # Store validation loss and iteration count
            self.validation_loss.append(val_loss)
            self.iteration_count.append(iteration)
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Training Loss = {loss:.4f}, Validation Loss = {val_loss:.4f}, LR = {current_lr:.6f}")
            
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
                print(f"Early stopping at iteration {iteration}")
                # Restore best weights
                self.weights = best_weights
                self.bias = best_bias
                break
            
            # Backward pass (gradient descent)
            # Create one-hot vectors for true labels
            y_train_one_hot = np.zeros((len(y_train), self.output_size))
            for i, label in enumerate(y_train):
                y_train_one_hot[i, label] = 1
            
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
        unknown_words = []
        for word in context:
            if word not in self.word_to_idx:
                unknown_words.append(word)
                print(f"Warning: Word '{word}' not in vocabulary. Using <UNK> token instead.")
        
        # Replace unknown words with <UNK> token
        if unknown_words:
            context = [word if word not in unknown_words else '<UNK>' for word in context]
        
        # Get embeddings for context words and concatenate
        context_vector = self.embeddings.get_embeddings_for_context(context)
        
        # Check if context vector has the expected size
        expected_size = self.context_size * self.embedding_dim
        if context_vector.shape[0] != expected_size:
            print(f"Warning: Context vector dimension mismatch. Expected {expected_size}, got {context_vector.shape[0]}.")
            
            # Resize context vector to match expected dimension
            if context_vector.shape[0] > expected_size:
                # Truncate if larger
                context_vector = context_vector[:expected_size]
                print(f"Truncated context vector to length {context_vector.shape[0]}")
            else:
                # Pad with zeros if smaller
                padding = np.zeros(expected_size - context_vector.shape[0])
                context_vector = np.concatenate([context_vector, padding])
                print(f"Padded context vector to length {context_vector.shape[0]}")
        
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
        unknown_words = []
        for word in context:
            if word not in self.word_to_idx:
                unknown_words.append(word)
                print(f"Warning: Word '{word}' not in vocabulary. Using <UNK> token instead.")
        
        # Replace unknown words with <UNK> token
        if unknown_words:
            context = [word if word not in unknown_words else '<UNK>' for word in context]
        
        # Get embeddings for context words and concatenate
        context_vector = self.embeddings.get_embeddings_for_context(context)
        
        # Check if context vector has the expected size
        expected_size = self.context_size * self.embedding_dim
        if context_vector.shape[0] != expected_size:
            print(f"Warning: Context vector dimension mismatch. Expected {expected_size}, got {context_vector.shape[0]}.")
            
            # Resize context vector to match expected dimension
            if context_vector.shape[0] > expected_size:
                # Truncate if larger
                context_vector = context_vector[:expected_size]
                print(f"Truncated context vector to length {context_vector.shape[0]}")
            else:
                # Pad with zeros if smaller
                padding = np.zeros(expected_size - context_vector.shape[0])
                context_vector = np.concatenate([context_vector, padding])
                print(f"Padded context vector to length {context_vector.shape[0]}")
        
        # Forward pass
        y_pred = self._forward(context_vector.reshape(1, -1))[0]
        
        # Get the top n predictions
        top_indices = np.argsort(y_pred)[-top_n:][::-1]
        top_probs = y_pred[top_indices]
        
        # Convert to words
        top_words = [self.idx_to_word[idx] for idx in top_indices]
        
        return list(zip(top_words, top_probs))
    
    def plot_training_loss(self):
        """
        Plot the training loss over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.training_loss) + 1), self.training_loss)
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Training Loss Over Iterations')
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
            'embedding_dim': self.embedding_dim,  # Save embedding dimension explicitly
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
        
        # Get embedding dimension from saved model data
        # If not available, calculate from input size and context size or use default
        if 'embedding_dim' in model_data:
            embedding_dim = model_data['embedding_dim']
        elif 'input_size' in model_data and 'context_size' in model_data:
            # Calculate embedding_dim if possible
            if model_data['context_size'] > 0:
                embedding_dim = model_data['input_size'] // model_data['context_size']
        else:
            embedding_dim = 50  # Default fallback
        
        # Create a new instance with the correct embedding dimension
        model = cls(
            context_size=model_data['context_size'],
            embedding_dim=embedding_dim,
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
        model.training_loss = model_data['training_loss']
        model.validation_loss = model_data.get('validation_loss', [])
        model.iteration_count = model_data.get('iteration_count', [])
        
        # Recreate the embeddings with the correct dimension
        model.embedding_dim = embedding_dim
        model.embeddings = WordEmbeddings(embedding_dim=embedding_dim, random_state=model.random_state)
        model.embeddings.vocabulary = model.vocabulary
        model.embeddings.word_to_idx = model.word_to_idx
        model.embeddings.idx_to_word = model.idx_to_word
        
        # Initialize embeddings with the correct size
        vocab_size = len(model.vocabulary) + len(model.embeddings.special_tokens)
        model.embeddings.embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
        # Make sure the embeddings are properly initialized
        # This ensures that get_embeddings_for_context will return vectors of the right dimension
        for token, idx in model.embeddings.special_tokens.items():
            if token == '<PAD>':
                model.embeddings.embeddings[idx] = np.zeros(embedding_dim)
            else:
                model.embeddings.embeddings[idx] = np.random.normal(0, 0.01, embedding_dim)
        
        return model


def main():
    """
    Train and demonstrate the simple language model.
    """
    # Sample text for training - expanded with more diverse content
    sample_text = """
    The quick brown fox jumps over the lazy dog. 
    A watched pot never boils. 
    Actions speak louder than words. 
    All that glitters is not gold. 
    Better late than never. 
    Birds of a feather flock together. 
    Cleanliness is next to godliness. 
    Don't count your chickens before they hatch. 
    Don't put all your eggs in one basket. 
    Early to bed and early to rise makes a man healthy, wealthy, and wise. 
    Easy come, easy go. 
    Every cloud has a silver lining. 
    Fortune favors the bold. 
    Haste makes waste. 
    Honesty is the best policy. 
    Hope for the best, prepare for the worst. 
    If it ain't broke, don't fix it. 
    It takes two to tango. 
    Keep your friends close and your enemies closer. 
    Laughter is the best medicine. 
    Let sleeping dogs lie. 
    Look before you leap. 
    No pain, no gain. 
    Practice makes perfect. 
    Rome wasn't built in a day. 
    The early bird catches the worm. 
    The pen is mightier than the sword. 
    Time is money. 
    Two wrongs don't make a right. 
    When in Rome, do as the Romans do. 
    You can't have your cake and eat it too. 
    You can't judge a book by its cover. 
    You can't teach an old dog new tricks. 
    You reap what you sow.
    
    Artificial intelligence is transforming how we live and work. Machine learning models can recognize patterns in data that humans might miss. Neural networks are inspired by the human brain but operate quite differently. Deep learning has revolutionized fields like computer vision and natural language processing.
    
    The sun rises in the east and sets in the west. Water boils at 100 degrees Celsius at sea level. Plants need sunlight and water to grow. The Earth orbits around the Sun once every 365.25 days.
    
    Programming languages like Python and JavaScript are widely used today. Good code should be readable and maintainable. Software development often involves teamwork and collaboration. Testing is an essential part of the development process.
    
    The human brain contains approximately 86 billion neurons. Our memories are stored across different regions of the brain. Sleep is crucial for cognitive function and memory consolidation. Exercise has been shown to improve brain health.
    
    Books can transport us to different worlds and times. Reading regularly improves vocabulary and comprehension. Libraries provide access to knowledge for everyone. Some of the most influential ideas in history were first written in books.
    
    Music has the power to evoke strong emotions. Different cultures have developed unique musical traditions. Learning to play an instrument can improve cognitive abilities. Rhythm and harmony are fundamental elements of music.
    
    Healthy eating involves consuming a variety of nutrients. Vegetables and fruits contain essential vitamins and minerals. Protein is necessary for building and repairing tissues. Staying hydrated is important for overall health.
    
    Climate change is affecting ecosystems worldwide. Renewable energy sources include solar, wind, and hydroelectric power. Conservation efforts are crucial for protecting biodiversity. Sustainable practices can help reduce our environmental impact.
    
    The internet has revolutionized how we communicate and access information. Social media platforms connect people across the globe. Digital literacy is becoming increasingly important in today's world. Online privacy and security are major concerns.
    
    Space exploration has led to numerous technological advances. The International Space Station orbits the Earth every 90 minutes. Mars has been a target for exploration due to its potential to have once harbored life. Telescopes allow us to observe distant galaxies and stars.
    """
    
    # Create and train the model
    print("Creating and training the language model...")
    model = SimpleLanguageModel(context_size=2, learning_rate=0.1, n_iterations=2000)
    model.fit(sample_text)
    
    # Plot the training loss
    plt_loss = model.plot_training_loss()
    plt_loss.savefig('language_model_loss.png')
    
    # Save the model
    model.save_model('simple_language_model.pkl')
    
    # Demonstrate prediction
    print("\nDemonstrating next word prediction:")
    
    test_contexts = [
        ["the", "quick"],
        ["over", "the"],
        ["is", "the"],
        ["early", "to"],
        ["you", "can't"]
    ]
    
    for context in test_contexts:
        next_word = model.predict_next_word(context)
        print(f"Context: '{' '.join(context)}' → Next word: '{next_word}'")
        
        # Show top 3 predictions
        top_predictions = model.get_top_predictions(context, top_n=3)
        print("Top 3 predictions:")
        for word, prob in top_predictions:
            print(f"  '{word}': {prob:.4f}")
    
    # Demonstrate generating a sequence
    print("\nGenerating sequences:")
    
    test_initial_contexts = [
        ["the", "quick"],
        ["honesty", "is"],
        ["time", "is"]
    ]
    
    for context in test_initial_contexts:
        sequence = model.predict_next_n_words(context, n=5)
        print(f"Initial context: '{' '.join(context)}' → Generated: '{' '.join(sequence)}'")
        print(f"Full sequence: '{' '.join(context)} {' '.join(sequence)}'")
    
    print("\nLanguage model training and demonstration complete.")
    print("You can now use this model for your own text data.")
    print("To train on your own data, modify the sample_text variable or load text from a file.")
    
    plt.show()


if __name__ == "__main__":
    main()