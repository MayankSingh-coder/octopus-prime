"""
Word Embeddings Implementation

This module provides an implementation of word embeddings for language modeling.
"""

import numpy as np
import os
import pickle

class WordEmbeddings:
    """
    Word embeddings implementation for language modeling.
    
    This class provides functionality for creating and managing word embeddings,
    which are vector representations of words that capture semantic relationships.
    """
    
    def __init__(self, embedding_dim=50, random_state=42, use_pretrained=False, pretrained_path=None):
        """
        Initialize word embeddings.
        
        Parameters:
        -----------
        embedding_dim : int
            Dimensionality of word embeddings
        random_state : int
            Random seed for reproducibility
        use_pretrained : bool
            Whether to use pretrained embeddings
        pretrained_path : str, optional
            Path to pretrained embeddings file
        """
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.use_pretrained = use_pretrained
        self.pretrained_path = pretrained_path
        self.embeddings = {}
        self.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)
        
        # Initialize special token embeddings
        for token in self.special_tokens:
            self.embeddings[self.special_tokens[token]] = self.rng.normal(0, 0.1, embedding_dim)
        
        # Load pretrained embeddings if specified
        if use_pretrained:
            self.load_pretrained_embeddings(pretrained_path)
    
    def add_word(self, word, word_idx=None):
        """
        Add a word to the embeddings.
        
        Parameters:
        -----------
        word : str
            The word to add
        word_idx : int, optional
            Index for the word. If None, a new index is assigned.
            
        Returns:
        --------
        int
            Index of the added word
        """
        # If word_idx is not provided, use the current size of embeddings
        if word_idx is None:
            word_idx = len(self.embeddings)
        
        # Check if word already has an embedding
        if word_idx in self.embeddings:
            return word_idx
        
        # Create a random embedding for the word
        self.embeddings[word_idx] = self.rng.normal(0, 0.1, self.embedding_dim)
        
        return word_idx
    
    def get_embedding(self, word_idx):
        """
        Get the embedding for a word by its index.
        
        Parameters:
        -----------
        word_idx : int
            Index of the word
            
        Returns:
        --------
        numpy.ndarray
            Embedding vector for the word
        """
        # If word_idx is not in embeddings, return the <UNK> embedding
        if word_idx not in self.embeddings:
            return self.embeddings[self.special_tokens['<UNK>']]
        
        return self.embeddings[word_idx]
    
    def load_pretrained_embeddings(self, path=None):
        """
        Load pretrained word embeddings from a file.
        
        Parameters:
        -----------
        path : str, optional
            Path to the pretrained embeddings file
        """
        # If path is not provided, use a default path
        if path is None:
            # Check if we have a default path
            if self.pretrained_path is None:
                print("No path provided for pretrained embeddings. Using random embeddings.")
                return
            path = self.pretrained_path
        
        # Check if the file exists
        if not os.path.exists(path):
            print(f"Pretrained embeddings file not found at {path}. Using random embeddings.")
            return
        
        # Load embeddings based on file format
        if path.endswith('.pkl'):
            self._load_pickle_embeddings(path)
        elif path.endswith('.txt') or path.endswith('.vec'):
            self._load_text_embeddings(path)
        else:
            print(f"Unsupported embeddings file format: {path}. Using random embeddings.")
    
    def _load_pickle_embeddings(self, path):
        """
        Load embeddings from a pickle file.
        
        Parameters:
        -----------
        path : str
            Path to the pickle file
        """
        try:
            with open(path, 'rb') as f:
                loaded_embeddings = pickle.load(f)
            
            # Update embeddings with loaded ones
            self.embeddings.update(loaded_embeddings)
            print(f"Loaded {len(loaded_embeddings)} pretrained embeddings from {path}")
        except Exception as e:
            print(f"Error loading pretrained embeddings from {path}: {str(e)}")
    
    def _load_text_embeddings(self, path):
        """
        Load embeddings from a text file.
        
        Parameters:
        -----------
        path : str
            Path to the text file
        """
        try:
            word_to_idx = {}
            loaded_embeddings = {}
            
            with open(path, 'r', encoding='utf-8') as f:
                # Skip header if present
                first_line = f.readline().strip().split()
                if len(first_line) == 2:
                    # This is a header with vocab_size and embedding_dim
                    vocab_size, embedding_dim = map(int, first_line)
                    if embedding_dim != self.embedding_dim:
                        print(f"Warning: Pretrained embedding dimension ({embedding_dim}) does not match specified dimension ({self.embedding_dim})")
                else:
                    # This is the first embedding, go back to the beginning
                    f.seek(0)
                
                # Read embeddings
                idx = len(self.embeddings)
                for line in f:
                    parts = line.strip().split()
                    if len(parts) <= self.embedding_dim:
                        continue  # Skip lines with insufficient data
                    
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:self.embedding_dim+1]])
                    
                    # Normalize vector
                    vector = vector / np.linalg.norm(vector)
                    
                    word_to_idx[word] = idx
                    loaded_embeddings[idx] = vector
                    idx += 1
            
            # Update embeddings with loaded ones
            self.embeddings.update(loaded_embeddings)
            print(f"Loaded {len(loaded_embeddings)} pretrained embeddings from {path}")
        except Exception as e:
            print(f"Error loading pretrained embeddings from {path}: {str(e)}")
    
    def save_embeddings(self, path):
        """
        Save embeddings to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the embeddings
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save embeddings based on file format
        if path.endswith('.pkl'):
            with open(path, 'wb') as f:
                pickle.dump(self.embeddings, f)
        elif path.endswith('.txt') or path.endswith('.vec'):
            with open(path, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"{len(self.embeddings)} {self.embedding_dim}\n")
                
                # Write embeddings
                for idx, embedding in self.embeddings.items():
                    # Convert idx to word if we have a reverse mapping
                    word = str(idx)  # Default to index as string
                    
                    # Write word and embedding
                    vector_str = ' '.join([str(x) for x in embedding])
                    f.write(f"{word} {vector_str}\n")
        else:
            print(f"Unsupported file format for saving embeddings: {path}")
            return False
        
        return True