"""
Custom Tokenizers Implementation

This module provides implementations of custom tokenizers for language modeling.
"""

import re
import collections
import numpy as np

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implementation.
    
    BPE is a subword tokenization algorithm that iteratively merges the most
    frequent pairs of characters or character sequences to form new tokens.
    """
    
    def __init__(self, vocab_size=10000, min_frequency=2):
        """
        Initialize the BPE tokenizer.
        
        Parameters:
        -----------
        vocab_size : int
            Maximum vocabulary size
        min_frequency : int
            Minimum frequency for a token to be considered
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocabulary = []
        self.word_counts = collections.Counter()
        self.merges = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    def train(self, words):
        """
        Train the BPE tokenizer on a list of words.
        
        Parameters:
        -----------
        words : list of str
            List of words to train on
        """
        # Count word frequencies
        self.word_counts = collections.Counter(words)
        
        # Initialize vocabulary with characters and special tokens
        chars = set()
        for word in self.word_counts:
            for char in word:
                chars.add(char)
        
        # Add special tokens and characters to vocabulary
        self.vocabulary = self.special_tokens.copy()
        self.vocabulary.extend(sorted(chars))
        
        # Initialize each word as a sequence of characters
        word_splits = {word: list(word) for word in self.word_counts}
        
        # Perform merges until we reach the desired vocabulary size
        while len(self.vocabulary) < self.vocab_size:
            # Count pairs
            pair_counts = collections.Counter()
            for word, freq in self.word_counts.items():
                splits = word_splits[word]
                if len(splits) == 1:
                    continue
                
                for i in range(len(splits) - 1):
                    pair = (splits[i], splits[i+1])
                    pair_counts[pair] += freq
            
            # If no more pairs, break
            if not pair_counts:
                break
            
            # Get the most frequent pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            
            # Skip pairs with low frequency
            if pair_counts[best_pair] < self.min_frequency:
                break
            
            # Create new token from the pair
            new_token = ''.join(best_pair)
            
            # Add to vocabulary
            if new_token not in self.vocabulary:
                self.vocabulary.append(new_token)
            
            # Add merge rule
            self.merges[best_pair] = new_token
            
            # Apply the merge to all words
            for word in word_splits:
                splits = word_splits[word]
                i = 0
                while i < len(splits) - 1:
                    if i < len(splits) - 1 and (splits[i], splits[i+1]) == best_pair:
                        splits[i] = new_token
                        splits.pop(i+1)
                    else:
                        i += 1
    
    def tokenize(self, word):
        """
        Tokenize a word using the trained BPE model.
        
        Parameters:
        -----------
        word : str
            Word to tokenize
            
        Returns:
        --------
        list of str
            List of tokens
        """
        # If word is empty, return empty list
        if not word:
            return []
        
        # Initialize as character sequence
        splits = list(word)
        
        # Apply merges
        i = 0
        while i < len(splits) - 1:
            pair = (splits[i], splits[i+1])
            if pair in self.merges:
                splits[i] = self.merges[pair]
                splits.pop(i+1)
                i = 0  # Start over to catch overlapping merges
            else:
                i += 1
        
        # Filter out tokens not in vocabulary
        return [token for token in splits if token in self.vocabulary]
    
    def get_vocabulary(self):
        """
        Get the vocabulary.
        
        Returns:
        --------
        list of str
            List of tokens in the vocabulary
        """
        return self.vocabulary


class WordPieceTokenizer:
    """
    WordPiece tokenizer implementation.
    
    WordPiece is a subword tokenization algorithm used in BERT and other
    transformer models. It splits words into subwords based on a greedy
    longest-match-first approach.
    """
    
    def __init__(self, vocab_size=10000, min_frequency=2):
        """
        Initialize the WordPiece tokenizer.
        
        Parameters:
        -----------
        vocab_size : int
            Maximum vocabulary size
        min_frequency : int
            Minimum frequency for a token to be considered
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocabulary = []
        self.word_counts = collections.Counter()
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.prefix = '##'
    
    def train(self, words):
        """
        Train the WordPiece tokenizer on a list of words.
        
        Parameters:
        -----------
        words : list of str
            List of words to train on
        """
        # Count word frequencies
        self.word_counts = collections.Counter(words)
        
        # Initialize vocabulary with special tokens
        self.vocabulary = self.special_tokens.copy()
        
        # Add whole words to vocabulary
        for word, count in self.word_counts.most_common(self.vocab_size - len(self.vocabulary)):
            if count >= self.min_frequency and word not in self.vocabulary:
                self.vocabulary.append(word)
        
        # If we still have room in the vocabulary, add subwords
        if len(self.vocabulary) < self.vocab_size:
            # Generate all possible subwords
            subword_counts = collections.Counter()
            for word, count in self.word_counts.items():
                # Add all prefixes
                for i in range(1, len(word) + 1):
                    prefix = word[:i]
                    if i == len(word):
                        subword_counts[prefix] += count
                    else:
                        subword_counts[prefix] += count / 2
                
                # Add all suffixes with prefix marker
                for i in range(1, len(word)):
                    suffix = self.prefix + word[i:]
                    subword_counts[suffix] += count / 2
            
            # Add most common subwords to vocabulary
            for subword, count in subword_counts.most_common(self.vocab_size - len(self.vocabulary)):
                if count >= self.min_frequency and subword not in self.vocabulary:
                    self.vocabulary.append(subword)
                
                if len(self.vocabulary) >= self.vocab_size:
                    break
    
    def tokenize(self, word):
        """
        Tokenize a word using the trained WordPiece model.
        
        Parameters:
        -----------
        word : str
            Word to tokenize
            
        Returns:
        --------
        list of str
            List of tokens
        """
        # If word is empty, return empty list
        if not word:
            return []
        
        # If word is in vocabulary, return it as a single token
        if word in self.vocabulary:
            return [word]
        
        # Try to split the word into subwords
        tokens = []
        start = 0
        while start < len(word):
            # Try to find the longest subword starting from this position
            end = len(word)
            while start < end:
                subword = word[start:end]
                if start > 0:
                    subword = self.prefix + subword
                
                if subword in self.vocabulary:
                    tokens.append(subword)
                    break
                
                end -= 1
            
            # If no subword found, use <UNK> token
            if start == end:
                tokens.append('<UNK>')
                break
            
            start = end
        
        # If no tokens were found, return <UNK>
        if not tokens:
            return ['<UNK>']
        
        return tokens
    
    def get_vocabulary(self):
        """
        Get the vocabulary.
        
        Returns:
        --------
        list of str
            List of tokens in the vocabulary
        """
        return self.vocabulary